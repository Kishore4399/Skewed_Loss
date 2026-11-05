"""Backbones with a single regression output (2D for images, 3D for brain MRI)."""
import torch.nn as nn
from torchvision import models

MODELS_2D = ('resnet18', 'resnet34', 'resnet50', 'small_resnet')
MODELS_3D = ('resnet18_3d', 'small_resnet3d')


def _torchvision_resnet(name, pretrained):
    builders = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    }
    builder, weights = builders[name]
    net = builder(weights=weights if pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


_LAYERS = {2: (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d),
           3: (nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d)}


class _Block(nn.Module):
    """Residual block of the paper's small ResNet: two 3x3(x3) convs + 1x1 shortcut, ELU, max-pool."""

    def __init__(self, c_in, c_out, dims=2):
        super().__init__()
        Conv, BN, Pool, _ = _LAYERS[dims]
        self.body = nn.Sequential(
            Conv(c_in, c_out, 3, padding=1, bias=False), BN(c_out), nn.ELU(),
            Conv(c_out, c_out, 3, padding=1, bias=False), BN(c_out),
        )
        self.shortcut = Conv(c_in, c_out, 1, bias=False)
        self.act = nn.ELU()
        self.pool = Pool(3, stride=2)

    def forward(self, x):
        return self.pool(self.act(self.body(x) + self.shortcut(x)))


class SmallResNet(nn.Module):
    """The paper's ResNet: 5 blocks, 8->128 filters doubling after each pool.

    dims=3 is the original 3D MRI network; dims=2 is the version for face images.
    """

    def __init__(self, in_channels=3, dims=2):
        super().__init__()
        _, _, _, AvgPool = _LAYERS[dims]
        chans = [in_channels, 8, 16, 32, 64, 128]
        self.features = nn.Sequential(*[_Block(chans[i], chans[i + 1], dims) for i in range(5)])
        self.head = nn.Sequential(
            AvgPool(2), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(128 * 2 ** dims, 64), nn.ELU(), nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        return self.head(self.features(x))


class MonaiResNet3D(nn.Module):
    """3D ResNet-18 from MONAI with a linear head, the brain-age network of UnbiasedBrainAge."""

    def __init__(self, in_channels=1, feature_dim=512):
        super().__init__()
        from monai.networks.nets import resnet18
        self.backbone = resnet18(spatial_dims=3, n_input_channels=in_channels, feed_forward=False,
                                 shortcut_type='A', pretrained=False)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def build_model(args):
    modality = getattr(args, 'modality', 'image')
    allowed = MODELS_3D if modality == 'mri' else MODELS_2D
    if args.model not in allowed:
        raise ValueError(f'--model {args.model} does not fit --modality {modality}; choose one of {allowed}')
    if args.model == 'small_resnet':
        return SmallResNet()
    if args.model == 'small_resnet3d':
        return SmallResNet(in_channels=1, dims=3)
    if args.model == 'resnet18_3d':
        return MonaiResNet3D()
    return _torchvision_resnet(args.model, args.pretrained)
