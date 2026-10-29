"""Backbones with a single regression output."""
import torch.nn as nn
from torchvision import models


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


class _Block(nn.Module):
    """Residual block of the paper's small ResNet, in 2D: two 3x3 convs + 1x1 shortcut, ELU."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
        )
        self.shortcut = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.act = nn.ELU()
        self.pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return self.pool(self.act(self.body(x) + self.shortcut(x)))


class SmallResNet(nn.Module):
    """2D version of the paper's ResNet: 5 blocks, 8->128 filters doubling after each pool."""

    def __init__(self, in_channels=3):
        super().__init__()
        chans = [in_channels, 8, 16, 32, 64, 128]
        self.features = nn.Sequential(*[_Block(chans[i], chans[i + 1]) for i in range(5)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(128 * 4, 64), nn.ELU(), nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        return self.head(self.features(x))


def build_model(args):
    if args.model == 'small_resnet':
        return SmallResNet()
    return _torchvision_resnet(args.model, args.pretrained)
