"""ImageNet ResNet backbones with a single regression output (the `fc` layer)."""
import torch.nn as nn
from torchvision import models

BUILDERS = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def build_model(args):
    builder, weights = BUILDERS[args.model]
    net = builder(weights=weights if args.pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net
