import torch.nn as nn
from torchvision import models


def build_model(cfg):
    pretrained = bool(cfg.get("pretrained", True))
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

    model = models.resnet18(weights=weights)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    # Train only the classifier
    for p in model.fc.parameters():
        p.requires_grad = True

    return model
