import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from .decoding import decode_heatmap


class Block(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, channels), nn.SiLU(inplace=True))


class Localizer(nn.Module):
    """Stride-4 FPN; heatmap, auxiliary box mask, optional presence logits."""
    def __init__(self, backbone='resnet34', pretrained=False, channels=96):
        super().__init__()
        self.backbone_name = backbone
        self.freeze_bn = pretrained
        if backbone in ('resnet18', 'resnet34'):
            weights = getattr(models, 'ResNet18_Weights' if backbone=='resnet18' else 'ResNet34_Weights').DEFAULT if pretrained else None
            args = {} if pretrained else {'norm_layer': lambda c: nn.GroupNorm(32, c)}
            self.encoder = getattr(models, backbone)(weights=weights, **args)
            self.encoder.fc = nn.Identity()
            encoder_channels = [64, 128, 256, 512]
        elif backbone == 'convnext_tiny':
            self.encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None).features
            encoder_channels = [96, 192, 384, 768]
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
        self.lateral = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in encoder_channels)
        self.smooth = nn.ModuleList(Block(channels) for _ in encoder_channels[:-1])
        self.heatmap = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 1, 1)
        self.presence = nn.Linear(encoder_channels[-1], 1)
        nn.init.normal_(self.heatmap.weight, std=0.001)
        nn.init.zeros_(self.heatmap.bias)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        if self.backbone_name.startswith('resnet'):
            e = self.encoder
            x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
            features=[]
            for layer in [e.layer1,e.layer2,e.layer3,e.layer4]:
                x = layer(x)
                features.append(x)
        else:
            features=[]
            for i,layer in enumerate(self.encoder):
                x = layer(x)
                if i in (1,3,5,7):
                    features.append(x)
        pyramid = self.lateral[-1](features[-1])
        for i in range(2,-1,-1):
            pyramid = self.smooth[i](self.lateral[i](features[i]) + F.interpolate(pyramid, size=features[i].shape[-2:], mode='bilinear', align_corners=False))
        logits = self.heatmap(pyramid).float()
        xy = decode_heatmap(logits)
        return dict(logits=logits, xy=xy, box=self.box(pyramid).float(),
                    presence=self.presence(features[-1].mean((-2,-1))).squeeze(1).float())
