# 人脸情感分类模型：ConvNeXt-Tiny + CBAM 注意力 + 迁移学习
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import timm


# 通道注意力（SENet 思想）
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


# CBAM：通道注意力 + 空间注意力
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class FERModel(nn.Module):
    """ConvNeXt-Tiny backbone + CBAM + 分类头"""

    def __init__(self, num_classes=6):
        super().__init__()
        # 加载 ConvNeXt-Tiny，去掉分类头，只保留特征提取部分
        self.backbone = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        self.num_features = self.backbone.num_features  # 特征通道数

        # CBAM 注意力（加在特征图之后、全局池化之前）
        self.cbam = CBAM(self.num_features)

        # 分类头
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, num_classes),
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)  # 得到特征图 [B, C, H, W]
        x = self.cbam(x)                        # 施加注意力
        feat = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)  # 全局平均池化 -> embedding [B, C]
        logits = self.head(feat)                # 分类
        return logits, feat                     # 返回 logits + 特征（供 Center Loss 使用）


def create_model(num_classes=6):
    return FERModel(num_classes=num_classes)
