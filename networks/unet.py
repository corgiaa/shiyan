# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math


def pjoin(*args):
    """Custom path join function that always uses forward slash for checkpoint keys."""
    return "/".join(args)


import torch
import torch.nn as nn
import numpy as np

# 引入U-Net所需的额外层，如MaxPool2d, ConvTranspose2d, Identity
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, MaxPool2d, ConvTranspose2d, Identity
from torch.nn.modules.utils import _pair
from scipy import ndimage
# 保留原始的ViT配置导入，尽管U-Net结构不会使用其大部分参数
from . import vit_seg_configs as configs

# 移除ViT混合编码器中使用的ResNetV2导入，因为U-Net不使用它
# from .vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


# --- 原始文件中保留的通用工具函数 ---
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# --- U-Net 特定的构建块 ---

class Conv2dReLU(nn.Sequential):
    """
    一个辅助块，结合了Conv2d、BatchNorm2d（可选）和ReLU激活。
    在U-Net中，这种模式很常见。
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),  # 当使用BatchNorm时，通常省略偏置
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DoubleConv(nn.Module):
    """
    U-Net标准的双卷积块：(Conv => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, use_batchnorm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv2dReLU(in_channels, mid_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(mid_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    U-Net的下采样块：MaxPool2d 后接一个 DoubleConv。
    """

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    U-Net的上采样块：上采样（双线性插值或ConvTranspose2d）后，
    与跳跃连接特征拼接，再接一个 DoubleConv。
    """

    def __init__(self, in_channels, out_channels, use_batchnorm=True, bilinear=True):  # 修正参数顺序
        super().__init__()

        # 如果使用双线性插值，则通过常规卷积减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # DoubleConv的输入通道将是 `in_channels` (来自前一层) + `skip_channels` (上采样后为 `in_channels // 2`)
            # 因此，DoubleConv的有效 `mid_channels` 为 `in_channels`
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_batchnorm=use_batchnorm)
        else:
            # ConvTranspose2d 会自行将通道数减半
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # DoubleConv的输入通道将是 `in_channels` (来自前一层，ConvTranspose2d后为 `in_channels // 2`) + `skip_channels`
            self.conv = DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 如果尺寸不匹配（例如由于奇数/偶数维度），则对x1进行填充以匹配x2的尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])

        # 将上采样后的特征图与跳跃连接拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetModified(nn.Module):
    """
    主U-Net架构。
    它接收输入图像并输出与原始分辨率相同的特征图，
    这些特征图随后传递给分割头。
    """

    def __init__(self, in_channels, num_intermediate_channels, bilinear=True, use_batchnorm=True):
        super(UNetModified, self).__init__()
        self.in_channels = in_channels
        self.num_intermediate_channels = num_intermediate_channels  # 最后一个解码器块的输出通道数
        self.bilinear = bilinear
        self.use_batchnorm = use_batchnorm

        # 编码器路径
        self.inc = DoubleConv(in_channels, 64, use_batchnorm=use_batchnorm)
        self.down1 = Down(64, 128, use_batchnorm=use_batchnorm)
        self.down2 = Down(128, 256, use_batchnorm=use_batchnorm)
        self.down3 = Down(256, 512, use_batchnorm=use_batchnorm)

        # 瓶颈层
        factor = 2 if bilinear else 1  # 如果使用双线性上采样，则最深层通道数减半的因子
        self.down4 = Down(512, 1024 // factor, use_batchnorm=use_batchnorm)  # 最深层

        # 解码器路径
        # 修正：调整 Up 构造函数中参数的顺序
        self.up1 = Up(1024, 512 // factor, use_batchnorm, bilinear)
        self.up2 = Up(512, 256 // factor, use_batchnorm, bilinear)
        self.up3 = Up(256, 128 // factor, use_batchnorm, bilinear)
        self.up4 = Up(128, num_intermediate_channels, use_batchnorm, bilinear)  # 最终解码器阶段输出到SegmentationHead

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # 瓶颈层特征

        # 带跳跃连接的解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)  # 最终分类前的输出特征
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class VisionTransformer(nn.Module):
    """
    此类别已重构为实现U-Net架构，同时保持原始类名和forward方法签名。
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        # self.classifier = config.classifier # ViT特定参数，此处不使用。

        in_channels = 3
        unet_intermediate_channels = 64

        self.unet_model = UNetModified(
            in_channels=in_channels,
            num_intermediate_channels=unet_intermediate_channels,
            bilinear=True,
            use_batchnorm=True
        )

        self.segmentation_head = SegmentationHead(
            in_channels=unet_intermediate_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=1,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features_before_segmentation_head = self.unet_model(x)

        logits = self.segmentation_head(features_before_segmentation_head)
        return logits

    def load_from(self, weights):
        logger.warning(
            "The 'load_from' method is specific to Vision Transformer weights and is not applicable to the U-Net architecture. Skipping weight loading.")
        pass


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
