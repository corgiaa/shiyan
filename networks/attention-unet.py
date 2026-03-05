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


class Attention_block(nn.Module):
    """
    Attention Gate (注意力门) 模块。
    它接收来自解码器的 gating signal (g) 和来自编码器的 skip connection (x_skip)。
    通过学习一个注意力系数，对 x_skip 进行加权，使其关注 g 所指示的相关区域。
    """

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g (int): Gating signal (g) 的通道数。
            F_l (int): Skip connection (x_skip) 的通道数。
            F_int (int): 中间层的通道数，通常是 F_l 的一半或 F_g 的一半。
        """
        super(Attention_block, self).__init__()
        # 对 gating signal 进行 1x1 卷积，调整通道数到 F_int
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 对 skip connection 进行 1x1 卷积，调整通道数到 F_int
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 结合 W_g 和 W_x 的输出，通过 ReLU 激活，再进行 1x1 卷积到单通道，
        # 经过 BatchNorm 和 Sigmoid 得到注意力系数。
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x_skip):
        """
        Args:
            g (torch.Tensor): Gating signal (来自解码器上采样后的特征图)。
            x_skip (torch.Tensor): Skip connection (来自编码器对应层的特征图)。
        Returns:
            torch.Tensor: 经过注意力加权后的 skip connection 特征图。
        """
        # 对 gating signal 进行处理
        g1 = self.W_g(g)
        # 对 skip connection 进行处理
        x1 = self.W_x(x_skip)

        # 确保 g1 和 x1 具有相同的空间尺寸，以便相加
        # 如果 g1 和 x1 空间尺寸不同，需要调整其中一个。
        # 在U-Net中，g通常是上采样后的，所以其空间尺寸应该与x_skip匹配或略小。
        # 如果不匹配，这里需要进行插值或裁剪。
        # 假设在Up模块中已经处理了空间尺寸匹配问题，这里直接相加。

        # 计算注意力权重
        psi = self.psi(g1 + x1)

        # 将注意力权重上采样到 x_skip 的空间尺寸（如果需要）
        # 注意力门论文中通常会将psi上采样到x_skip的尺寸，这里假设psi已经与x_skip尺寸匹配
        # 或者通过卷积操作隐式处理了尺寸。
        # 实际上，W_g和W_x的1x1卷积不改变空间尺寸，所以g1和x1应该在进入Attention_block前就匹配好尺寸。
        # 在Up模块中，我们确保了g (x1_up) 和 x_skip (x2) 在pad后空间尺寸一致，
        # 这样W_g(g)和W_x(x_skip)的输出g1和x1空间尺寸也一致。

        # 将注意力权重应用于 skip connection
        return x_skip * psi


class Up(nn.Module):
    """
    Attention U-Net的上采样块：上采样（双线性插值或ConvTranspose2d）后，
    将编码器跳跃连接特征通过注意力门处理，然后与上采样特征拼接，再接一个 DoubleConv。
    """

    def __init__(self, in_channels, out_channels, skip_channels, use_batchnorm=True, bilinear=True):
        """
        Args:
            in_channels (int): 来自前一个解码器阶段的特征图通道数。
            out_channels (int): 此Up块输出的特征图通道数。
            skip_channels (int): 来自编码器跳跃连接的特征图通道数。
            use_batchnorm (bool): 是否使用批量归一化。
            bilinear (bool): 是否使用双线性插值进行上采样。
        """
        super().__init__()
        self.bilinear = bilinear

        # 上采样层
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 如果使用双线性插值，x1_up的通道数仍为 in_channels
            # 拼接后的特征图通道数 = in_channels (x1_up) + skip_channels (x2_att)
            concat_channels = in_channels + skip_channels
        else:
            # 如果使用 ConvTranspose2d，它会自行将通道数减半
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 拼接后的特征图通道数 = (in_channels // 2) (x1_up) + skip_channels (x2_att)
            concat_channels = (in_channels // 2) + skip_channels

        # 注意力门
        # F_g: gating signal (上采样后的 x1) 的通道数
        # F_l: skip connection (x2) 的通道数
        # F_int: 注意力门的中间通道数，通常是 F_l 的一半
        ag_g_channels = in_channels if bilinear else in_channels // 2
        ag_l_channels = skip_channels
        ag_int_channels = ag_l_channels // 2  # 常见的选择，可以根据需要调整
        self.att = Attention_block(F_g=ag_g_channels, F_l=ag_l_channels, F_int=ag_int_channels)

        # 双卷积层，处理拼接后的特征图
        self.conv = DoubleConv(concat_channels, out_channels, use_batchnorm=use_batchnorm)

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): 来自前一个解码器阶段的特征图 (较粗糙)。
            x2 (torch.Tensor): 来自编码器跳跃连接的特征图 (较精细)。
        Returns:
            torch.Tensor: 经过上采样、注意力加权和双卷积处理后的特征图。
        """
        # 1. 对 x1 进行上采样
        x1_up = self.up(x1)

        # 2. 如果尺寸不匹配（例如由于奇数/偶数维度），则对 x1_up 进行填充以匹配 x2 的尺寸
        # 这一步确保 x1_up 和 x2 具有相同的空间尺寸，以便 Attention_block 和后续的拼接操作
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]

        x1_up = torch.nn.functional.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                                                diffY // 2, diffY - diffY // 2])

        # 3. 将编码器跳跃连接 x2 通过注意力门处理，使用 x1_up 作为 gating signal
        x2_att = self.att(g=x1_up, x_skip=x2)

        # 4. 将经过注意力加权的 x2_att 和上采样后的 x1_up 拼接
        x = torch.cat([x2_att, x1_up], dim=1)  # 通常将跳跃连接放在前面

        # 5. 通过双卷积层
        return self.conv(x)


class UNetModified(nn.Module):
    """
    主Attention U-Net架构。
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
        # 如果使用双线性上采样，最深层通道数通常不会减半，因为上采样不改变通道数
        # 如果使用 ConvTranspose2d，它会减半通道数，所以瓶颈层输出通道数需要调整
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_batchnorm=use_batchnorm)  # 最深层

        # 解码器路径
        # Up(in_channels_from_prev_decoder, out_channels_of_this_block, skip_channels_from_encoder, ...)
        # up1: 输入来自 down4 (1024//factor), 跳跃连接来自 down3 (512)
        self.up1 = Up(1024 // factor, 512 // factor, 512, use_batchnorm, bilinear)
        # up2: 输入来自 up1 (512//factor), 跳跃连接来自 down2 (256)
        self.up2 = Up(512 // factor, 256 // factor, 256, use_batchnorm, bilinear)
        # up3: 输入来自 up2 (256//factor), 跳跃连接来自 down1 (128)
        self.up3 = Up(256 // factor, 128 // factor, 128, use_batchnorm, bilinear)
        # up4: 输入来自 up3 (128//factor), 跳跃连接来自 inc (64)
        self.up4 = Up(128 // factor, num_intermediate_channels, 64, use_batchnorm, bilinear)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)  # out: 64 channels
        x2 = self.down1(x1)  # out: 128 channels
        x3 = self.down2(x2)  # out: 256 channels
        x4 = self.down3(x3)  # out: 512 channels
        x5 = self.down4(x4)  # out: 1024//factor channels (瓶颈层特征)

        # 带跳跃连接的解码器 (现在是 Attention U-Net 的解码器)
        x = self.up1(x5, x4)  # x5 (gating), x4 (skip)
        x = self.up2(x, x3)  # x (gating), x3 (skip)
        x = self.up3(x, x2)  # x (gating), x2 (skip)
        x = self.up4(x, x1)  # x (gating), x1 (skip) (最终分类前的输出特征)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class VisionTransformer(nn.Module):
    """
    此类别已重构为实现Attention U-Net架构，同时保持原始类名和forward方法签名。
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        # self.classifier = config.classifier # ViT特定参数，此处不使用。

        in_channels = 3
        # 确保 num_intermediate_channels 能够被 Attention Gate 的 F_int 逻辑整除
        # 这里设置为 64 是一个常见且合理的选择，因为 inc 的输出是 64
        unet_intermediate_channels = 64

        self.unet_model = UNetModified(
            in_channels=in_channels,
            num_intermediate_channels=unet_intermediate_channels,
            bilinear=True,  # 保持与原U-Net一致
            use_batchnorm=True  # 保持与原U-Net一致
        )

        self.segmentation_head = SegmentationHead(
            in_channels=unet_intermediate_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=1,
        )
        self.config = config

    def forward(self, x):
        # 确保输入是3通道，如果单通道则复制
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 通过 Attention U-Net 提取特征
        features_before_segmentation_head = self.unet_model(x)

        # 通过分割头生成 logits
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
