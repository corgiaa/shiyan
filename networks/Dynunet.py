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

# 引入MONAI的DynUNet
from monai.networks.nets import DynUNet
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, MaxPool2d, ConvTranspose2d, Identity
from torch.nn.modules.utils import _pair
from scipy import ndimage
# 保留原始的ViT配置导入，尽管U-Net结构不会使用其大部分参数
from . import vit_seg_configs as configs

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


class SegmentationHead(nn.Sequential):
    """
    分割头，用于将U-Net的输出特征图转换为最终的类别预测。
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class VisionTransformer(nn.Module):
    """
    此类别已重构为实现MONAI的DynUNet架构，同时保持原始类名和forward方法签名。
    它接收输入图像并输出与原始分辨率相同的特征图，
    这些特征图随后传递给分割头。
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        # self.classifier = config.classifier # ViT特定参数，此处不使用。

        in_channels = 3  # 输入图像通道数，通常为RGB的3通道
        # DynUNet的最终输出通道数，这将作为SegmentationHead的输入通道数。
        # 对应原始Attention U-Net的最后一个解码器块的输出通道数 (64)。
        unet_output_channels = 64

        # --- MONAI DynUNet 参数配置 ---
        # 空间维度为2D图像
        spatial_dims = 2

        # 编码器和解码器的特征图通道数。
        # 对应原始U-Net的通道数 progression: 64, 128, 256, 512, 1024 (瓶颈层)。
        # DynUNet的filters列表定义了每个编码器阶段的输出通道数。
        filters = [64, 128, 256, 512, 1024]

        # 编码器每个阶段的卷积核大小。
        # 第一个元素用于初始卷积块（通常不进行下采样）。
        # 后续元素用于下采样后的卷积块。
        # 对应原始U-Net中常用的3x3卷积。
        kernel_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]

        # 编码器每个阶段的步长。
        # 第一个元素用于初始卷积块，不进行下采样，所以步长为1。
        # 后续元素用于下采样操作，步长为2，对应原始U-Net的MaxPool2d(2)。
        strides = [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2)]

        # 解码器每个阶段的上采样卷积核大小。
        # 对应原始U-Net中常用的2x2 ConvTranspose2d 或 双线性插值上采样。
        upsample_kernel_size = [(2, 2), (2, 2), (2, 2), (2, 2)]  # 4个上采样阶段

        # 归一化层和激活函数，与原始U-Net保持一致
        norm_name = "batch"  # 使用BatchNorm2d
        act_name = "relu"  # 使用ReLU激活函数

        # DynUNet的其他配置
        deep_supervision = False  # 通常在训练时启用，这里为简化设置为False
        res_block = False  # 原始U-Net使用简单的卷积块，而非残差块

        self.unet_model = DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=unet_output_channels,  # DynUNet的最终输出通道数，作为分割头的输入
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            norm_name=norm_name,
            act_name=act_name,
            deep_supervision=deep_supervision,
            res_block=res_block,
            # DynUNet的self_attention参数通常指Transformer-style attention，
            # 而非原始Attention U-Net中的Attention Gate。
            # 默认DynUNet不包含Attention Gate。
            # self_attention=False,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=unet_output_channels,  # 接收DynUNet的输出特征图
            out_channels=num_classes,  # 输出最终的类别数量
            kernel_size=1,
            upsampling=1,
        )
        self.config = config

    def forward(self, x):
        # 确保输入是3通道，如果单通道则复制
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 通过 MONAI DynUNet 提取特征
        # 当 deep_supervision=False 时，DynUNet 直接返回最终解码器层的输出。
        features_before_segmentation_head = self.unet_model(x)

        # 通过分割头生成 logits
        logits = self.segmentation_head(features_before_segmentation_head)
        return logits

    def load_from(self, weights):
        logger.warning(
            "The 'load_from' method is specific to Vision Transformer weights and is not applicable to the DynUNet architecture. Skipping weight loading.")
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
