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

# 引入 segmentation_models_pytorch 库
import segmentation_models_pytorch as smp

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, MaxPool2d, ConvTranspose2d, Identity
from torch.nn.modules.utils import _pair
from scipy import ndimage
# 保留原始的ViT配置导入，尽管FPN结构不会使用其大部分参数
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
    分割头，用于将模型的输出特征图转换为最终的类别预测。
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class VisionTransformer(nn.Module):
    """
    此类别已重构为实现 segmentation_models_pytorch 的 FPN 架构，
    同时保持原始类名和forward方法签名。
    它接收输入图像并输出与原始分辨率相同的特征图，
    这些特征图随后传递给分割头。
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        # self.classifier = config.classifier # ViT特定参数，此处不使用。

        in_channels = 3  # 输入图像通道数，通常为RGB的3通道
        # FPN 的中间输出通道数，这将作为SegmentationHead的输入通道数。
        # 这里设置为256，表示FPN模型会输出256个通道的特征图。
        fpn_output_channels = 256 # 命名更改以反映FPN

        # --- segmentation_models_pytorch FPN 参数配置 ---
        # 选择一个编码器骨干。'resnet50' 是一个常见的选择。
        encoder_name = "resnet50"
        # 设置 encoder_weights='imagenet' 表示使用在 ImageNet 上预训练的权重。
        # 如果不需要预训练权重，可以设置为 None。
        encoder_weights = "imagenet"  # 使用预训练权重

        # 编码器深度，对应骨干网络的下采样级别数。
        # resnet50 通常有5个级别。
        encoder_depth = 5

        # 激活函数，通常在最终输出层之前不使用，让SegmentationHead处理。
        # 默认是None，表示输出原始logits。
        activation = None

        # !!! 将分割模型从 U-Net 更改为 FPN (Feature Pyramid Network) !!!
        self.segmentation_model = smp.FPN( # 变量名改为更通用的 segmentation_model
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=fpn_output_channels,  # FPN 的中间输出通道数，由SegmentationHead进一步处理
            encoder_depth=encoder_depth,
            # FPN 也有 decoder_channels 参数，但 smp 会为不同的编码器提供合理的默认值。
            # 我们可以选择不显式指定，让库自动处理。
            # 例如，对于 resnet50，默认可能是 (256, 128, 64, 32, 16) 或其他。
            activation=activation,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=fpn_output_channels,  # 接收FPN的中间特征图
            out_channels=num_classes,  # 输出最终的类别数量
            kernel_size=1,
            upsampling=1,
        )
        self.config = config

    def forward(self, x):
        # 确保输入是3通道，如果单通道则复制
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 通过 segmentation_models_pytorch FPN 提取特征
        features_before_segmentation_head = self.segmentation_model(x)

        # 通过分割头生成 logits
        logits = self.segmentation_head(features_before_segmentation_head)
        return logits

    def load_from(self, weights):
        logger.warning(
            "The 'load_from' method is specific to Vision Transformer weights and is not applicable to the FPN architecture. Skipping weight loading.")
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
