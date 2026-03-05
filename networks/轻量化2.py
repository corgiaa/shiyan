# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, MaxPool2d, ConvTranspose2d, Identity
from scipy import ndimage
from . import vit_seg_configs as configs

logger = logging.getLogger(__name__)


# --- 工具函数 ---
def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# --- 创新模块 2: 拉普拉斯边界引导上采样块 (LGBU) ---
class LaplacianGuidedUpBlock(nn.Module):
    """
    创新点2: LGBU
    在上采样过程中，显式计算特征图的拉普拉斯边缘，并利用边缘信息
    来细化上采样后的特征，防止边界模糊。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 1. 基础特征变换
        self.up_layer = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 拉普拉斯边缘提取器 (固定权重，不可学习，纯数学算子)
        # 定义拉普拉斯卷积核
        laplacian_kernel = torch.tensor([[-1., -1., -1.],
                                         [-1., 8., -1.],
                                         [-1., -1., -1.]], dtype=torch.float32)
        # 重塑为 [1, 1, 3, 3] 以便用于卷积
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        # 注册为 buffer，不作为模型参数更新，但在 state_dict 中保存
        self.register_buffer('laplacian_kernel', laplacian_kernel)

        # 3. 边界注意力生成器
        # 将提取的边缘图变换为注意力权重
        self.edge_process = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 4. 最终混合
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. 上采样和主特征提取
        x_up = self.up_layer(x)
        x_feat = self.conv_main(x_up)  # [B, out_channels, H, W]

        # 2. 计算拉普拉斯边缘 (Depthwise处理)
        # 我们对 x_feat 的每个通道求边缘，然后取平均或最大值
        # 为了效率，我们先在通道维度聚合，再求边缘
        x_feat_mean = torch.mean(x_feat, dim=1, keepdim=True)  # [B, 1, H, W]

        # 使用 F.conv2d 进行固定核卷积
        edge_map = F.conv2d(x_feat_mean, self.laplacian_kernel, padding=1)
        # 取绝对值，因为边缘可能是正或负的梯度变化
        edge_map = torch.abs(edge_map)

        # 3. 生成边界注意力
        # 我们利用原始特征来生成对边缘图的响应，或者直接处理边缘图
        # 这里我们将边缘图注入到特征中

        # 修正逻辑：
        # 边缘图本身就是高频信息。我们希望网络在边缘处加强特征表示。
        # 将边缘图作为一种“提示”加到注意力生成网络中。

        edge_attention = self.edge_process(x_feat * edge_map)  # 边缘区域的特征被放大用于生成权重

        # 4. 特征细化
        # 在原有特征基础上，加上 (特征 * 边缘注意力)，使得边缘处的特征值被增强
        x_refined = x_feat + (x_feat * edge_attention)

        return self.final_conv(x_refined)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class LightweightTransUNetDecoder(nn.Module):
    """
    修改后的解码器，集成了 LGBU 模块，并增加了 Dropout 层。
    """

    def __init__(self, encoder_channels, decoder_hidden_dim=128, out_channels=128):
        super().__init__()
        fused_input_channels = sum(encoder_channels)

        # 1. 初始融合和通道数调整
        self.initial_fusion = nn.Sequential(
            nn.Conv2d(fused_input_channels, decoder_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_hidden_dim),
            nn.ReLU(inplace=True)
        )

        # --- 添加 Dropout 层 ---
        self.dropout = nn.Dropout2d(0.4)

        # 2. 上采样路径
        # 使用创新模块 2: LGBU 替换普通的 Sequential 上采样
        self.up_blocks = nn.ModuleList([
            # 第一次上采样 (例如 14x14 -> 28x28)
            LaplacianGuidedUpBlock(decoder_hidden_dim, decoder_hidden_dim // 2),

            # 第二次上采样 (例如 28x28 -> 56x56)
            LaplacianGuidedUpBlock(decoder_hidden_dim // 2, decoder_hidden_dim // 4)
        ])

        final_decoder_channels = decoder_hidden_dim // 4
        if final_decoder_channels == 0:
            final_decoder_channels = 1

        # 3. 最终输出卷积
        self.output_conv = nn.Conv2d(final_decoder_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, features):
        if not features:
            raise ValueError("LightweightTransUNetDecoder received an empty list of features.")

        # (B, C_vit*num_features, H_feat, W_feat)
        fused_features = torch.cat(features, dim=1)

        # 1. 初始降维
        x = self.initial_fusion(fused_features)

        # --- 应用 Dropout ---
        x = self.dropout(x)

        # 2. 逐步上采样 (使用 LGBU)
        for up_block in self.up_blocks:
            x = up_block(x)

        # 3. 最终输出卷积
        output = self.output_conv(x)

        return output


class VisionTransformer(nn.Module):
    """
    主类 VisionTransformer。
    使用了 timm 的 ViT-Tiny 作为骨干，并结合了带有 LGBU 的解码器。
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.img_size = img_size

        in_channels = 3
        decoder_output_channels = 256

        # --- timm ViT-Tiny/16 Backbone 配置 ---
        vit_model_name = "vit_tiny_patch16_224"
        vit_pretrained = False

        self.vit_backbone = timm.create_model(
            vit_model_name,
            pretrained=vit_pretrained,
            img_size=img_size,
            drop_path_rate=0.1
        )

        encoder_channels = [self.vit_backbone.embed_dim] * 4
        logger.info(f"ViT-Tiny/16 encoder feature channels: {encoder_channels}")

        # --- 使用改进后的解码器 ---
        self.decoder = LightweightTransUNetDecoder(
            encoder_channels=encoder_channels,
            decoder_hidden_dim=128,
            out_channels=decoder_output_channels
        )

        # --- 分割头 ---
        patch_size = self.vit_backbone.patch_embed.patch_size[0]
        initial_feat_h = img_size // patch_size
        final_decoder_feat_h = initial_feat_h * (2 ** len(self.decoder.up_blocks))
        up_factor = img_size // final_decoder_feat_h

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=up_factor,
        )

        logger.info(f"Using timm ViT backbone: {vit_model_name} with pretrained={vit_pretrained}")
        logger.info("Initialized with Innovative Module: LaplacianGuidedUpBlock and Dropout (0.2)")

        if not vit_pretrained:
            logger.warning(
                "ViT backbone is not using timm's pretrained weights. Local weights will be loaded via load_from().")

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B, C, H, W = x.shape
        patch_size = self.vit_backbone.patch_embed.patch_size[0]
        feat_h, feat_w = H // patch_size, W // patch_size

        # 1. ViT Backbone Forward
        x = self.vit_backbone.patch_embed(x)
        cls_token = self.vit_backbone.cls_token
        pos_embed = self.vit_backbone.pos_embed
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)
        x = x + pos_embed
        x = self.vit_backbone.norm_pre(x)

        features_from_blocks = []
        block_indices = [2, 5, 8, 11]

        for i, blk in enumerate(self.vit_backbone.blocks):
            x = blk(x)
            if i in block_indices:
                x_no_cls = x[:, 1:]
                feature_map = x_no_cls.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
                features_from_blocks.append(feature_map)

        # 2. Decoder Forward (包含 LGBU 和 Dropout)
        decoded_features = self.decoder(features_from_blocks)

        # 3. Segmentation Head
        logits = self.segmentation_head(decoded_features)
        return logits

    def load_from(self, weights):
        """
        强制加载 ViT-T_16.bin 权重。
        """
        logger.info(f"Attempting to load weights. Received type for 'weights' argument: {type(weights)}")
        target_vit_t_path = "model/vit_checkpoint/imagenet21k/ViT-T_16.bin"
        state_dict_to_load = None

        if not os.path.exists(target_vit_t_path):
            logger.error(f"ERROR: Desired ViT-T_16.bin not found at '{target_vit_t_path}'.")
            return

        logger.info(f"Overriding 'weights' argument. Forcing load of PyTorch state_dict from '{target_vit_t_path}'.")
        try:
            state_dict_to_load = torch.load(target_vit_t_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Failed to load PyTorch state_dict from '{target_vit_t_path}': {e}")
            raise

        if 'state_dict' in state_dict_to_load and isinstance(state_dict_to_load['state_dict'], dict):
            state_dict_to_load = state_dict_to_load['state_dict']

        new_state_dict = {}
        model_state_dict = self.vit_backbone.state_dict()

        for k, v in state_dict_to_load.items():
            if k.startswith('model.'):
                k = k[len('model.'):]
            if k.startswith('encoder.'):
                k = k[len('encoder.'):]

            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                new_state_dict[k] = v
            elif 'head' in k or 'norm.0.weight' in k or 'norm.0.bias' in k:
                continue
            else:
                pass

        self.vit_backbone.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Successfully loaded '{target_vit_t_path}' into timm ViT backbone (strict=False).")


# 兼容性配置
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
