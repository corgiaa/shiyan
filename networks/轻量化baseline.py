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


# --- 标准上采样块 (替代 LGBU) ---
class StandardUpBlock(nn.Module):
    """
    一个标准的上采样块，包含双线性插值上采样，卷积，批归一化，ReLU激活，以及 Dropout。
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super().__init__()
        self.up_layer = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) # 添加 Dropout
        )

    def forward(self, x):
        x = self.up_layer(x)
        x = self.conv_main(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class LightweightTransUNetDecoder(nn.Module):
    """
    修改后的解码器，移除了 DDFE 和 LGBU 模块，并添加了 Dropout。
    """

    def __init__(self, encoder_channels, decoder_hidden_dim=128, out_channels=128, dropout_rate=0.4):
        super().__init__()
        fused_input_channels = sum(encoder_channels)

        # 1. 初始融合和通道数调整，并添加 Dropout
        self.initial_fusion = nn.Sequential(
            nn.Conv2d(fused_input_channels, decoder_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) # 添加 Dropout
        )

        # 2. 上采样路径
        # 使用 StandardUpBlock 替换 LGBU
        self.up_blocks = nn.ModuleList([
            StandardUpBlock(decoder_hidden_dim, decoder_hidden_dim // 2, dropout_rate),
            StandardUpBlock(decoder_hidden_dim // 2, decoder_hidden_dim // 4, dropout_rate)
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

        # 1. 初始降维和 Dropout
        x = self.initial_fusion(fused_features)

        # 2. 逐步上采样 (使用 StandardUpBlock)
        for up_block in self.up_blocks:
            x = up_block(x)

        # 3. 最终输出卷积
        output = self.output_conv(x)

        return output


class VisionTransformer(nn.Module):
    """
    主类 VisionTransformer。
    使用了 timm 的 ViT-Tiny 作为骨干，并结合了带有 Dropout 的标准解码器。
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.img_size = img_size

        in_channels = 3
        decoder_output_channels = 256
        dropout_rate = 0.5 # 定义 Dropout 率

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

        # --- 使用修改后的解码器，并传递 Dropout 率 ---
        self.decoder = LightweightTransUNetDecoder(
            encoder_channels=encoder_channels,
            decoder_hidden_dim=128,
            out_channels=decoder_output_channels,
            dropout_rate=dropout_rate # 传递 Dropout 率
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
        logger.info("Initialized with standard decoder and Dropout.")

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

        # 2. Decoder Forward (现在是标准解码器，包含 Dropout)
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
                # Skip head weights or specific norm layers that might not match
                continue
            else:
                pass # Unmatched keys are simply skipped

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
