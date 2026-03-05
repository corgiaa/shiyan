# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
import torchvision.models as models  # 用于ResNet骨干
from torchvision.models import ResNet50_Weights  # 明确指定ResNet50的权重

# 保留原始的ViT配置导入，但SSFormer不直接使用这些配置
from . import vit_seg_configs as configs

logger = logging.getLogger(__name__)


# --- 通用工具函数 ---
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


# ACT2FN 不再用于SSFormer的Mlp，因为SSFormerMlp直接使用nn.GELU
# def swish(x):
#     return x * torch.sigmoid(x)
# ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# =============================================================================
# SSFormer 组件实现
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Expansion Block for channel-wise attention.
    Applied within the MLP of the Transformer Block in SSFormer.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        # AdaptiveAvgPool1d is used for sequence data (B, N, C) -> (B, C, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, N, C)
        B, N, C = x.shape
        # Squeeze operation: Global average pooling across sequence dimension
        y = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # (B, C)
        # Expansion operation: FC layers + sigmoid to get attention weights
        y = self.fc(y).unsqueeze(1).expand_as(x)  # (B, 1, C) -> (B, N, C)
        return x * y


class SSFormerMlp(nn.Module):
    """
    SSFormer's MLP block, incorporating the SEBlock.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., reduction=16):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.se = SEBlock(hidden_features, reduction=reduction)  # SE block applied here
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.se(x)  # Apply SE after first linear and activation
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SSFormerAttention(nn.Module):
    """
    Standard Multi-head Self-Attention for Vision Transformers.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SSFormerBlock(nn.Module):
    """
    SSFormer's Transformer Block, composed of SSFormerAttention and SSFormerMlp.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, reduction=16):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSFormerAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # DropPath is usually used here, but for simplicity, we use nn.Identity()
        # For a full implementation, consider using timm.models.layers.DropPath
        self.drop_path = nn.Identity()  # stochastic depth decay rule (not implemented here for simplicity)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SSFormerMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                               act_layer=act_layer, drop=drop, reduction=reduction)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SSFormerDecoder(nn.Module):
    """
    A simple convolutional decoder for SSFormer, upsampling the Transformer output.
    """

    def __init__(self, encoder_dim, num_classes):
        super().__init__()
        # The decoder typically takes the final Transformer output and upsamples it.
        # The paper suggests a simple decoder with 4 convolutional layers, BatchNorm, ReLU
        # and then bilinear upsampling to the original image size.
        self.conv1 = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(encoder_dim // 2, encoder_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(encoder_dim // 4, encoder_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_dim // 8),
            nn.ReLU(inplace=True)
        )
        # Final layer to output num_classes, often without activation for logits
        self.conv4 = nn.Conv2d(encoder_dim // 8, num_classes, kernel_size=1)

    def forward(self, transformer_output, input_shape):
        B, N, D = transformer_output.shape

        # Calculate spatial dimensions from N (number of patches)
        # Assuming square patches and that N is a perfect square for simplicity
        H_feat = int(math.sqrt(N))
        W_feat = int(math.sqrt(N))

        # Robustness check for H_feat, W_feat inference
        if H_feat * W_feat != N:
            # If N is not a perfect square, or if the assumption of square patches is wrong,
            # infer from input_shape and the expected downsampling ratio (1/32 for ResNet layer4)
            expected_H_feat = input_shape[0] // 32
            expected_W_feat = input_shape[1] // 32

            if expected_H_feat * expected_W_feat == N:
                H_feat, W_feat = expected_H_feat, expected_W_feat
            else:
                # Fallback if direct inference from N or expected ratio doesn't match
                logger.warning(
                    f"Transformer output N={N} does not match expected H_feat*W_feat={expected_H_feat * expected_W_feat} for input_shape={input_shape} at 1/32 resolution. Attempting to infer from N.")
                # Try to infer from N, assuming one dimension is known or can be derived
                if N % (input_shape[1] // 32) == 0:
                    W_feat = input_shape[1] // 32
                    H_feat = N // W_feat
                elif N % (input_shape[0] // 32) == 0:
                    H_feat = input_shape[0] // 32
                    W_feat = N // H_feat
                else:  # Last resort, assume square and warn
                    H_feat = W_feat = int(math.sqrt(N))
                    logger.warning(f"Assuming square feature map {H_feat}x{W_feat} from N={N} as a last resort.")

        # Reshape transformer output from (B, N, D) to (B, D, H_feat, W_feat)
        x = transformer_output.transpose(1, 2).reshape(B, D, H_feat, W_feat)

        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 1/16

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 1/8

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 1/4

        x = self.conv4(x)  # Final classification layer

        # Final upsample to original input size
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x


# =============================================================================
# 主类 VisionTransformer (SSFormer 包装)
# =============================================================================

class VisionTransformer(nn.Module):
    """
    重构为 SSFormer 架构 (ResNet Backbone + Transformer Encoder with SE + Simple Decoder)。
    保持原始类名和 forward 方法签名以兼容现有接口。
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.config = config  # 保留接口，但SSFormer不直接使用此config对象

        # --- SSFormer Backbone: ResNet50 as feature extractor ---
        # 使用 ResNet50_Weights.IMAGENET1K_V1 明确指定权重，避免UserWarning
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Remove the classification head of ResNet
        self.resnet.fc = nn.Identity()

        # We will extract features from layer4 (output stride 32)
        # ResNet50 layer4 output channels: 2048
        cnn_out_channels = 2048

        # Transformer Encoder parameters (example for a ViT-B like encoder)
        encoder_dim = 768  # Common transformer embedding dimension
        num_transformer_layers = 12  # Typical for ViT-B
        num_heads = 12  # Typical for ViT-B
        mlp_ratio = 4
        se_reduction = 16  # Reduction ratio for SEBlock

        # Projection layer to match CNN output channels to Transformer's embedding dimension
        self.conv_proj = nn.Conv2d(cnn_out_channels, encoder_dim, kernel_size=1)

        # Positional embedding for Transformer input
        # Corrected: ResNet layer4 output is 224/32 = 7x7 for img_size=224
        num_patches = (img_size // 32) * (img_size // 32)  # <--- 修改点1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)  # Initialize position embedding

        # Transformer Encoder
        self.transformer_blocks = nn.ModuleList([
            SSFormerBlock(
                dim=encoder_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=0.1,  # Dropout rate
                attn_drop=0.0,  # Attention dropout rate
                drop_path=0.0,  # Stochastic depth, simplified to 0 for this example
                norm_layer=LayerNorm,
                reduction=se_reduction
            ) for _ in range(num_transformer_layers)
        ])
        self.norm = LayerNorm(encoder_dim)  # Final LayerNorm after transformer blocks

        # --- SSFormer Decoder ---
        self.decoder = SSFormerDecoder(encoder_dim=encoder_dim, num_classes=num_classes)

        # --- 加载预训练权重 ---
        # ResNet50的预训练权重已经在models.resnet50(weights=...)中加载
        # 如果需要加载SSFormer的特定预训练权重，可以在这里添加逻辑。
        # 原有的mit_b3.pth路径不再适用。
        logger.info("ResNet50 backbone initialized with ImageNet pre-trained weights.")
        logger.info("SSFormer Transformer and Decoder initialized randomly.")

    def _load_pretrained_weights(self, path):
        """
        此方法现在主要用于加载ResNet的预训练权重（已在__init__中处理），
        或未来用于加载SSFormer整体的预训练权重。
        原有的SegFormer MiT权重加载逻辑已移除。
        """
        if os.path.exists(path):
            logger.warning(
                f"Provided path {path} is for SegFormer MiT weights, which are not compatible with SSFormer. Skipping.")
        else:
            logger.info(
                "No SSFormer specific pretrained weights provided or found. Using randomly initialized Transformer and Decoder.")

    def forward(self, x):
        # 确保输入是3通道
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        input_shape = x.shape[-2:]  # (H, W)

        # --- ResNet Backbone (Feature Extraction) ---
        # ResNet forward pass to get features before the global average pooling
        x_res = self.resnet.conv1(x)
        x_res = self.resnet.bn1(x_res)
        x_res = self.resnet.relu(x_res)
        x_res = self.resnet.maxpool(x_res)  # 1/4 resolution

        x_res = self.resnet.layer1(x_res)  # 1/4 resolution, 256 channels
        x_res = self.resnet.layer2(x_res)  # 1/8 resolution, 512 channels
        x_res = self.resnet.layer3(x_res)  # 1/16 resolution, 1024 channels
        x_res = self.resnet.layer4(x_res)  # 1/32 resolution, 2048 channels (this is what we'll use)

        # --- Project CNN features to Transformer's embedding dimension ---
        # x_res: (B, 2048, H/32, W/32)
        x_proj = self.conv_proj(x_res)  # (B, encoder_dim, H/32, W/32)

        # --- Prepare for Transformer Encoder ---
        # Flatten spatial dimensions and transpose to (B, N, D)
        B, D, H_feat, W_feat = x_proj.shape
        x_flat = x_proj.flatten(2).transpose(1, 2)  # (B, H_feat*W_feat, D)

        # Add positional embedding
        x_flat = x_flat + self.pos_embed

        # --- Transformer Encoder ---
        for blk in self.transformer_blocks:
            x_flat = blk(x_flat)
        x_flat = self.norm(x_flat)  # (B, N, D) - final transformer output

        # --- Decoder ---
        logits = self.decoder(x_flat, input_shape)

        return logits

    def load_from(self, weights):
        """
        保留接口。如果外部调用此方法加载整个模型的权重（包含Head），可以在这里处理。
        目前主要依赖 __init__ 中的 ResNet 预训练权重加载。
        """
        if 'state_dict' in weights:
            self.load_state_dict(weights['state_dict'], strict=False)
        else:
            self.load_state_dict(weights, strict=False)


# 保持CONFIGS字典不变，但请注意这些配置不再直接适用于SSFormer的内部结构。
# 它们可能用于设置num_classes或img_size等通用参数。
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