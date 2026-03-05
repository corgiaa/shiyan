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


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, hidden_size, num_attention_heads, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.dropout = Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class MLP(nn.Module):
    """MLP块"""

    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.dense_1 = Linear(hidden_size, intermediate_size)
        self.activation = torch.nn.functional.gelu
        self.dense_2 = Linear(intermediate_size, hidden_size)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    """Transformer层"""

    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout_rate)
        self.mlp = MLP(hidden_size, intermediate_size, dropout_rate)

    def forward(self, hidden_states, attention_mask=None):
        # Self Attention
        attention_output = self.attention(
            self.attention_norm(hidden_states),
            self.attention_norm(hidden_states),
            self.attention_norm(hidden_states),
            attention_mask=attention_mask
        )
        hidden_states = hidden_states + attention_output

        # MLP
        mlp_output = self.mlp(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + mlp_output

        return hidden_states


class SETRDecoder(nn.Module):
    """SETR解码器 - Progressive Upsampling (PUP)"""

    def __init__(self,
                 in_channels=768,  # ViT-B输出维度
                 num_classes=21843,
                 decoder_channels=[256, 128, 64],
                 dropout_rate=0.1):
        super().__init__()

        self.num_classes = num_classes

        # 1x1卷积降维
        self.input_proj = nn.Conv2d(in_channels, decoder_channels[0], kernel_size=1)

        # Progressive Upsampling layers
        self.decoder_blocks = nn.ModuleList()

        # First upsampling block (2x)
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1],
                               kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        ))

        # Second upsampling block (2x)
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2],
                               kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        ))

        # Third upsampling block (2x)
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[2], decoder_channels[2],
                               kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        ))

        # Final classification layer
        self.classifier = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)

    def forward(self, features):
        """
        features: [B, C, H, W] - 来自ViT backbone的特征
        """
        # Input projection
        x = self.input_proj(features)

        # Progressive upsampling
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)

        # Final classification
        output = self.classifier(x)

        return output


class SETRNaiveDecoder(nn.Module):
    """SETR朴素解码器 - 直接上采样"""

    def __init__(self,
                 in_channels=768,  # ViT-B输出维度
                 num_classes=21843,
                 hidden_channels=256,
                 dropout_rate=0.1):
        super().__init__()

        self.num_classes = num_classes

        # 特征处理
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        # 分类层
        self.classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, features):
        """
        features: [B, C, H, W] - 来自ViT backbone的特征
        """
        x = self.conv1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 分类
        output = self.classifier(x)

        return output


class SETRMLADecoder(nn.Module):
    """SETR Multi-Level Aggregation (MLA)解码器"""

    def __init__(self,
                 in_channels=768,
                 num_classes=21843,
                 mla_channels=256,
                 dropout_rate=0.1):
        super().__init__()

        self.num_classes = num_classes
        self.mla_channels = mla_channels

        # 多层特征聚合
        # 这里简化版本，实际需要从多个Transformer层提取特征
        self.feature_proj = nn.Conv2d(in_channels, mla_channels, kernel_size=1)

        # MLA模块
        self.mla_p2 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(inplace=True)
        )

        self.mla_p3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(mla_channels, mla_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.mla_p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(mla_channels, mla_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(mla_channels * 3, mla_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # 分类层
        self.classifier = nn.Conv2d(mla_channels, num_classes, kernel_size=1)

    def forward(self, features):
        """
        features: [B, C, H, W] - 来自ViT backbone的特征
        """
        # 特征投影
        x = self.feature_proj(features)

        # 多尺度特征提取
        p2 = self.mla_p2(x)
        p3 = self.mla_p3(x)
        p4 = self.mla_p4(x)

        # 特征融合
        fused = torch.cat([p2, p3, p4], dim=1)
        fused = self.fusion(fused)

        # 分类
        output = self.classifier(fused)

        return output


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class VisionTransformer(nn.Module):
    """
    SETR with ViT-B backbone
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,
                 decoder_type='pup'):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.img_size = img_size
        self.decoder_type = decoder_type

        # --- ViT-B/16 Backbone using timm ---
        vit_model_name = "vit_base_patch16_224"
        vit_pretrained = False  # 我们会手动加载权重

        self.vit_backbone = timm.create_model(
            vit_model_name,
            pretrained=vit_pretrained,
            img_size=img_size,
            drop_path_rate=0.1,
            num_classes=0  # 移除分类头
        )

        # --- SETR Decoder ---
        if decoder_type == 'pup':
            self.decoder = SETRDecoder(
                in_channels=768,
                num_classes=num_classes,
                decoder_channels=[256, 128, 64],
                dropout_rate=0.1
            )
        elif decoder_type == 'mla':
            self.decoder = SETRMLADecoder(
                in_channels=768,
                num_classes=num_classes,
                mla_channels=256,
                dropout_rate=0.1
            )
        else:  # naive
            self.decoder = SETRNaiveDecoder(
                in_channels=768,
                num_classes=num_classes,
                hidden_channels=256,
                dropout_rate=0.1
            )

        # 计算上采样因子
        patch_size = 16
        feature_resolution = img_size // patch_size
        self.upsampling_factor = img_size // feature_resolution

        logger.info(f"Using ViT-B/16 backbone with SETR-{decoder_type.upper()} decoder")
        logger.info(f"Num classes: {num_classes}, Feature resolution: {feature_resolution}x{feature_resolution}")

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B, C, H, W = x.shape

        # 1. ViT Backbone Forward
        # 通过patch embedding
        x_patches = self.vit_backbone.patch_embed(x)

        # 添加位置编码和CLS token
        cls_token = self.vit_backbone.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat((cls_token, x_patches), dim=1)
        x_with_cls = x_with_cls + self.vit_backbone.pos_embed
        x_with_cls = self.vit_backbone.pos_drop(x_with_cls)

        # 通过transformer blocks
        for blk in self.vit_backbone.blocks:
            x_with_cls = blk(x_with_cls)

        x_with_cls = self.vit_backbone.norm(x_with_cls)

        # 移除CLS token，reshape到特征图格式
        x_no_cls = x_with_cls[:, 1:]  # [B, num_patches, embed_dim]
        patch_size = self.vit_backbone.patch_embed.patch_size[0]
        feat_h = feat_w = H // patch_size
        pixel_features = x_no_cls.transpose(1, 2).reshape(B, -1, feat_h, feat_w)

        # 2. SETR Decoder
        if self.decoder_type == 'pup':
            # Progressive Upsampling Decoder
            seg_logits = self.decoder(pixel_features)

            # 确保输出尺寸与输入匹配
            if seg_logits.shape[-2:] != (H, W):
                seg_logits = F.interpolate(
                    seg_logits,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
        else:
            # Naive或MLA Decoder需要上采样
            seg_logits = self.decoder(pixel_features)

            # 上采样到输入分辨率
            seg_logits = F.interpolate(
                seg_logits,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        return seg_logits

    def load_from(self, weights):
        """
        加载ViT-B_16.npz预训练权重
        """
        logger.info(f"Attempting to load weights. Received type for 'weights' argument: {type(weights)}")
        target_vit_path = r"F:\danzi10\4.5w\lunwen\TransUNet-main - 高性能\model\vit_checkpoint\imagenet21k\ViT-B_16.npz"

        if not os.path.exists(target_vit_path):
            logger.error(f"ERROR: Desired ViT-B_16.npz not found at '{target_vit_path}'.")
            return

        logger.info(f"Loading ViT-B weights from '{target_vit_path}'.")
        try:
            weights_data = np.load(target_vit_path)
        except Exception as e:
            logger.error(f"Failed to load weights from '{target_vit_path}': {e}")
            raise

        # 加载权重到ViT backbone
        model_dict = self.vit_backbone.state_dict()
        loaded_count = 0

        for key in weights_data.files:
            numpy_weight = weights_data[key]

            # 处理不同���权重类型
            if key == "embedding/kernel":
                # Patch embedding weights
                target_key = "patch_embed.proj.weight"
                if target_key in model_dict:
                    # 转换numpy权重格式 [H, W, in_ch, out_ch] -> [out_ch, in_ch, H, W]
                    torch_weight = np2th(numpy_weight, conv=True)
                    if model_dict[target_key].shape == torch_weight.shape:
                        model_dict[target_key] = torch_weight
                        loaded_count += 1

            elif key == "embedding/bias":
                # Patch embedding bias
                target_key = "patch_embed.proj.bias"
                if target_key in model_dict:
                    torch_weight = np2th(numpy_weight)
                    if model_dict[target_key].shape == torch_weight.shape:
                        model_dict[target_key] = torch_weight
                        loaded_count += 1

            elif key == "cls":
                # CLS token
                target_key = "cls_token"
                if target_key in model_dict:
                    torch_weight = np2th(numpy_weight)
                    if model_dict[target_key].shape == torch_weight.shape:
                        model_dict[target_key] = torch_weight
                        loaded_count += 1

            elif key == "Transformer/posembed_input/pos_embedding":
                # Position embedding
                target_key = "pos_embed"
                if target_key in model_dict:
                    torch_weight = np2th(numpy_weight)
                    if model_dict[target_key].shape == torch_weight.shape:
                        model_dict[target_key] = torch_weight
                        loaded_count += 1

            elif "encoderblock" in key:
                # Transformer block weights
                parts = key.split("/")
                block_num = int(parts[1].split("_")[-1])

                if "LayerNorm_0" in key:
                    if "scale" in key:
                        target_key = f"blocks.{block_num}.norm1.weight"
                    elif "bias" in key:
                        target_key = f"blocks.{block_num}.norm1.bias"
                elif "LayerNorm_2" in key:
                    if "scale" in key:
                        target_key = f"blocks.{block_num}.norm2.weight"
                    elif "bias" in key:
                        target_key = f"blocks.{block_num}.norm2.bias"
                elif "MultiHeadDotProductAttention_1" in key:
                    if "query/kernel" in key:
                        target_key = f"blocks.{block_num}.attn.qkv.weight"
                        # 需要特殊处理QKV权重
                        torch_weight = np2th(numpy_weight).flatten(1).T
                        # 这里简化处理，实际可能需要更复杂的权重重组
                    elif "query/bias" in key:
                        target_key = f"blocks.{block_num}.attn.qkv.bias"
                    elif "out/kernel" in key:
                        target_key = f"blocks.{block_num}.attn.proj.weight"
                        torch_weight = np2th(numpy_weight).flatten(1)
                    elif "out/bias" in key:
                        target_key = f"blocks.{block_num}.attn.proj.bias"
                        torch_weight = np2th(numpy_weight)
                elif "MlpBlock_3" in key:
                    if "Dense_0" in key:
                        if "kernel" in key:
                            target_key = f"blocks.{block_num}.mlp.fc1.weight"
                            torch_weight = np2th(numpy_weight)
                        elif "bias" in key:
                            target_key = f"blocks.{block_num}.mlp.fc1.bias"
                            torch_weight = np2th(numpy_weight)
                    elif "Dense_1" in key:
                        if "kernel" in key:
                            target_key = f"blocks.{block_num}.mlp.fc2.weight"
                            torch_weight = np2th(numpy_weight)
                        elif "bias" in key:
                            target_key = f"blocks.{block_num}.mlp.fc2.bias"
                            torch_weight = np2th(numpy_weight)
                else:
                    continue

                # 如果不是QKV特殊情况，进行标准转换
                if 'torch_weight' not in locals():
                    torch_weight = np2th(numpy_weight)

                if target_key in model_dict and model_dict[target_key].shape == torch_weight.shape:
                    model_dict[target_key] = torch_weight
                    loaded_count += 1

                # 清除临时变量
                if 'torch_weight' in locals():
                    del torch_weight

            elif key == "Transformer/encoder_norm/scale":
                target_key = "norm.weight"
                if target_key in model_dict:
                    torch_weight = np2th(numpy_weight)
                    if model_dict[target_key].shape == torch_weight.shape:
                        model_dict[target_key] = torch_weight
                        loaded_count += 1

            elif key == "Transformer/encoder_norm/bias":
                target_key = "norm.bias"
                if target_key in model_dict:
                    torch_weight = np2th(numpy_weight)
                    if model_dict[target_key].shape == torch_weight.shape:
                        model_dict[target_key] = torch_weight
                        loaded_count += 1

        # 加载处理后的权重
        self.vit_backbone.load_state_dict(model_dict, strict=False)
        logger.info(f"Successfully loaded {loaded_count} weight tensors from ViT-B_16.npz into ViT backbone.")


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
