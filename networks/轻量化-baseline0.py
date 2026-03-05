# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os

import timm  # 引入 timm 库，用于 Vision Transformer (ViT) 主干
import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, MaxPool2d, ConvTranspose2d, Identity
from torch.nn.modules.utils import _pair
from scipy import ndimage

# 保留原始的ViT配置导入，尽管它们现在主要用于兼容性
# 假设 vit_seg_configs 模块在当前目录或 Python 路径中
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
        # 如果 upsampling > 1，则应用双线性上采样。否则，使用 Identity。
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


# --- 新增：轻量级 TransUNet 风格解码器 ---
class LightweightTransUNetDecoder(nn.Module):
    """
    一个轻量级的 TransUNet 风格解码器。
    它接收来自 ViT 编码器的多个特征图（这些特征图具有相同的空间分辨率），
    将它们融合，并通过一系列上采样和卷积层逐步恢复分辨率。

    由于 ViT 编码器（如 vit_tiny_patch16_224）不产生多尺度特征金字塔，
    这里的“跳跃连接”是特征在相同分辨率下的融合，然后进行逐步上采样。
    """

    def __init__(self, encoder_channels, decoder_hidden_dim=128, out_channels=128):
        super().__init__()
        # encoder_channels 是一个列表，但对于 vanilla ViT，所有通道数都相同。
        # 我们将所有特征图在通道维度上拼接，所以输入通道是所有 encoder_channels 的和。
        fused_input_channels = sum(encoder_channels)

        # 1. 初始融合和通道数调整
        # 将所有来自 ViT 的特征图拼接后，通过 1x1 卷积减少通道数到 decoder_hidden_dim。
        self.initial_fusion = nn.Sequential(
            nn.Conv2d(fused_input_channels, decoder_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 上采样路径
        # 假设 ViT 输出特征图是 H/16 x W/16 (例如 14x14)。
        # 我们需要逐步上采样，例如 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224。
        # 这里的解码器负责将特征图从 H/16 x W/16 上采样到 H/4 x W/4 (或 H/2 x W/2)，
        # 最终的上采样到 H x W 由 SegmentationHead 完成。
        # 为了轻量化，我们只进行少量上采样，让 SegmentationHead 完成大部分工作。
        # 这里的目标是输出一个比 ViT 特征图分辨率更高一些的特征图，但仍低于原始输入分辨率。
        # 例如，从 H/16 到 H/4 (2次2x上采样)

        self.up_blocks = nn.ModuleList([
            # 第一次上采样 (例如 14x14 -> 28x28)
            nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(decoder_hidden_dim, decoder_hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_hidden_dim // 2),
                nn.ReLU(inplace=True),
            ),
            # 第二次上采样 (例如 28x28 -> 56x56)
            nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(decoder_hidden_dim // 2, decoder_hidden_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_hidden_dim // 4),
                nn.ReLU(inplace=True),
            )
        ])

        # 调整 out_channels 以匹配最后一个上采样块的输出
        final_decoder_channels = decoder_hidden_dim // (2 ** len(self.up_blocks))
        if final_decoder_channels == 0:  # 避免通道数为0
            final_decoder_channels = 1  # 至少为1

        # 3. 最终输出卷积
        # 将通道数调整为 `out_channels`，以匹配 SegmentationHead 的输入
        self.output_conv = nn.Conv2d(final_decoder_channels, out_channels, kernel_size=1)

        # 记录实际的输出通道数，供外部使用
        self.out_channels = out_channels

    def forward(self, features):
        # features 是来自编码器的特征图列表，所有特征图都具有相同的空间分辨率。
        # 例如：[B, C_vit, H/16, W/16], [B, C_vit, H/16, W/16], ...

        if not features:
            raise ValueError("LightweightTransUNetDecoder received an empty list of features.")

        # 1. 拼接所有特征图
        # (B, C_vit*num_features, H_feat, W_feat)
        fused_features = torch.cat(features, dim=1)

        # 2. 初始融合和通道数调整
        x = self.initial_fusion(fused_features)

        # 3. 逐步上采样
        for up_block in self.up_blocks:
            x = up_block(x)

        # 4. 最终输出卷积
        output = self.output_conv(x)

        return output


class VisionTransformer(nn.Module):
    """
    此类别已重构为使用 timm 库的 ViT-Tiny/16 (vit_tiny_patch16_224) 作为主干，
    并结合轻量级的 TransUNet 风格解码器。

    **重要提示：**
    ViT-Tiny/16 是一个非分层式 Vision Transformer，它不像 SegFormer 的 MiT 主干那样，
    自然地输出多尺度 2D 特征图。为了使其与 TransUNet 风格解码器兼容，
    我们在 forward 方法中手动从 ViT 的不同 Transformer 块中提取 token 序列，
    并将其整形为 2D 特征图。
    这意味着所有“多尺度”特征图实际上将具有相同的空间分辨率（例如，14x14）。
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config  # 保留以兼容
        self.img_size = img_size  # 保存 img_size 以便在 forward 中计算特征图分辨率

        in_channels = 3  # 输入图像通道数，通常为RGB的3通道

        # 解码器输出的中间通道数，这将作为 SegmentationHead 的输入通道数。
        # 保持与 SegFormerDecoder 相似的输出通道数，但解码器内部可以更小。
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

        # ViT-Tiny/16 的嵌入维度是 192。
        # 我们将从 ViT 的不同块中提取特征，它们的通道数都是 192。
        # 假设我们提取 4 个特征图，它们的通道数都将是 192。
        encoder_channels = [self.vit_backbone.embed_dim] * 4
        logger.info(f"ViT-Tiny/16 encoder feature channels: {encoder_channels}")

        # --- 轻量级 TransUNet 风格解码器 ---
        # 这里的 decoder_hidden_dim 设置为更小的值，以减少参数量
        self.decoder = LightweightTransUNetDecoder(
            encoder_channels=encoder_channels,
            decoder_hidden_dim=128,  # 显著减小解码器内部的隐藏维度，以实现轻量化
            out_channels=decoder_output_channels  # 解码器输出到分割头的通道数
        )

        # --- 分割头 ---
        # 分割头接收解码器的输出。
        # ViT-T 的特征图分辨率为 img_size / patch_size (例如 224/16 = 14)。
        # LightweightTransUNetDecoder 会将特征图上采样多次。
        # 假设 ViT 原始输出分辨率是 H_feat x W_feat (例如 14x14)。
        # 解码器进行了 2 次 2x 上采样，所以输出分辨率是 H_feat * 4 x W_feat * 4 (例如 56x56)。
        # 那么，分割头需要从 H_feat * 4 x W_feat * 4 上采样到 img_size x img_size。
        # up_factor = img_size / (H_feat * 4) = img_size / (img_size/patch_size * 4) = patch_size / 4
        patch_size = self.vit_backbone.patch_embed.patch_size[0]  # 通常是 16

        # 计算解码器输出后的特征图分辨率
        initial_feat_h = img_size // patch_size  # 例如 14
        final_decoder_feat_h = initial_feat_h * (2 ** len(self.decoder.up_blocks))  # 例如 14 * 4 = 56

        up_factor = img_size // final_decoder_feat_h  # 例如 224 / 56 = 4

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,  # 接收解码器融合后的特征图
            out_channels=num_classes,  # 输出最终的类别数量
            kernel_size=1,
            upsampling=up_factor,  # 上采样到原始输入分辨率
        )

        logger.info(f"Using timm ViT backbone: {vit_model_name} with pretrained={vit_pretrained}")
        if not vit_pretrained:
            logger.warning(
                "ViT backbone is not using timm's pretrained weights. Local weights will be loaded via load_from().")

    def forward(self, x):
        # 确保输入是3通道；如果单通道则复制
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B, C, H, W = x.shape
        patch_size = self.vit_backbone.patch_embed.patch_size[0]
        # 计算特征图的 H 和 W (例如 224/16 = 14)
        feat_h, feat_w = H // patch_size, W // patch_size

        # 通过 ViT 主干提取特征
        # 1. 补丁嵌入
        x = self.vit_backbone.patch_embed(x)  # (B, num_patches, embed_dim)

        # 2. 添加位置嵌入和 CLS token
        cls_token = self.vit_backbone.cls_token
        pos_embed = self.vit_backbone.pos_embed

        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)  # (B, num_patches+1, embed_dim)
        x = x + pos_embed
        x = self.vit_backbone.norm_pre(x)  # Pre-norm layer

        # 3. 通过 Transformer 块，并提取中间特征
        features_from_blocks = []
        # 选择 ViT 的不同层来获取“多尺度”特征
        # ViT-Tiny 有 12 个块，我们取 3, 6, 9, 12 块的输出
        # (0-indexed, so blocks 2, 5, 8, 11)
        block_indices = [2, 5, 8, 11]

        for i, blk in enumerate(self.vit_backbone.blocks):
            x = blk(x)
            if i in block_indices:
                # 提取非 CLS token 部分
                x_no_cls = x[:, 1:]

                # 整形为 2D 特征图: (B, embed_dim, feat_h, feat_w)
                # 注意：这里所有特征图的分辨率都是相同的
                feature_map = x_no_cls.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
                features_from_blocks.append(feature_map)

        # 将这些“多尺度”特征传递给轻量级 TransUNet 风格解码器
        # 解码器将融合这些特征并输出一个例如 [B, decoder_output_channels, H_feat*4, W_feat*4] 的特征图
        decoded_features = self.decoder(features_from_blocks)

        # 通过分割头生成最终的 logits。
        # 分割头将进行上采样，以匹配原始输入分辨率。
        logits = self.segmentation_head(decoded_features)  # 输出为 [B, num_classes, H, W]
        return logits

    def load_from(self, weights):
        """
        从预设路径强制加载 ViT-T_16.bin 权重到 timm ViT 主干。
        此方法忽略传入的 'weights' 参数，因为 main1.py 不可修改，
        且其当前行为是加载不兼容的 .npz 文件。

        Args:
            weights: 理论上是预训练权重，但在此方法中会被忽略。
                     传入的可能是 np.lib.npyio.NpzFile 对象。
        """
        logger.info(f"Attempting to load weights. Received type for 'weights' argument: {type(weights)}")

        # --- 强制加载 ViT-T_16.bin ---
        # 这是为了绕过 main1.py 中固定或错误的 pretrained_path 设置。
        # 请确保此路径与您的 ViT-T_16.bin 文件实际位置一致。
        # 假设 'model/vit_checkpoint/imagenet21k/' 是相对于您的项目根目录的路径。
        target_vit_t_path = "model/vit_checkpoint/imagenet21k/ViT-T_16.bin"

        state_dict_to_load = None

        if not os.path.exists(target_vit_t_path):
            logger.error(f"ERROR: Desired ViT-T_16.bin not found at '{target_vit_t_path}'.")
            logger.error("Please ensure the file exists at the specified path.")
            logger.error("Since main1.py cannot be modified, this model will start without pre-trained weights.")
            return  # 无法加载，直接返回

        logger.info(f"Overriding 'weights' argument. Forcing load of PyTorch state_dict from '{target_vit_t_path}'.")
        try:
            # map_location='cpu' 可以确保权重先加载到CPU，避免在某些情况下GPU内存不足的问题
            state_dict_to_load = torch.load(target_vit_t_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Failed to load PyTorch state_dict from '{target_vit_t_path}': {e}")
            raise  # 重新抛出异常，因为这是我们期望加载的权重

        # Handle potential wrapping of state_dict (e.g., if it's {'state_dict': actual_weights})
        if 'state_dict' in state_dict_to_load and isinstance(state_dict_to_load['state_dict'], dict):
            state_dict_to_load = state_dict_to_load['state_dict']

        new_state_dict = {}
        model_state_dict = self.vit_backbone.state_dict()

        for k, v in state_dict_to_load.items():
            # 调整键名以匹配 timm 模型。
            # 预训练模型可能带有 'model.' 或 'encoder.' 等前缀，需要移除。
            if k.startswith('model.'):
                k = k[len('model.'):]
            if k.startswith('encoder.'):
                k = k[len('encoder.'):]

            # 检查键是否存在且形状匹配
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                new_state_dict[k] = v
            # 忽略分类头权重或其他与分割任务无关的层
            elif 'head' in k or 'norm.0.weight' in k or 'norm.0.bias' in k:
                logger.debug(f"Skipping head/irrelevant weight: {k}")
                continue
            else:
                logger.warning(
                    f"Key '{k}' from loaded weights not found or shape mismatch in timm ViT backbone. Skipping.")

        # 加载过滤后的状态字典，允许不严格匹配（忽略缺失的键）
        self.vit_backbone.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Successfully loaded '{target_vit_t_path}' into timm ViT backbone (strict=False).")


# 原始的 ViT 配置字典保持不变，主要用于兼容性。
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