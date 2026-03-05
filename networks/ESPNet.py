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
    它接收解码器的输出特征图，并通过一个 1x1 卷积将其映射到类别数量，
    然后进行双线性上采样以恢复到原始输入图像的分辨率。
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        # 1x1 卷积，用于将解码器输出的特征通道数映射到最终的类别数
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # 如果 upsampling > 1，则应用双线性上采样。否则，使用 Identity（不进行操作）。
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class DilatedResidualBlock(nn.Module):
    """
    一个轻量级块，灵感来源于 ESPNet 等高效架构，使用扩张卷积和残差连接。
    它旨在有效扩大感受野，同时避免显著增加计算成本和改变特征图分辨率。

    结构:
    1.  **瓶颈 (Bottleneck)**: 1x1 卷积，减少通道数。
    2.  **扩张卷积 (Dilated Convolution)**: 3x3 扩张卷积，高效地进行空间特征提取并扩大感受野。
    3.  **恢复 (Restore)**: 1x1 卷积，恢复通道数。
    所有卷积后都跟随 BatchNorm 和 ReLU (除了最后一个 1x1 卷积，其 ReLU 在残差连接后)。
    包含残差连接，如果输入和输出通道数不同，则通过 1x1 卷积进行匹配。
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=2):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2  # 默认使用瓶颈设计，通道数减半

        self.conv_block = nn.Sequential(
            # 1x1 卷积，用于减少通道数 (瓶颈层)
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 3x3 扩张卷积，用于高效地扩大感受野
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 1x1 卷积，用于恢复通道数
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 残差连接：如果输入和输出通道数不同，则使用 1x1 卷积来匹配通道数。
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # 如果通道数相同，则直接是恒等映射

        self.relu = nn.ReLU(inplace=True)  # 用于残差连接后的最终激活

    def forward(self, x):
        identity = self.shortcut(x)  # 获取残差连接的路径
        out = self.conv_block(x)  # 通过卷积块处理
        out += identity  # 添加残差
        return self.relu(out)  # 最终激活


class ESPNetInspiredDecoder(nn.Module):
    """
    一个轻量级解码器，灵感来源于 ESPNet 中的高效空间金字塔 (ESP) 模块，
    旨在融合来自 ViT 主干的同分辨率特征。

    它使用一系列 DilatedResidualBlock 来高效地处理和精炼融合后的特征，
    在不改变特征图分辨率的情况下扩大感受野。

    设计策略：
    1.  **通道对齐**：对每个编码器特征图应用 `1x1 卷积 + BatchNorm + ReLU`，
        将其通道数统一到 `decoder_hidden_dim`。
    2.  **特征拼接**：将所有处理后的特征图沿通道维度拼接起来。
    3.  **初始融合**：通过一个 `1x1 卷积 + BatchNorm + ReLU` 将拼接后的特征通道数
        从 `len(encoder_channels) * decoder_hidden_dim` 减少到 `decoder_hidden_dim`。
        这作为初步的紧凑融合。
    4.  **扩张块精炼**：使用一系列 `DilatedResidualBlock` 进一步处理和增强特征。
        这些块在保持分辨率的同时，通过扩张卷积增加有效感受野。
    5.  **最终输出卷积**：最后通过一个 `1x1 卷积` 将通道数调整到 `out_channels`，
        以匹配 SegmentationHead 的输入要求。
    """

    def __init__(self, encoder_channels, decoder_hidden_dim=256, out_channels=256, num_blocks=2, dilation_rates=[2, 4]):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_hidden_dim = decoder_hidden_dim

        # 1. 通道对齐：对每个编码器特征图进行 1x1 卷积、BatchNorm 和 ReLU
        self.channel_projection_blocks = nn.ModuleList()
        for channels in encoder_channels:
            self.channel_projection_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, decoder_hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(decoder_hidden_dim),
                    nn.ReLU(inplace=True)
                )
            )

        # 2. 初始融合：处理拼接后的特征，将其通道数调整到 decoder_hidden_dim
        fused_in_channels = len(encoder_channels) * decoder_hidden_dim
        self.initial_fusion_conv = nn.Sequential(
            nn.Conv2d(fused_in_channels, decoder_hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 3. 精炼块：使用 DilatedResidualBlock 序列
        self.refinement_blocks = nn.ModuleList()
        for i in range(num_blocks):
            # 循环使用扩张率，以引入不同尺度的上下文信息
            dilation = dilation_rates[i % len(dilation_rates)]
            self.refinement_blocks.append(
                DilatedResidualBlock(decoder_hidden_dim, decoder_hidden_dim, dilation=dilation)
            )

        # 4. 最终输出卷积层，将通道数调整为 `out_channels`
        self.output_conv = nn.Conv2d(decoder_hidden_dim, out_channels, kernel_size=1)

    def forward(self, features):
        processed_features = []
        for i, f in enumerate(features):
            # 1. 应用通道对齐与初步融合模块
            proj_f = self.channel_projection_blocks[i](f)
            processed_features.append(proj_f)

        # 2. 拼接所有处理过的特征图
        fused_features = torch.cat(processed_features, dim=1)

        # 3. 初始融合
        x = self.initial_fusion_conv(fused_features)

        # 4. 通过扩张残差块序列进行精炼
        for block in self.refinement_blocks:
            x = block(x)

        # 5. 应用最终的输出卷积
        output = self.output_conv(x)
        return output


class VisionTransformer(nn.Module):
    """
    此类别已重构为使用 timm 库的 ViT-Tiny/16 (vit_tiny_patch16_224) 作为主干，
    并结合一个基于 ESPNet 论文思想的轻量级解码器。

    **重要提示：**
    ViT-Tiny/16 是一个非分层式 Vision Transformer，它不像 SegFormer 的 MiT 主干那样，
    自然地输出多尺度 2D 特征图。为了使其与 ESPNetInspiredDecoder 兼容，
    我们在 forward 方法中手动从 ViT 的不同 Transformer 块中提取 token 序列，
    并将其整形为 2D 特征图。
    这意味着所有“多尺度”特征图实际上将具有相同的空间分辨率（例如，对于 224x224 输入，它们都是 14x14）。
    ESPNetInspiredDecoder 被设计为能够有效处理这些同分辨率的特征。
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config  # 保留以兼容
        self.img_size = img_size  # 保存 img_size 以便在 forward 中计算特征图分辨率

        in_channels = 3  # 输入图像通道数，通常为RGB的3通道

        # 解码器输出的中间通道数，这将作为 SegmentationHead 的输入通道数。
        decoder_output_channels = 256

        # --- timm ViT-Tiny/16 Backbone 配置 ---
        vit_model_name = "vit_tiny_patch16_224"
        vit_pretrained = False  # 不使用 timm 的预训练权重，因为我们要加载本地的 .bin 文件

        # 使用 timm.create_model 创建 ViT 主干。
        self.vit_backbone = timm.create_model(
            vit_model_name,
            pretrained=vit_pretrained,  # 不从 timm 下载预训练权重
            img_size=img_size,
            drop_path_rate=0.1  # 增加 drop_path_rate 以匹配 TransUNet 训练设置
        )

        # ViT-Tiny/16 的嵌入维度是 192。
        # 我们将从 ViT 的不同块中提取特征，它们的通道数都是 192。
        # 假设我们提取 4 个特征图，它们的通道数都将是 192。
        encoder_channels = [self.vit_backbone.embed_dim] * 4
        logger.info(f"ViT-Tiny/16 encoder feature channels: {encoder_channels}")

        # --- 使用 ESPNetInspiredDecoder ---
        self.decoder = ESPNetInspiredDecoder(
            encoder_channels=encoder_channels,
            decoder_hidden_dim=decoder_output_channels,  # 解码器内部处理维度
            out_channels=decoder_output_channels,  # 解码器输出到分割头的通道数
            num_blocks=2,  # 可以调整解码器中 DilatedResidualBlock 的数量
            dilation_rates=[2, 4]  # DilatedResidualBlock 使用的扩张率
        )

        # --- 分割头 ---
        # 分割头接收解码器的输出。
        # ViT-T 的特征图分辨率为 img_size / patch_size (例如 224/16 = 14)。
        # ESPNetInspiredDecoder 最终输出的分辨率将是 ViT 最小特征图的分辨率 (这里是 14x14)。
        # 因此，需要 (img_size / 14) 倍的上采样。对于 224x224 输入，up_factor = 224/14 = 16。
        patch_size = self.vit_backbone.patch_embed.patch_size[0]  # 通常是 16
        up_factor = img_size // (img_size // patch_size)  # 计算上采样因子
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_output_channels,  # 接收解码器融合后的特征图
            out_channels=num_classes,  # 输出最终的类别数量
            kernel_size=1,
            upsampling=up_factor,  # 上采样到原始输入分辨率
        )

        logger.info(f"Using timm ViT backbone: {vit_model_name} with pretrained={vit_pretrained}")
        logger.info(f"Using ESPNetInspiredDecoder.")
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

        # timm 的 _pos_embed 方法通常在内部处理位置嵌入的插值
        # 这里我们直接使用 timm 的标准流程
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)  # (B, num_patches+1, embed_dim)
        x = x + pos_embed
        x = self.vit_backbone.norm_pre(x)  # Pre-norm layer

        # 3. 通过 Transformer 块，并提取中间特征
        # 我们将从 ViT 的不同块中提取特征，以模拟多尺度输出。
        # 对于 ViT-Tiny/16 (12 个块)，我们选择第 3, 6, 9, 12 个块的输出。
        # 这些输出都是 (B, num_patches+1, embed_dim) 格式。
        # 我们需要去除 CLS token 并整形为 (B, embed_dim, H_feat, W_feat)。

        features_from_blocks = []
        # 选择 ViT 的不同层来获取“多尺度”特征
        # 这里我们选择最后几个块的输出，并进行整形
        # ViT-Tiny 有 12 个块，我们取 0-indexed 的 2, 5, 8, 11 块的输出
        block_indices = [2, 5, 8, 11]

        for i, blk in enumerate(self.vit_backbone.blocks):
            x = blk(x)
            if i in block_indices:
                # 提取非 CLS token 部分
                # x_no_cls: (B, num_patches, embed_dim)
                x_no_cls = x[:, 1:]

                # 整形为 2D 特征图: (B, embed_dim, feat_h, feat_w)
                # 注意：这里所有特征图的分辨率都是相同的
                feature_map = x_no_cls.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
                features_from_blocks.append(feature_map)

        # 将这些“多尺度”特征传递给 ESPNetInspiredDecoder
        # 解码器将融合这些特征并输出一个例如 [B, decoder_output_channels, feat_h, feat_w] 的特征图
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
