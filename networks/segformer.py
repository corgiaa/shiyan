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


class SegFormerDecoder(nn.Module):
    """
    SegFormer 论文中描述的轻量级 MLP 解码器。
    它接收来自分层 Transformer 编码器（如 MiT）的多尺度特征，
    并将其融合以生成用于分割头的单个特征图。

    **注意：**
    当与非分层式 ViT (如 vit_tiny_patch16_224) 结合使用时，
    传入的 'features' 列表可能包含相同分辨率的特征图。
    在这种情况下，解码器仍会进行线性投影和拼接，但上采样操作将不起作用，
    因为所有特征图已经是相同的目标分辨率。
    """

    def __init__(self, encoder_channels, decoder_hidden_dim=256, out_channels=256):
        super().__init__()
        self.encoder_channels = encoder_channels  # 编码器各阶段的通道数列表
        self.decoder_hidden_dim = decoder_hidden_dim

        # 对每个编码器阶段的特征图进行线性投影（1x1 卷积）
        self.linear_projections = nn.ModuleList()
        for channels in encoder_channels:
            self.linear_projections.append(
                nn.Conv2d(channels, decoder_hidden_dim, kernel_size=1)
            )

        # 最终的 MLP 层用于融合特征
        self.fuse_mlp = nn.Sequential(
            nn.Conv2d(len(encoder_channels) * decoder_hidden_dim, decoder_hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 输出卷积层，将通道数调整为 `out_channels`，以匹配 SegmentationHead 的输入
        self.output_conv = nn.Conv2d(decoder_hidden_dim, out_channels, kernel_size=1)

    def forward(self, features):
        # features 是来自编码器的特征图列表，按分辨率从高到低排序。
        # 例如：[B, C_0, H/4, W/4], [B, C_1, H/8, W/8], [B, C_2, H/16, W/16], [B, C_3, H/32, W/32]

        # 确定上采样的目标分辨率（来自编码器的最高分辨率特征图）
        # 这通常是第一个特征图的分辨率
        # 如果 ViT-T 产生的所有特征图分辨率相同，这里会取第一个的 H, W
        target_h, target_w = features[0].shape[2:]

        processed_features = []
        for i, f in enumerate(features):
            # 1. 应用 1x1 卷积来减少通道数
            proj_f = self.linear_projections[i](f)

            # 2. 上采样到目标分辨率
            # 对于 ViT-T，如果所有特征图分辨率相同，这里实际上不会进行上采样
            if proj_f.shape[2:] != (target_h, target_w):
                proj_f = nn.functional.interpolate(
                    proj_f,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False  # 通常设置为 False 以避免边缘伪影
                )
            processed_features.append(proj_f)

        # 3. 拼接所有处理过并上采样后的特征
        fused_features = torch.cat(processed_features, dim=1)

        # 4. 应用最终的融合 MLP
        fused_features = self.fuse_mlp(fused_features)

        # 5. 应用输出卷积
        output = self.output_conv(fused_features)

        return output


class VisionTransformer(nn.Module):
    """
    此类别已重构为使用 timm 库的 ViT-Tiny/16 (vit_tiny_patch16_224) 作为主干，
    并结合 SegFormer 论文中的 MLP 解码器。

    **重要提示：**
    ViT-Tiny/16 是一个非分层式 Vision Transformer，它不像 SegFormer 的 MiT 主干那样，
    自然地输出多尺度 2D 特征图。为了使其与 SegFormerDecoder 兼容，
    我们在 forward 方法中手动从 ViT 的不同 Transformer 块中提取 token 序列，
    并将其整形为 2D 特征图。
    这意味着所有“多尺度”特征图实际上将具有相同的空间分辨率（例如，14x14）。
    这与 SegFormerDecoder 的原始设计（期望不同分辨率的特征）有所不同，
    可能会影响模型性能。
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
        # *** 修改模型名称为 'vit_tiny_patch16_224' ***
        vit_model_name = "vit_tiny_patch16_224"
        # 不使用 timm 的预训练权重，因为我们要加载本地的 .bin 文件
        vit_pretrained = False

        # 使用 timm.create_model 创建 ViT 主干。
        # 注意：这里不使用 features_only=True，因为我们要手动从块中提取特征，
        # 并且 features_only=True 对于 vanilla ViT 通常返回 token 序列，而不是 2D 特征图。
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

        # --- SegFormer 论文中的 MLP 解码器 ---
        self.decoder = SegFormerDecoder(
            encoder_channels=encoder_channels,
            decoder_hidden_dim=decoder_output_channels,  # 解码器内部处理维度
            out_channels=decoder_output_channels  # 解码器输出到分割头的通道数
        )

        # --- 分割头 ---
        # 分割头接收解码器的输出。
        # ViT-T 的特征图分辨率为 img_size / patch_size (例如 224/16 = 14)。
        # SegFormerDecoder 最终输出的分辨率将是 ViT 最小特征图的分辨率 (这里是 14x14)。
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

        # 检查是否需要对位置嵌入进行插值，以防输入图像大小变化
        # timm 模型的 pos_embed_interpolator 通常在 forward 中处理
        # 简单起见，这里直接使用 timm 的 _pos_embed 方法来处理位置嵌入
        # 或者直接使用原始 ViT 的插值逻辑
        if x.shape[1] != pos_embed.shape[1] - 1:  # -1 for CLS token
            # 原始 ViT 的插值逻辑，timm 内部也有类似处理
            # 这里简化，假设输入尺寸固定或 timm 内部处理
            pass  # timm's _pos_embed handles this automatically if needed

        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)  # (B, num_patches+1, embed_dim)
        # timm 的 ViT 模型通常在 patch_embed 之后和 blocks 之前应用 norm_pre
        # 以及在 pos_embed 之后
        x = x + pos_embed
        x = self.vit_backbone.norm_pre(x)  # Pre-norm layer

        # 3. 通过 Transformer 块，并提取中间特征
        # 我们将从 ViT 的不同块中提取特征，以模拟多尺度输出。
        # 对于 ViT-Tiny/16 (12 个块)，我们可以选择例如第 3, 6, 9, 12 个块的输出。
        # 这些输出都是 (B, num_patches+1, embed_dim) 格式。
        # 我们需要去除 CLS token 并整形为 (B, embed_dim, H_feat, W_feat)。

        features_from_blocks = []
        # 选择 ViT 的不同层来获取“多尺度”特征
        # 这里我们选择最后几个块的输出，并进行整形
        # ViT-Tiny 有 12 个块，我们取 3, 6, 9, 12 块的输出
        block_indices = [2, 5, 8, 11]  # 0-indexed, so blocks 3, 6, 9, 12

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

        # 将这些“多尺度”特征传递给 SegFormer 解码器
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
            return # 无法加载，直接返回

        logger.info(f"Overriding 'weights' argument. Forcing load of PyTorch state_dict from '{target_vit_t_path}'.")
        try:
            # map_location='cpu' 可以确保权重先加载到CPU，避免在某些情况下GPU内存不足的问题
            state_dict_to_load = torch.load(target_vit_t_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Failed to load PyTorch state_dict from '{target_vit_t_path}': {e}")
            raise # 重新抛出异常，因为这是我们期望加载的权重

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
                logger.warning(f"Key '{k}' from loaded weights not found or shape mismatch in timm ViT backbone. Skipping.")

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
