# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm, Embedding, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from scipy import ndimage
from einops import rearrange

from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2

logger = logging.getLogger(__name__)

# Weight mapping constants (for ViT encoder, unchanged)
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


# ==========================================
# NEW MODULE: MCAA (Multi-scale Channel Attention Atrous Fusion)
# ==========================================
class MCAA(nn.Module):
    def __init__(self, dim, reduction=8): # Corrected: expects 'dim'
        super(MCAA, self).__init__()
        self.dim = dim

        # 1. 通道注意力分支
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.GELU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

        # 2. 多尺度空间注意力分支
        # These convolutions operate on (B, D, H, W)
        self.spatial_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.spatial_conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        self.spatial_conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=4, dilation=4, groups=dim)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x is expected to be (B, L, D) where L = H*W, D = dim
        B, L, D = x.shape
        H = W = int(math.sqrt(L)) # Assuming square feature maps
        if H * W != L: # Handle non-square or non-perfect square cases
            raise ValueError(f"Spatial length L ({L}) is not a perfect square for H*W. Cannot reshape to 2D. "
                             "MCAA expects input tokens to represent a flattened square feature map.")

        x_reshaped = x.transpose(-1, -2).view(B, D, H, W) # (B, D, H, W)

        # Channel Attention
        c_feat = self.avg_pool(x_reshaped).view(B, D)
        c_attn = self.channel_attn(c_feat).view(B, D, 1, 1)
        x_c = x_reshaped * c_attn

        # Spatial Attention
        s_feat = self.spatial_conv1(x_reshaped) + self.spatial_conv2(x_reshaped) + self.spatial_conv3(x_reshaped)
        s_attn = self.spatial_gate(s_feat)
        x_s = x_reshaped * s_attn

        out = x_c + self.gamma * x_s
        out = out.flatten(2).transpose(-1, -2) # Reshape back to (B, L, D)
        return x + out # Residual connection


# ==========================================
# Original Attention Module (unchanged)
# ==========================================
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


# ==========================================
# Original Mlp Module (unchanged)
# ==========================================
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ==========================================
# Gated Adapter Module (New Innovation)
# ==========================================
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True): # Corrected: expects 'D_features'
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = nn.GELU()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        # 门控机制：学习一个权重来控制Adapter分支的贡献
        self.gate = nn.Linear(D_features, 1)
        self.gate_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 瓶颈结构
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        # 计算门控权重 (基于输入 x 生成门控信号)
        gate_weight = self.gate_sigmoid(self.gate(x))
        xs = xs * gate_weight

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x



# ==========================================
# Modified Block Module (with Gated Adapters)
# ==========================================
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

        # Add Gated Adapters
        # Corrected: Pass config.hidden_size to D_features
        self.attn_adapter = Adapter(config.hidden_size)
        self.ffn_adapter = Adapter(config.hidden_size)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        attn_output, weights = self.attn(x)
        # Apply attention adapter: add adapter output to attention output
        attn_output = attn_output + self.attn_adapter(attn_output)
        x = h + attn_output  # Standard residual connection for attention block

        h = x
        x = self.ffn_norm(x)
        mlp_output = self.ffn(x)
        # Apply FFN adapter: add adapter output to MLP output
        mlp_output = mlp_output + self.ffn_adapter(mlp_output)
        x = h + mlp_output  # Standard residual connection for MLP block

        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            self.attn.query.weight.copy_(
                np2th(weights[f"{ROOT}/{ATTENTION_Q}/kernel"]).view(self.hidden_size, self.hidden_size).t())
            self.attn.key.weight.copy_(
                np2th(weights[f"{ROOT}/{ATTENTION_K}/kernel"]).view(self.hidden_size, self.hidden_size).t())
            self.attn.value.weight.copy_(
                np2th(weights[f"{ROOT}/{ATTENTION_V}/kernel"]).view(self.hidden_size, self.hidden_size).t())
            self.attn.out.weight.copy_(
                np2th(weights[f"{ROOT}/{ATTENTION_OUT}/kernel"]).view(self.hidden_size, self.hidden_size).t())
            self.attn.query.bias.copy_(np2th(weights[f"{ROOT}/{ATTENTION_Q}/bias"]).view(-1))
            self.attn.key.bias.copy_(np2th(weights[f"{ROOT}/{ATTENTION_K}/bias"]).view(-1))
            self.attn.value.bias.copy_(np2th(weights[f"{ROOT}/{ATTENTION_V}/bias"]).view(-1))
            self.attn.out.bias.copy_(np2th(weights[f"{ROOT}/{ATTENTION_OUT}/bias"]).view(-1))
            self.ffn.fc1.weight.copy_(np2th(weights[f"{ROOT}/{FC_0}/kernel"]).t())
            self.ffn.fc2.weight.copy_(np2th(weights[f"{ROOT}/{FC_1}/kernel"]).t())
            self.ffn.fc1.bias.copy_(np2th(weights[f"{ROOT}/{FC_0}/bias"]).t())
            self.ffn.fc2.bias.copy_(np2th(weights[f"{ROOT}/{FC_1}/bias"]).t())
            self.attention_norm.weight.copy_(np2th(weights[f"{ROOT}/{ATTENTION_NORM}/scale"]))
            self.attention_norm.bias.copy_(np2th(weights[f"{ROOT}/{ATTENTION_NORM}/bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[f"{ROOT}/{MLP_NORM}/scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[f"{ROOT}/{MLP_NORM}/bias"]))
            # Adapters are new layers and are not part of the pre-trained weights,
            # so they are not loaded here. They will be randomly initialized.


# ==========================================
# Modified Embeddings Module (with MCAA)
# ==========================================
class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)

        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        in_channels_resnet = self.hybrid_model.width * 16

        # ==========================================
        # INSERT MCAA HERE
        # ==========================================
        # We insert MCAA after ResNet features but before patch embedding.
        # This enriches the features spatially before they are tokenized.
        # Corrected: Pass in_channels_resnet to 'dim' parameter
        self.mcaa = MCAA(dim=in_channels_resnet)
        # ==========================================

        # For TransUNet's hybrid model, the ViT's patch_embeddings is a 1x1 Conv2d
        # applied to the 1/16th downsampled feature map from ResNetV2.
        patch_size_conv = (1, 1)  # Kernel size for the Conv2d

        h_feat, w_feat = img_size[0] // 16, img_size[1] // 16  # Output resolution of ResNetV2 stem

        # Calculate n_patches based on the actual feature map dimensions
        n_patches = h_feat * w_feat  # e.g., 14 * 14 = 196 for 224x224 input

        self.patch_embeddings = Conv2d(in_channels=in_channels_resnet,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size_conv,
                                       stride=patch_size_conv)  # Stride also (1,1)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.n_patches = n_patches
        self.h_feat, self.w_feat = h_feat, w_feat

    def forward(self, x):
        x_resnet, _ = self.hybrid_model(x) # x_resnet shape: (B, C_resnet, H_feat, W_feat)

        # Corrected: Reshape x_resnet for MCAA (B, C, H, W) -> (B, H*W, C)
        B, C_resnet, H_feat, W_feat = x_resnet.shape
        x_resnet_tokens = x_resnet.flatten(2).transpose(-1, -2) # (B, H_feat*W_feat, C_resnet)

        # Apply MCAA Fusion
        mcaa_output_tokens = self.mcaa(x_resnet_tokens) # (B, H_feat*W_feat, C_resnet)

        # Corrected: Reshape MCAA output back to (B, C, H, W) for patch_embeddings
        x_after_mcaa = mcaa_output_tokens.transpose(-1, -2).view(B, C_resnet, H_feat, W_feat)

        x = self.patch_embeddings(x_after_mcaa) # This expects (B, C, H, W)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


# ==========================================
# Original Encoder Module (unchanged)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights  # encoded is (B, N, C)


# ==========================================
# SAM-like Image Encoder (reusing Transformer class)
# ==========================================
class ImageEncoder(nn.Module):
    def __init__(self, config, img_size, vis):
        super(ImageEncoder, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)
        self.config = config
        self.img_size = img_size

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)  # (B, N, C)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, N, C)

        # Reshape encoded tokens back to a 2D feature map for the mask decoder
        B, N, C = encoded.shape
        H_feat, W_feat = self.embeddings.h_feat, self.embeddings.w_feat
        image_features = encoded.permute(0, 2, 1).contiguous().view(B, C, H_feat, W_feat)  # (B, C, H_feat, W_feat)

        return image_features, attn_weights  # Return 2D feature map and attention weights


# ==========================================
# SAM-like Prompt Encoder (Simplified)
# ==========================================
class PromptEncoder(nn.Module):
    def __init__(self, config, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.point_embeddings = Embedding(4, embed_dim)

        self.box_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),  # (x1, y1, x2, y2)
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.no_prompt_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, point_coords=None, point_labels=None, box_coords=None,
                batch_size: int = 1):  # <--- ADDED batch_size
        """
        Encodes prompts into embeddings.
        Args:
            point_coords (torch.Tensor): (B, N_points, 2)
            point_labels (torch.Tensor): (B, N_points) (0: negative, 1: positive, -1: padding)
            box_coords (torch.Tensor): (B, N_boxes, 4)
            batch_size (int): The actual batch size of the input image. <--- NEW
        Returns:
            torch.Tensor: Combined prompt embeddings (B, N_total_prompts, embed_dim)
        """
        sparse_embeddings = []
        if point_coords is not None and point_labels is not None:
            # batch_size is now passed, no need to infer here from prompt tensors
            point_labels_mapped = point_labels.clone()
            point_labels_mapped[point_labels_mapped == -1] = 0  # Padding token
            point_labels_mapped[point_labels_mapped == 0] = 2  # Background token
            point_labels_mapped[point_labels_mapped == 1] = 1  # Foreground token

            point_embeds = self.point_embeddings(point_labels_mapped)  # (B, N_points, embed_dim)
            sparse_embeddings.append(point_embeds)

        if box_coords is not None:
            # batch_size is now passed, no need to infer here from prompt tensors
            box_embeds = self.box_encoder(box_coords.float())  # (B, N_boxes, embed_dim)
            sparse_embeddings.append(box_embeds)

        if len(sparse_embeddings) == 0:
            # If no prompts, use the provided batch_size
            return self.no_prompt_embed.repeat(batch_size, 1, 1)

        return torch.cat(sparse_embeddings, dim=1)


# ==========================================
# SAM-like Mask Decoder (Simplified)
# ==========================================
class MaskDecoder(nn.Module):
    def __init__(self, config, image_feature_dim, prompt_embed_dim, num_mask_tokens=1):
        super().__init__()
        self.image_feature_dim = image_feature_dim
        self.prompt_embed_dim = prompt_embed_dim
        self.num_mask_tokens = num_mask_tokens

        self.mask_tokens = nn.Parameter(torch.randn(1, num_mask_tokens, prompt_embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=prompt_embed_dim,
            nhead=config.transformer["num_heads"],
            dim_feedforward=config.transformer["mlp_dim"],
            dropout=config.transformer["dropout_rate"],
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.image_feature_projection = Conv2d(image_feature_dim, prompt_embed_dim, kernel_size=1)

        self.mask_upsampling_layers = nn.Sequential(
            # First upsample layer: from prompt_embed_dim to prompt_embed_dim // 4
            nn.ConvTranspose2d(prompt_embed_dim, prompt_embed_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(),
            # Second upsample layer: from prompt_embed_dim // 4 to prompt_embed_dim
            # This ensures the channel dimension matches prompt_embed_dim for bmm
            nn.ConvTranspose2d(prompt_embed_dim // 4, prompt_embed_dim, kernel_size=2, stride=2),
            nn.ReLU(),
            # Final Conv2d to produce the mask logits, now takes prompt_embed_dim channels as input
            nn.Conv2d(prompt_embed_dim, num_mask_tokens, kernel_size=1)
        )

    def forward(self, image_features, prompt_embeddings, original_image_size):
        """
        Args:
            image_features (torch.Tensor): (B, C_img, H_img, W_img) from image encoder
            prompt_embeddings (torch.Tensor): (B, N_prompts, C_prompt) from prompt encoder
            original_image_size (tuple): (H, W) of the original input image
        Returns:
            torch.Tensor: Mask logits (B, num_mask_tokens, H_orig, W_orig)
        """
        B, C_img, H_img, W_img = image_features.shape

        image_features_proj = self.image_feature_projection(image_features)  # (B, prompt_embed_dim, H_img, W_img)

        image_features_flat = image_features_proj.view(B, self.prompt_embed_dim, -1).permute(0, 2,
                                                                                             1)  # (B, H_img*W_img, prompt_embed_dim)

        mask_tokens = self.mask_tokens.repeat(B, 1, 1)  # (B, num_mask_tokens, prompt_embed_dim)

        decoder_memory = torch.cat([prompt_embeddings, image_features_flat], dim=1)

        refined_mask_tokens = self.transformer_decoder(mask_tokens,
                                                       decoder_memory)  # (B, num_mask_tokens, prompt_embed_dim)

        # Upsample the projected image features to an intermediate resolution
        # After the fix, intermediate_upsampled_features will have prompt_embed_dim channels
        intermediate_upsampled_features = self.mask_upsampling_layers[0:4](
            image_features_proj)  # (B, prompt_embed_dim, H_img*4, W_img*4)

        # Combine with refined_mask_tokens
        # (B, num_mask_tokens, prompt_embed_dim) @ (B, prompt_embed_dim, H_inter*W_inter)
        final_mask_logits_inter = torch.bmm(refined_mask_tokens,
                                            intermediate_upsampled_features.view(B, self.prompt_embed_dim,
                                                                                 -1))
        final_mask_logits_inter = final_mask_logits_inter.view(B, self.num_mask_tokens,
                                                               intermediate_upsampled_features.shape[2],
                                                               intermediate_upsampled_features.shape[3])

        # Finally, interpolate to the original image size
        mask_logits = F.interpolate(final_mask_logits_inter,
                                    size=original_image_size,
                                    mode='bilinear',
                                    align_corners=False)

        return mask_logits


# ==========================================
# Modified VisionTransformer Module (now SAM-like)
# ==========================================
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=1024, num_classes=1, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.config = config
        self.img_size = img_size
        self.num_classes = num_classes

        self.image_encoder = ImageEncoder(config, img_size, vis)

        self.prompt_encoder = PromptEncoder(config, embed_dim=config.hidden_size)

        self.mask_decoder = MaskDecoder(config,
                                        image_feature_dim=config.hidden_size,
                                        prompt_embed_dim=config.hidden_size,
                                        num_mask_tokens=num_classes)

    def forward(self, image, point_coords=None, point_labels=None, box_coords=None, mask_input=None):
        """
        SAM-like forward pass.
        Args:
            image (torch.Tensor): Input image (B, C, H, W)
            point_coords (torch.Tensor): Point coordinates (B, N_points, 2)
            point_labels (torch.Tensor): Point labels (B, N_points) (0: negative, 1: positive, -1: padding)
            box_coords (torch.Tensor): Bounding box coordinates (B, N_boxes, 4)
            mask_input (torch.Tensor): Optional mask input (B, 1, H_lowres, W_lowres) - NOT IMPLEMENTED IN THIS SIMPLIFIED VERSION
        Returns:
            torch.Tensor: Predicted mask logits (B, num_classes, H_orig, W_orig)
        """
        original_image_size = image.shape[2:]  # (H, W)
        batch_size = image.shape[0]  # <--- Get batch size from image

        # 1. Image Encoding
        image_features, _ = self.image_encoder(image)

        # 2. Prompt Encoding
        # Pass the batch_size to the prompt_encoder
        prompt_embeddings = self.prompt_encoder(
            point_coords=point_coords,
            point_labels=point_labels,
            box_coords=box_coords,
            batch_size=batch_size  # <--- Pass batch_size here
        )

        # 3. Mask Decoding
        mask_logits = self.mask_decoder(image_features, prompt_embeddings, original_image_size)

        return mask_logits

    def load_from(self, weights):
        logger.info("Loading weights for Image Encoder (ResNetV2 + ViT parts).")
        with torch.no_grad():
            self.image_encoder.embeddings.hybrid_model.root.conv.weight.copy_(
                np2th(weights["conv_root/kernel"], conv=True))
            gn_weight = np2th(weights["gn_root/scale"]).view(-1)
            gn_bias = np2th(weights["gn_root/bias"]).view(-1)
            self.image_encoder.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
            self.image_encoder.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

            for bname, block in self.image_encoder.embeddings.hybrid_model.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)

            self.image_encoder.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.image_encoder.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.image_encoder.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.image_encoder.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.image_encoder.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.image_encoder.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                self.image_encoder.embeddings.position_embeddings.copy_(posemb[:, 1:])
            else:
                logger.info(f"Interpolating positional embeddings from {posemb.size()} to {posemb_new.size()}")
                ntok_new = posemb_new.size(1)
                posemb_grid = posemb[0, 1:] if posemb.size(1) > ntok_new else posemb[0, :]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))

                if gs_old == 0:
                    logger.warning("Original positional embedding grid size is zero, cannot interpolate.")
                    self.image_encoder.embeddings.position_embeddings.zero_()
                else:
                    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                    # Add a check for zoom factors being zero to prevent errors in ndimage.zoom
                    if zoom[0] == 0 or zoom[1] == 0:
                        logger.warning(
                            f"New positional embedding grid size is zero ({gs_new}x{gs_new}), cannot interpolate. Initializing with zeros.")
                        self.image_encoder.embeddings.position_embeddings.zero_()
                    else:
                        posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                        self.image_encoder.embeddings.position_embeddings.copy_(np2th(posemb_grid))

        logger.info(
            "Prompt Encoder and Mask Decoder are not loaded from these weights and require separate initialization/training.")


CONFIGS = {
    'R50-ViT-B_16': configs.get_r50_b16_config(),
}