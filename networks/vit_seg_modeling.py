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
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from einops import rearrange

from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2

logger = logging.getLogger(__name__)

# Weight mapping constants
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
# 创新模块: Multi-Scale Channel-Axial Attention (MCAA)
# ==========================================
class MCAA(nn.Module):
    def __init__(self, dim, reduction=8):
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
        self.spatial_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.spatial_conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        self.spatial_conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=4, dilation=4, groups=dim)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, L, D = x.shape
        H = W = int(math.sqrt(L))
        x_reshaped = x.transpose(-1, -2).view(B, D, H, W)

        c_feat = self.avg_pool(x_reshaped).view(B, D)
        c_attn = self.channel_attn(c_feat).view(B, D, 1, 1)
        x_c = x_reshaped * c_attn

        s_feat = self.spatial_conv1(x_reshaped) + self.spatial_conv2(x_reshaped) + self.spatial_conv3(x_reshaped)
        s_attn = self.spatial_gate(s_feat)
        x_s = x_reshaped * s_attn

        out = x_c + self.gamma * x_s
        out = out.flatten(2).transpose(-1, -2)
        return x + out


# ==========================================
# 修改后的模块: Gated Adapter (门控 Adapter)
# ==========================================
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True):
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


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

        self.adapter_scale = getattr(config, 'adapter_scale', 0.5)
        self.thd = getattr(config, 'thd', False)

        # 这里使用的 Adapter 现在是带有门控机制的
        self.MLP_Adapter = Adapter(config.hidden_size, skip_connect=False)
        self.Space_Adapter = Adapter(config.hidden_size, skip_connect=True)
        self.mcaa = MCAA(config.hidden_size)

        if self.thd:
            self.Depth_Adapter = Adapter(config.hidden_size, skip_connect=False)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)

        attn_out, weights = self.attn(x)
        attn_out = self.mcaa(attn_out)
        x = self.Space_Adapter(attn_out)

        if self.thd:
            xd = self.Depth_Adapter(attn_out)
            x = x + xd

        x = x + h

        h = x
        x = self.ffn_norm(x)

        mlp_out = self.ffn(x)
        adapter_out = self.MLP_Adapter(x)
        # 结合原有的 adapter_scale 和内部的门控
        x = h + mlp_out + self.adapter_scale * adapter_out

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


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])

        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


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
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = list(self.config.skip_channels)
        if self.config.n_skip != 0:
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
                  zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                np2th(weights["conv_root/kernel"], conv=True))
            gn_weight = np2th(weights["gn_root/scale"]).view(-1)
            gn_bias = np2th(weights["gn_root/bias"]).view(-1)
            self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
            self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

            for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                self.transformer.embeddings.position_embeddings.copy_(posemb[:, 1:])
            else:
                ntok_new = posemb_new.size(1)
                posemb_grid = posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb_grid))

            for i, block in enumerate(self.transformer.encoder.layer):
                block.load_from(weights, n_block=i)


CONFIGS = {
    'R50-ViT-B_16': configs.get_r50_b16_config(),
}