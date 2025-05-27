# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import xformers
import xformers.ops
from xformers.components.feedforward import MLP
from xformers.components import Activation
from inspect import isfunction
from einops import rearrange

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
 
class PointEmbed(nn.Module):
    """Point Embedding (keep the original argument list for compatibility)
    """
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv1d(in_chans, embed_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Point embedding
        B, Cin, N = x.shape

        # Point embedding
        point_emb = self.proj(x)
        
        return self.relu(point_emb)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype
    
class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs//s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=False):
        """
        Drops labels to enable classifier-free guidance.
        """
        if not force_drop_ids:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = torch.ones(labels.shape[0]).bool().cuda()
        labels = torch.where(drop_ids[:, None], self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=False):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)

class ConditionEmbedder(nn.Module):
    """
    Embeds conditions into fourier features.
    """
    def __init__(self, hidden_size, uncond_prob, num_conditions, num_cyclic_conditions, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.num_cyclic_conditions = num_cyclic_conditions
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(num_conditions, hidden_size) / hidden_size ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, condition, force_drop_ids=False):
        """
        Drops labels to enable classifier-free guidance.
        """
        if not force_drop_ids:
            drop_ids = torch.rand(condition.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = torch.ones(condition.shape[0]).bool().cuda()
        condition = torch.where(drop_ids[:, None, None], self.y_embedding, condition)
        return condition

    @staticmethod
    def cyclic_embedding(condition, dim):
        batch_size, _ = condition.shape
        half_dim = dim // 2

        frequencies = (- torch.arange(0, half_dim) * np.log(10000) / (half_dim - 1)).exp()
        frequencies = frequencies[None, None, :].repeat(batch_size, 1, 1).cuda()

        sin_sin_emb = ((condition[:, :, None]).sin() * frequencies).sin()
        sin_cos_emb = ((condition[:, :, None]).cos() * frequencies).sin()
        emb = torch.cat([sin_sin_emb, sin_cos_emb], dim=2)
        
        if dim % 2:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        return emb
    
    @staticmethod
    def positional_embedding(condition, dim):
        half_dim = dim // 2
        emb = np.ones(condition.shape[0]) * np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb[:,None])).float().to(torch.device('cuda'))
        emb = condition[:, :, None] * emb[:, None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=2)
        if dim % 2:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        return emb

    def forward(self, c, train, force_drop_ids=None):
        c_freq_cyc = self.cyclic_embedding(c[:, :self.num_cyclic_conditions], self.frequency_embedding_size)
        c_freq_pos = self.positional_embedding(c[:, self.num_cyclic_conditions:], self.frequency_embedding_size)
        c_freq = torch.concat((c_freq_cyc, c_freq_pos), 1)
        c_emb = self.mlp(c_freq)
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids):
            c_emb = self.token_drop(c_emb, force_drop_ids)
        return c_emb
    
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=qkv_bias)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            n_heads: int,
            d_head: int = 64,
            dropout=0.0,
            context_dim=None,
            gated_ff: bool = True,
            checkpoint: bool = True,
            qkv_bias=False
    ):
        super().__init__()
        self.attn1 = MemoryEfficientCrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, qkv_bias=qkv_bias)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = MemoryEfficientCrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, qkv_bias=False)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def forward(self, hidden_states, context=None):
        hidden_states = hidden_states.contiguous() if hidden_states.device.type == "mps" else hidden_states
        hidden_states = self.attn1(hidden_states) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class LOGenBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, context_dim=None, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        self.attn = BasicTransformerBlock(dim=hidden_size, n_heads=num_heads, context_dim=context_dim, qkv_bias=True)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp = MLP(dim_model=hidden_size, hidden_layer_multiplier=int(mlp_ratio),
                       activation=Activation("gelu"), dropout=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, context=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), context=context)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LOGen(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        context_dim=768,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        num_cyclic_conditions=1,
        num_total_conditions=6,
        uncond_prob=.1,
        num_classes=1,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.x_embedder = PointEmbed(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, uncond_prob)

        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        self.c_embedder = ConditionEmbedder(hidden_size, uncond_prob, num_conditions=num_total_conditions, num_cyclic_conditions=num_cyclic_conditions)

        self.secondary_device = torch.device("cpu")

        self.blocks = [
            LOGenBlock(hidden_size, num_heads, context_dim=context_dim, mlp_ratio=mlp_ratio) for _ in range(depth)
        ]

        self.initialize_weights()
        self.blocks = nn.ModuleList(self.blocks)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize label embedding table:
        if self.num_classes > 1:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize condition embedding MLP:
        nn.init.normal_(self.c_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.c_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LOGen blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, c, force_dropout=False):
        """
        Forward pass of LOGen.
        x: (B, max_points_in_batch, input_channels) tensor of point inputs
        t: (B,) tensor of diffusion timesteps
        y: (B, num_classes) tensor of class labels (uneeded for single class)
        c: (B, 6) tensor of conditions (angle, distance, z, l, w, h)
        """
        x = self.x_embedder(x)  
        t = self.t_embedder(t)
        c = self.c_embedder(c, self.training, force_drop_ids=force_dropout)

        if self.num_classes > 1:
            y = self.y_embedder(y, self.training, force_drop_ids=force_dropout)
            context = torch.cat((y, c), dim=1)
        else:
            context = c

        x = x.transpose(-1,-2)
        for block in self.blocks:
            x = block(x, t, context)

        x = self.final_layer(x, t)
        del t

        return x.transpose(-1,-2)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
def LOGen_L_4(pretrained=False, **kwargs):
    return LOGen(depth=24, hidden_size=1152, context_dim=1152, num_heads=16, **kwargs)

def LOGen_B_4(pretrained=False, **kwargs):
    return LOGen(depth=12, hidden_size=768, context_dim=768, num_heads=12, **kwargs)

def LOGen_XS_4(pretrained=False, **kwargs):
    return LOGen(depth=12, hidden_size=192, context_dim=192, num_heads=3, **kwargs)

LOGen_models = {
    'DiT-XS/4':  LOGen_XS_4,
    'DiT-B/4': LOGen_B_4,
    'DiT-L/4': LOGen_L_4,
}