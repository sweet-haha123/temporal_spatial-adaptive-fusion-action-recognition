

import math
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# from timm.models.layers import drop_path, to_2tuple
from timm.models.layers import drop_path

class DropPath(nn.Module):
    """ Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x
class KTPBlock(nn.Module):
    """ Transformer Block with Keyframe-centric Token Pruning.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_checkpoint=False, keep_rate=0., enhanced_weight=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = KTPAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_checkpoint=use_checkpoint, keep_rate=keep_rate, enhanced_weight=enhanced_weight)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_checkpoint = use_checkpoint
        self.keep_rate = keep_rate

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward_part1(self, x, last_idx, window_size):
        if self.gamma_1 is None:
            attn, idx = self.attn(self.norm1(x), last_idx, window_size)
            return self.drop_path(attn), idx
        else:
            attn, idx = self.attn(self.norm1(x), last_idx, window_size)
            return self.drop_path(self.gamma_1 * attn), idx

    def forward_part2(self, x):
        if self.gamma_2 is None:
            return self.drop_path(self.mlp(self.norm2(x)))
        else:
            return self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

    def forward(self, x, last_idx, window_size):
        # attn
        tmp, idx = self.forward_part1(x, last_idx, window_size)
        x = x + tmp

        # keyframe-centric token pruning
        B, N, C = x.shape
        if self.keep_rate > 1:
            num_s_tokens = window_size[1] * window_size[2]
            x_key = x[:, :num_s_tokens]  # keyframe tokens on the first column
            x_nonkey = x[:, num_s_tokens:]

            index = idx.unsqueeze(-1).expand(-1, -1, C)  # (B, N_keep, C_e)
            x_nonkey = torch.gather(x_nonkey, dim=1, index=index)  # (B, N_keep, C_e)

            # update the global index in video sequence
            idx = torch.gather(last_idx, dim=1, index=idx)

            x = torch.cat([x_key, x_nonkey], dim=1)

        # mlp
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x, idx


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class KTPAttention(Attention):
    """ Attention with Keyframe-centric Token Pruning (KTP).
    """

    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, use_checkpoint=False, keep_rate=0., enhanced_weight=1):
        super(KTPAttention, self).__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                           proj_drop, attn_head_dim)
        self.keep_rate = keep_rate
        self.enhanced_weight = enhanced_weight
        self.use_checkpoint = use_checkpoint

    def forward_part1(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.keep_rate > 1:
            return x, attn
        return x

    def forward(self, x, last_idx=None, ws=None):
        B, N, C = x.shape
        if self.keep_rate > 1:
            if self.use_checkpoint:
                x, attn = checkpoint.checkpoint(self.forward_part1, x)
            else:
                x, attn = self.forward_part1(x)
        else:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.forward_part1, x)
            else:
                x = self.forward_part1(x)

        # keep top-k tokens and the corresponding indexes
        if self.keep_rate > 1:
            num_keep_tokens = math.ceil(self.keep_rate*8)
            if num_keep_tokens == N:
                return x, last_idx

            num_s_tokens = ws[1] * ws[2]
            num_keep_tokens -= num_s_tokens

            nonkey_attn = attn[..., num_s_tokens:]  # (B, H, N, N - key)
            # keyframe query enhancement
            nonkey_attn[:, :, :num_s_tokens] *= self.enhanced_weight
            # average on all queries and num_heads
            nonkey_attn = nonkey_attn.mean(dim=2).mean(dim=1)  # (B, N - N_key)

            _, idx = torch.topk(nonkey_attn, num_keep_tokens, dim=1, largest=True)  # (B, N_keep)
            return x, idx

        return x, last_idx