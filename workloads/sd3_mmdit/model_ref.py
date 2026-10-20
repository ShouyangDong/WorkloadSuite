import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import numpy as np
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """
    Minimal attention (no mask / no causal).

    Expected shapes:
      - q, k: (batch, seq_len, heads * head_dim)
      - v:    (batch, seq_len, heads * head_dim) or (batch, seq_len, heads, head_dim)
    """
    b, q_len, q_dim = q.shape
    dim_head = q_dim // heads

    q = q.view(b, q_len, heads, dim_head).transpose(1, 2)  # (B, H, Q, D)
    k = k.view(b, k.shape[1], heads, dim_head).transpose(1, 2)  # (B, H, K, D)
    v = v.view(b, v.shape[1], heads, dim_head).transpose(1, 2)  # (B, H, K, D)

    scale = dim_head ** -0.5
    attn = torch.matmul(q, k.transpose(2, 3)) * scale
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2).reshape(b, q_len, heads * dim_head)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=torch.float16, device='cuda'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PatchEmbed(nn.Module):

    def __init__(self, img_size: Optional[int]=224, patch_size: int=16, in_chans: int=3, embed_dim: int=768, flatten: bool=True, bias: bool=True, strict_img_size: bool=True, dynamic_img_pad: bool=False, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t, dtype, **kwargs):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class VectorEmbedder(nn.Module):

    def __init__(self, input_dim: int, hidden_size: int, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return (qkv[0], qkv[1], qkv[2])

class SelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, qkv_bias: bool=False, qk_scale: Optional[float]=None, pre_only: bool=False, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if not pre_only:
            self.proj = nn.Linear(dim, dim, dtype=dtype, device=device)
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor):
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = q.reshape(q.shape[0], q.shape[1], -1)
        k = k.reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.pre_attention(x)
        x = attention(q, k, v, self.num_heads)
        x = self.post_attention(x)
        return x

class DismantledBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=False, pre_only: bool=False, dtype=torch.float16, device='cuda', **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, pre_only=pre_only, dtype=dtype, device=device)
        if not pre_only:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU(approximate='tanh'), dtype=dtype, device=device)
        n_mods = 6 if not pre_only else 2
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
        if not self.pre_only:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return (qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp))
        else:
            shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return (qkv, None)

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (q, k, v), intermediates = self.pre_attention(x, c)
        attn = attention(q, k, v, self.attn.num_heads)
        return self.post_attention(attn, *intermediates)

def block_mixing(context, x, context_block, x_block, c):
    context_qkv, context_intermediates = context_block.pre_attention(context, c)
    x_qkv, x_intermediates = x_block.pre_attention(x, c)
    o = []
    for t in range(3):
        o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
    q, k, v = tuple(o)
    attn = attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = (attn[:, :context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1]:])
    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None
    x = x_block.post_attention(x_attn, *x_intermediates)
    return (context, x)

class JointBlock(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop('pre_only')
        self.context_block = DismantledBlock(*args, pre_only=pre_only, **kwargs)
        self.x_block = DismantledBlock(*args, pre_only=False, **kwargs)

    def forward(self, *args, **kwargs):
        return block_mixing(*args, context_block=self.context_block, x_block=self.x_block, **kwargs)

class FinalLayer(nn.Module):

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, total_out_channels: Optional[int]=None, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device) if total_out_channels is None else nn.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Model(nn.Module):

    def __init__(self, input_size: int=1024, patch_size: int=16, in_channels: int=4, depth: int=28, mlp_ratio: float=4.0, learn_sigma: bool=False, adm_in_channels: Optional[int]=None, context_embedder_config: Optional[Dict]=None, register_length: int=0, out_channels: Optional[int]=None, pos_embed_scaling_factor: Optional[float]=None, pos_embed_offset: Optional[float]=None, pos_embed_max_size: Optional[int]=64, num_patches=4096, qkv_bias: bool=True, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size
        hidden_size = 64 * depth
        num_heads = depth
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=self.pos_embed_max_size is None, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device)
        self.register_length = register_length
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size, dtype=dtype, device=device))
        self.joint_blocks = nn.ModuleList([JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, pre_only=i == depth - 1, dtype=dtype, device=device) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=dtype, device=device)

    def cropped_pos_embed(self, hw):
        p = self.x_embedder.patch_size[0]
        h, w = hw
        h = h // p
        w = w // p
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(self.pos_embed, '1 (h w) c -> 1 h w c', h=self.pos_embed_max_size, w=self.pos_embed_max_size)
        spatial_pos_embed = spatial_pos_embed[:, top:top + h, left:left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, '1 h w c -> 1 (h w) c')
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h, w = hw
        h = h // p
        w = w // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(self, x: torch.Tensor, c_mod: torch.Tensor, context: Optional[torch.Tensor]=None) -> torch.Tensor:
        for block in self.joint_blocks:
            context, x = block(context, x, c=c_mod)
        x = self.final_layer(x, c_mod)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]=None, context: Optional[torch.Tensor]=None) -> torch.Tensor:
        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(hw)
        c = self.t_embedder(t, dtype=torch.float16)
        x = self.forward_core_with_concat(x, c, context)
        x = self.unpatchify(x, hw=hw)
        return x


batch_size = 4
C = 4
H = 1024
W = 1024

def get_inputs():
    x = torch.randn(batch_size, C, H, W, dtype=torch.float16)
    t = torch.full((batch_size,), 3.0, dtype=torch.float32)
    # Use a token sequence shape (B, T, hidden) to match attention blocks.
    context = torch.randn(batch_size, 1, 64 * 28, dtype=torch.float16)
    return [x, t, None, context]

def get_init_inputs():
    return []
if __name__ == '__main__':
    torch.set_default_device('cuda')
    model = Model()
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    print(output.shape)
    print(output.sum())

'''
python run/ab.py --ref run/sd3_mmdit/model_naive.py --test run/sd3_mmdit/model.py --bench-runs 20 --compile-run-times 1
'''