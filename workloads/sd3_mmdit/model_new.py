import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

import numpy as np
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


# -----------------------------
# Triton fused attention (fwd)
# Based on `ref_kernels/triton_tutorials/06-fused-attention.py` structure,
# adapted to:
# - non-causal
# - fp16
# - arbitrary seq length (masked loads/stores)
# -----------------------------
TRITON_AVAILABLE = False
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


_BLOCK_M: int = 128
_BLOCK_N: int = 64
_NUM_WARPS: int = 4
_NUM_STAGES: int = 2


if TRITON_AVAILABLE:

    @triton.jit
    def _attn_fwd_noncausal_fp16(
        sm_scale,
        Q,
        K,
        V,
        O,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        tl.static_assert(BLOCK_N <= HEAD_DIM)

        pid_m = tl.program_id(0)
        pid_hz = tl.program_id(1)  # flattened (batch, head)

        base = pid_hz * (N_CTX * HEAD_DIM)

        start_m = pid_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)
        mask_m = offs_m < N_CTX

        q_ptrs = Q + base + offs_m[:, None] * HEAD_DIM + offs_k[None, :]
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float16)

        qk_scale = sm_scale * 1.44269504  # 1 / log(2)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for start_n in range(0, N_CTX, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_CTX

            k_ptrs = K + base + offs_n[None, :] * HEAD_DIM + offs_k[:, None]
            k = tl.load(k_ptrs, mask=mask_n[None, :] & (offs_k[:, None] < HEAD_DIM), other=0.0).to(tl.float16)
            qk = tl.dot(q, k).to(tl.float32)  # [BM, BN]
            qk = tl.where(mask_n[None, :], qk, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_ptrs = V + base + offs_n[:, None] * HEAD_DIM + offs_k[None, :]
            v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < HEAD_DIM), other=0.0).to(tl.float16)
            acc += tl.dot(p.to(tl.float16), v).to(tl.float32)

            m_i = m_ij

        l_i = tl.where(mask_m, l_i, 1.0)
        acc = acc / l_i[:, None]

        o_ptrs = O + base + offs_m[:, None] * HEAD_DIM + offs_k[None, :]
        tl.store(o_ptrs, acc.to(tl.float16), mask=mask_m[:, None])


def _triton_attention_noncausal_fp16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    # q,k,v: (B, H, N, D) contiguous fp16 cuda.
    o = torch.empty_like(q)
    b, h, n_ctx, d = q.shape
    grid = (triton.cdiv(n_ctx, _BLOCK_M), b * h)
    _attn_fwd_noncausal_fp16[grid](
        sm_scale,
        q,
        k,
        v,
        o,
        N_CTX=n_ctx,
        HEAD_DIM=d,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return o


def attention(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2).contiguous(), (q, k, v))

    use_triton = (
        TRITON_AVAILABLE
        and mask is None
        and q.is_cuda
        and q.dtype == torch.float16
        and k.dtype == torch.float16
        and v.dtype == torch.float16
    )

    if use_triton:
        out = _triton_attention_noncausal_fp16(q, k, v, sm_scale=1.0 / math.sqrt(dim_head))
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=torch.float16, device=None):
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

    def __init__(self, img_size: Optional[int]=224, patch_size: int=16, in_chans: int=3, embed_dim: int=768, flatten: bool=True, bias: bool=True, strict_img_size: bool=True, dynamic_img_pad: bool=False, dtype=torch.float16, device=None):
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

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.float16, device=None):
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

    def __init__(self, input_dim: int, hidden_size: int, dtype=torch.float16, device=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return (qkv[0], qkv[1], qkv[2])


def optimized_attention(qkv, num_heads):
    return attention(qkv[0], qkv[1], qkv[2], num_heads)


class SelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, qkv_bias: bool=False, qk_scale: Optional[float]=None, pre_only: bool=False, dtype=torch.float16, device=None):
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

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=False, pre_only: bool=False, dtype=torch.float16, device=None, **block_kwargs):
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

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, total_out_channels: Optional[int]=None, dtype=torch.float16, device=None):
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

    def __init__(self, input_size: int=1024, patch_size: int=16, in_channels: int=4, depth: int=28, mlp_ratio: float=4.0, learn_sigma: bool=False, adm_in_channels: Optional[int]=None, context_embedder_config: Optional[Dict]=None, register_length: int=0, out_channels: Optional[int]=None, pos_embed_scaling_factor: Optional[float]=None, pos_embed_offset: Optional[float]=None, pos_embed_max_size: Optional[int]=64, num_patches=4096, qkv_bias: bool=True, dtype=torch.float16, device=None):
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
    torch.set_default_device("mlu" if torch.mlu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Model()
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    print(output.shape)
    print(output.sum())

