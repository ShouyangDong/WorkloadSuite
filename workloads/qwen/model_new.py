import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


import triton
import triton.language as tl
@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptr,
    v_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    stride_s: tl.constexpr,
    stride_d: tl.constexpr,
):
    # FlashAttention-v2-style staged iteration.
    # STAGE == 1: off-band (0 .. start_m*BLOCK_M)
    # STAGE == 2: on-band (diagonal block)
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    else:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)

    offs_d = tl.arange(0, HEAD_DIM)

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ids = start_n + offs_n

        k_ptrs = k_ptr + k_ids[None, :] * stride_s + offs_d[:, None] * stride_d
        v_ptrs = v_ptr + k_ids[:, None] * stride_s + offs_d[None, :] * stride_d

        # K: [D, N] to feed dot(q, k)
        k = tl.load(
            k_ptrs,
            mask=(k_ids[None, :] < N_CTX) & (offs_d[:, None] < HEAD_DIM),
            other=0.0,
        )
        qk = tl.dot(q, k)

        if STAGE == 2:
            # Autoregressive mask within the diagonal block.
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            qk = qk - m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V: [N, D]
        v = tl.load(
            v_ptrs,
            mask=(k_ids[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        acc = tl.dot(p.to(tl.float16), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sm_scale,
    stride_z: tl.constexpr,
    stride_h: tl.constexpr,
    stride_s: tl.constexpr,
    stride_d: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ids
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    # base pointers for this (z, h)
    q_ptr = q_ptr + off_z * stride_z + off_h * stride_h
    k_ptr = k_ptr + off_z * stride_z + off_h * stride_h
    v_ptr = v_ptr + off_z * stride_z + off_h * stride_h
    o_ptr = o_ptr + off_z * stride_z + off_h * stride_h

    # offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # q: [M, D]
    q_ptrs = q_ptr + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM), other=0.0)

    # log2(e) scaled softmax
    qk_scale = sm_scale * 1.4426950408889634

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # causal fwd: stage 1 (off-band), stage 2 (on-band/diagonal)
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_ptr,
        v_ptr,
        start_m=start_m,
        qk_scale=qk_scale,
        BLOCK_M=BLOCK_M,
        HEAD_DIM=HEAD_DIM,
        BLOCK_N=BLOCK_N,
        STAGE=1,
        offs_m=offs_m,
        offs_n=offs_n,
        N_CTX=N_CTX,
        stride_s=stride_s,
        stride_d=stride_d,
    )
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_ptr,
        v_ptr,
        start_m=start_m,
        qk_scale=qk_scale,
        BLOCK_M=BLOCK_M,
        HEAD_DIM=HEAD_DIM,
        BLOCK_N=BLOCK_N,
        STAGE=2,
        offs_m=offs_m,
        offs_n=offs_n,
        N_CTX=N_CTX,
        stride_s=stride_s,
        stride_d=stride_d,
    )

    # write o
    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d
    tl.store(o_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


def _flash_attn_triton_bhsd(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """
    Triton fused attention for q,k,v in shape (B, H, S, D).
    Returns tensor of same shape.
    """
    assert causal, "Only causal=True is supported."
    assert q_bhsd.dtype == torch.float16 and k_bhsd.dtype == torch.float16 and v_bhsd.dtype == torch.float16
    assert q_bhsd.shape == k_bhsd.shape == v_bhsd.shape
    b, h, s, d = q_bhsd.shape

    # Ensure contiguous (the model builds q/k/v via transposes).
    q = q_bhsd.contiguous()
    k = k_bhsd.contiguous()
    v = v_bhsd.contiguous()
    o = torch.empty_like(q)

    grid = (triton.cdiv(s, 128), b * h)
    _attn_fwd[grid](
        q,
        k,
        v,
        o,
        sm_scale,
        stride_z=q.stride(0),
        stride_h=q.stride(1),
        stride_s=q.stride(2),
        stride_d=q.stride(3),
        Z=b,
        H=h,
        N_CTX=s,
        HEAD_DIM=d,
    )
    return o

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        :param dim: Hidden dimension size
        :param eps: Epsilon for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_states: Input tensor
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        """
        :param hidden_size: Embedding dimension
        :param intermediate_size: MLP intermediate dimension
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float16)

    def forward(self, x):
        """
        SwiGLU activation: (SiLU(Gate) * Up) -> Down
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies Rotary Position Embedding to query and key tensors.
    """
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True, dtype=torch.float16)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True, dtype=torch.float16)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True, dtype=torch.float16)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False, dtype=torch.float16)

        # Qwen3 specific: RMSNorm applied to Q and K before RoPE
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.shape

        # Projection
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Q/K Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        # Slice cos/sin to current sequence length
        cos = cos[:, :, :seq_len, :]
        sin = sin[:, :, :seq_len, :]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV for GQA (Grouped Query Attention)
        n_rep = self.num_heads // self.num_kv_heads
        k = (
            k[:, :, None, :, :]
            .expand(batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim)
            .reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        )
        v = (
            v[:, :, None, :, :]
            .expand(batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim)
            .reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        )

        output = _flash_attn_triton_bhsd(q, k, v, causal=True, sm_scale=float(self.scale))
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Qwen3Attention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)

    def forward(self, x, cos, sin):
        """
        Block structure:
        x = x + Attn(LN(x))
        x = x + MLP(LN(x))
        """
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Model(nn.Module):
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        num_layers=16,
        num_heads=16,
        num_kv_heads=16,
        intermediate_size=5504,
        max_position_embeddings=40960,
        rope_theta=1000000.0,
    ):
        """
        Implementation of Qwen3ForCausalLM.
        Mask creation is performed inside each attention call (not shared across layers).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.float16)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(hidden_size, intermediate_size, num_heads, num_kv_heads) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.float16)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE buffers
        head_dim = hidden_size // num_heads
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Register as buffers (persistent=False means they are not saved in state_dict usually, but here we keep them accessible)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, input_ids):
        """
        :param input_ids: Input integer tensor, shape (batch_size, seq_len)
        :return: Logits tensor, shape (batch_size, seq_len, vocab_size)
        """
        x = self.embed_tokens(input_ids)

        # Retrieve RoPE caches
        cos = self.cos.to(torch.float16)
        sin = self.sin.to(torch.float16)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


vocab_size = 151936
hidden_size = 4096
num_layers = 36
num_heads = 32
num_kv_heads = 8
batch_size = 1
max_seq_len = 4096
intermediate_size = 12288


def get_inputs():
    # Returns a list containing a random integer tensor of shape (batch_size, seq_len)
    return [torch.randint(0, vocab_size, (batch_size, max_seq_len))]


def get_init_inputs():
    # Returns arguments to initialize the model:
    # [vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size]
    return [vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size]


if __name__ == "__main__":
    torch.set_default_device("cuda")
    model = Model()
    model.eval()
    inputs = get_inputs()
    out = model(*inputs)
    print("max:", out.max().item(), "min:", out.min().item())
    print(out.shape, out.dtype)

