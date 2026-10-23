import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, cast
from dataclasses import dataclass

import numpy as np
import random

import triton
import triton.language as tl

def _num_sms() -> int:
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).multi_processor_count)
        if torch.mlu.is_available():
            return int(torch.mlu.get_device_properties(0).multi_processor_count)
    except Exception:
        pass
    return 84


_NUM_SMS = _num_sms()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# =============================================================================
# DeepSeek-V3 MoE Components
# Ref: DeepSeek-V3 Technical Report (arXiv:2412.19437)
# =============================================================================


@dataclass
class MoEConfig:
    """Configuration for DeepSeek-V3 MoE layer (Table 5 of mHC paper)."""

    dim: int = 1280  # Hidden dimension
    moe_inter_dim: int = 896  # Expert intermediate dimension
    n_routed_experts: int = 64  # Total routed experts
    n_shared_experts: int = 2  # Shared experts (always active)
    n_activated_experts: int = 6  # Top-K active experts
    route_scale: float = 1.0  # Routing weight scale


class Gate(nn.Module):
    """
    Gating mechanism for MoE routing.
    Ref: DeepSeek-V3 Section 3.1, Loss-Free Load Balancing
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.dim = config.dim
        self.topk = config.n_activated_experts
        self.route_scale = config.route_scale

        # Router weights
        self.weight = nn.Parameter(torch.randn(config.n_routed_experts, config.dim, dtype=torch.float16) * config.dim ** -0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, D] input tensor
        Returns:
            weights: [B, K] routing weights for top-K experts
            indices: [B, K] indices of selected experts
        """
        # Compute routing scores
        scores = F.linear(x, self.weight)  # [B, N]
        scores = scores.sigmoid()
        original_scores = scores

        # Top-K selection
        indices = torch.topk(scores, self.topk, dim=-1)[1]  # [B, K]
        weights = original_scores.gather(1, indices)  # [B, K]

        # Normalize weights for sigmoid scoring
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale

        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Single SwiGLU Expert.
    Ref: DeepSeek-V2 Eq.(5)
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.float16)  # gate
        self.w2 = nn.Linear(inter_dim, dim, bias=False, dtype=torch.float16)  # down
        self.w3 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.float16)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    """
    Standard SwiGLU MLP (used for shared experts).
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.float16)
        self.w2 = nn.Linear(inter_dim, dim, bias=False, dtype=torch.float16)
        self.w3 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@triton.autotune(
    configs=[
        # Balanced configs for medium M, fixed (N,K) like (896,1280) / (1280,896).
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SM": _NUM_SMS}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SM": _NUM_SMS}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SM": _NUM_SMS}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "NUM_SM": _NUM_SMS}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=4),
        # Larger K tiles
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "NUM_SM": _NUM_SMS}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "NUM_SM": _NUM_SMS}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=3),
        # Very large K tiles (can help when K is 1280)
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128, "NUM_SM": _NUM_SMS}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "NUM_SM": _NUM_SMS}, num_warps=8, num_stages=3),
    ],
    key=["group_size"],
)
@triton.jit
def _grouped_matmul_a_bt_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. shape: [group_size * 3] flattened triples [M, N, K]
    group_gemm_sizes,
    # device tensor of leading dimension sizes. shape: [group_size * 3] flattened triples [lda, ldb, ldc]
    group_lds,
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute grouped GEMM:
      C[M, N] = A[M, K] @ B[N, K]^T
    Where B is stored row-major as [N, K] (contiguous in K), and we access it as transposed.
    """
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3 + 0).to(tl.int32)
        gn = tl.load(group_gemm_sizes + g * 3 + 1).to(tl.int32)
        gk = tl.load(group_gemm_sizes + g * 3 + 2).to(tl.int32)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        while (tile_idx >= last_problem_end) & (tile_idx < last_problem_end + num_tiles):
            lda = tl.load(group_lds + g * 3 + 0).to(tl.int32)
            ldb = tl.load(group_lds + g * 3 + 1).to(tl.int32)
            ldc = tl.load(group_lds + g * 3 + 2).to(tl.int32)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_am = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            # A block pointers: [BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            # B is [N, K], but we want [K, N] block for tl.dot.
            # B^T[k, n] = B[n, k] => address = b_ptr + n * ldb + k
            b_ptrs = b_ptr + offs_k[:, None] + offs_cn[None, :] * ldb

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                k_offs = kk * BLOCK_K + offs_k
                a = tl.load(a_ptrs, mask=(offs_am[:, None] < gm) & (k_offs[None, :] < gk), other=0.0)
                b = tl.load(b_ptrs, mask=(k_offs[:, None] < gk) & (offs_cn[None, :] < gn), other=0.0)
                acc += tl.dot(a, b)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K

            c = acc.to(tl.float16)
            c_ptrs = c_ptr + offs_am[:, None] * ldc + offs_cn[None, :]
            tl.store(c_ptrs, c, mask=(offs_am[:, None] < gm) & (offs_cn[None, :] < gn))

            tile_idx += NUM_SM

        last_problem_end += num_tiles


@triton.jit
def _silu_mul_inplace_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu = x * tl.sigmoid(x)
    out = silu * y
    tl.store(x_ptr + offs, out.to(tl.float16), mask=mask)


@triton.jit
def _bucket_scatter_by_expert_kernel(
    x_in_ptr,
    expert_in_ptr,
    weight_in_ptr,
    offsets_ptr,  # int32[E], initialized to starts
    x_out_ptr,
    pair_id_out_ptr,
    weight_out_ptr,
    NPAIRS,
    D: tl.constexpr,
    KSEL: tl.constexpr,
    NBLOCKS: tl.constexpr,
    STRIDE_XIN0: tl.constexpr,
    STRIDE_XIN1: tl.constexpr,
    STRIDE_XOUT0: tl.constexpr,
    STRIDE_XOUT1: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= NPAIRS:
        return

    token = pid // KSEL
    expert = tl.load(expert_in_ptr + pid).to(tl.int32)
    w = tl.load(weight_in_ptr + pid).to(tl.float16)

    pos = tl.atomic_add(offsets_ptr + expert, 1)
    tl.store(pair_id_out_ptr + pos, pid)
    tl.store(weight_out_ptr + pos, w)

    # Copy x row -> x_out[pos]
    x_row_base = x_in_ptr + token * STRIDE_XIN0
    out_row_base = x_out_ptr + pos * STRIDE_XOUT0
    # Iterate blocks of the feature dimension.
    for blk in tl.static_range(0, NBLOCKS):  # pyright: ignore[reportGeneralTypeIssues]
        offs = blk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D
        x_vals = tl.load(x_row_base + offs * STRIDE_XIN1, mask=mask, other=0.0)
        tl.store(out_row_base + offs * STRIDE_XOUT1, x_vals, mask=mask)


@triton.jit
def _scatter_add_weighted_kernel(
    o_ptr,
    token_ptr,
    weight_ptr,
    y_ptr,
    n_rows,
    n_cols,
    STRIDE_O0: tl.constexpr,
    STRIDE_O1: tl.constexpr,
    STRIDE_Y0: tl.constexpr,
    STRIDE_Y1: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nb = tl.program_id(1)
    offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < n_cols

    token = tl.load(token_ptr + pid_m).to(tl.int32)
    w = tl.load(weight_ptr + pid_m).to(tl.float32)

    o_ptrs = o_ptr + pid_m * STRIDE_O0 + offs_n * STRIDE_O1
    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    upd = (o * w).to(tl.float16)

    y_ptrs = y_ptr + token * STRIDE_Y0 + offs_n * STRIDE_Y1
    tl.atomic_add(y_ptrs, upd, mask=mask)


@triton.jit
def _write_pairs_weighted_kernel(
    o_ptr,
    pair_id_ptr,
    weight_ptr,
    o_pairs_ptr,
    n_rows,
    n_cols,
    STRIDE_O0: tl.constexpr,
    STRIDE_O1: tl.constexpr,
    STRIDE_P0: tl.constexpr,
    STRIDE_P1: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nb = tl.program_id(1)
    offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < n_cols

    dst_row = tl.load(pair_id_ptr + pid_m).to(tl.int32)
    w = tl.load(weight_ptr + pid_m).to(tl.float32)

    o_ptrs = o_ptr + pid_m * STRIDE_O0 + offs_n * STRIDE_O1
    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    out = (o * w).to(tl.float16)

    p_ptrs = o_pairs_ptr + dst_row * STRIDE_P0 + offs_n * STRIDE_P1
    tl.store(p_ptrs, out, mask=mask)


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-V3 Mixture-of-Experts Layer.
    Ref: DeepSeek-V3 Technical Report, DeepSeek-V2 Section 2.1.2

    Architecture:
    - Shared experts: Always active, outputs summed
    - Routed experts: Top-K selected via gating
    - Output = shared_output + routed_output
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.moe_inter_dim = config.moe_inter_dim

        # Gating mechanism
        self.gate = Gate(config)

        # Routed experts
        self.experts = nn.ModuleList([Expert(config.dim, config.moe_inter_dim) for _ in range(config.n_routed_experts)])

        # Shared experts (fused as single larger MLP)
        # Hidden dim = n_shared_experts * moe_inter_dim
        self.shared_experts = MLP(config.dim, config.n_shared_experts * config.moe_inter_dim)

        # Lazy-initialized device-side pointer caches for expert weights.
        self._ptr_device: Optional[torch.device] = None
        self._w1_ptrs: Optional[torch.Tensor] = None
        self._w2_ptrs: Optional[torch.Tensor] = None
        self._w3_ptrs: Optional[torch.Tensor] = None
        self._w1_ldb: int = self.dim
        self._w2_ldb: int = self.moe_inter_dim
        self._w3_ldb: int = self.dim

        # Eagerly initialize weight pointer caches if parameters are already on CUDA.
        # This avoids graph breaks under torch.compile in typical runs where default device is CUDA.
        first = cast(Expert, self.experts[0])
        if first.w1.weight.device.type in ("cuda", "mlu"):
            device: torch.device = first.w1.weight.device
            w1_addrs = [cast(Expert, self.experts[i]).w1.weight.data_ptr() for i in range(self.n_routed_experts)]
            w2_addrs = [cast(Expert, self.experts[i]).w2.weight.data_ptr() for i in range(self.n_routed_experts)]
            w3_addrs = [cast(Expert, self.experts[i]).w3.weight.data_ptr() for i in range(self.n_routed_experts)]
            self._w1_ptrs = torch.tensor(w1_addrs, device=device, dtype=torch.int64)
            self._w2_ptrs = torch.tensor(w2_addrs, device=device, dtype=torch.int64)
            self._w3_ptrs = torch.tensor(w3_addrs, device=device, dtype=torch.int64)
            self._ptr_device = device

    def _ensure_weight_ptrs(self, device: torch.device) -> None:
        if self._ptr_device == device and self._w1_ptrs is not None:
            return
        w1_addrs = [cast(Expert, self.experts[i]).w1.weight.data_ptr() for i in range(self.n_routed_experts)]
        w2_addrs = [cast(Expert, self.experts[i]).w2.weight.data_ptr() for i in range(self.n_routed_experts)]
        w3_addrs = [cast(Expert, self.experts[i]).w3.weight.data_ptr() for i in range(self.n_routed_experts)]
        self._w1_ptrs = torch.tensor(w1_addrs, device=device, dtype=torch.int64)
        self._w2_ptrs = torch.tensor(w2_addrs, device=device, dtype=torch.int64)
        self._w3_ptrs = torch.tensor(w3_addrs, device=device, dtype=torch.int64)
        self._ptr_device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] or [B, S, D] input tensor
        Returns:
            output: Same shape as input
        """
        shape = x.size()
        x = x.view(-1, self.dim)  # [T, D]
        T = x.shape[0]

        # Routing
        weights, indices = self.gate(x)  # [T, K], [T, K]

        # Shared experts (always active)
        z = self.shared_experts(x)

        # Routed experts (Triton grouped GEMM)
        device = x.device
        if device.type not in ("cuda", "mlu"):
            # Fallback: correctness-first on CPU (routed Triton grouped GEMM is CUDA/MLU-only).
            y = torch.zeros_like(x)
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
            for i in range(self.n_routed_experts):
                if counts[i] == 0:
                    continue
                expert = self.experts[i]
                idx, top = torch.where(indices == i)
                y[idx] += expert(x[idx]) * weights[idx, top, None]
            return (y + z).view(shape)

        self._ensure_weight_ptrs(device)
        assert self._w1_ptrs is not None
        assert self._w2_ptrs is not None
        assert self._w3_ptrs is not None

        Ksel = self.n_activated_experts
        Npairs = T * Ksel

        # Flatten routing results.
        flat_expert = indices.reshape(-1).to(torch.int32)  # [Npairs]
        flat_weight = weights.reshape(-1).to(torch.float16)  # [Npairs]

        # Bucket by expert (counting-sort style):
        # - counts/starts define per-expert contiguous ranges
        # - a single Triton kernel scatters (x, token, weight) into those ranges using atomic offsets
        E = self.n_routed_experts
        counts = torch.bincount(flat_expert, minlength=E).to(torch.int32)  # [E]
        starts = torch.cumsum(counts, dim=0) - counts  # [E]
        offsets = starts.clone()  # [E] mutated by atomic_add in the scatter kernel

        x_sorted = torch.empty((Npairs, self.dim), device=device, dtype=torch.float16)
        pair_id_sorted = torch.empty((Npairs,), device=device, dtype=torch.int32)
        weight_sorted = torch.empty((Npairs,), device=device, dtype=torch.float16)

        BLOCK_D = 256
        NBLOCKS = (self.dim + BLOCK_D - 1) // BLOCK_D
        grid_bucket = (Npairs,)
        _bucket_scatter_by_expert_kernel[grid_bucket](
            x,
            flat_expert,
            flat_weight,
            offsets,
            x_sorted,
            pair_id_sorted,
            weight_sorted,
            Npairs,
            D=self.dim,  # pyright: ignore[reportArgumentType]
            KSEL=Ksel,  # pyright: ignore[reportArgumentType]
            NBLOCKS=NBLOCKS,  # pyright: ignore[reportArgumentType]
            STRIDE_XIN0=x.stride(0),  # pyright: ignore[reportArgumentType]
            STRIDE_XIN1=x.stride(1),  # pyright: ignore[reportArgumentType]
            STRIDE_XOUT0=x_sorted.stride(0),  # pyright: ignore[reportArgumentType]
            STRIDE_XOUT1=x_sorted.stride(1),  # pyright: ignore[reportArgumentType]
            BLOCK_D=BLOCK_D,  # pyright: ignore[reportArgumentType]
        )

        # Build A pointers (x_sorted slices) and per-group sizes/strides without Python loops.
        a_base = x_sorted.data_ptr()
        a_stride0 = int(x_sorted.stride(0))
        a_ptrs = (starts.to(torch.int64) * a_stride0 * x_sorted.element_size()) + int(a_base)

        # g1/g3 outputs are contiguous [Npairs, inter], and each group writes to [starts[e]:starts[e]+counts[e]].
        g1_sorted = torch.empty((Npairs, self.moe_inter_dim), device=device, dtype=torch.float16)
        g3_sorted = torch.empty((Npairs, self.moe_inter_dim), device=device, dtype=torch.float16)
        c1_base = g1_sorted.data_ptr()
        c3_base = g3_sorted.data_ptr()
        c1_stride0 = int(g1_sorted.stride(0))
        c3_stride0 = int(g3_sorted.stride(0))
        c1_ptrs = (starts.to(torch.int64) * c1_stride0 * g1_sorted.element_size()) + int(c1_base)
        c3_ptrs = (starts.to(torch.int64) * c3_stride0 * g3_sorted.element_size()) + int(c3_base)

        # Per-group GEMM metadata.
        # For this MoE, N and K are fixed for all groups; only M varies (counts[e]).
        gm = counts
        gn_w1 = torch.full((E,), self.moe_inter_dim, device=device, dtype=torch.int32)
        gk_w1 = torch.full((E,), self.dim, device=device, dtype=torch.int32)
        g_sizes_w1 = torch.stack([gm, gn_w1, gk_w1], dim=1).reshape(-1).contiguous()

        lda = torch.full((E,), a_stride0, device=device, dtype=torch.int32)
        ldb_w1 = torch.full((E,), int(self._w1_ldb), device=device, dtype=torch.int32)
        ldc_w1 = torch.full((E,), int(g1_sorted.stride(0)), device=device, dtype=torch.int32)
        g_lds_w1 = torch.stack([lda, ldb_w1, ldc_w1], dim=1).reshape(-1).contiguous()

        # Launch grouped GEMMs writing directly to contiguous buffers (no per-expert tensors, no copy).
        grid = lambda META: (META["NUM_SM"],)
        _grouped_matmul_a_bt_kernel[grid](
            a_ptrs,
            self._w1_ptrs,
            c1_ptrs,
            g_sizes_w1,
            g_lds_w1,
            E,
        )
        _grouped_matmul_a_bt_kernel[grid](
            a_ptrs,
            self._w3_ptrs,
            c3_ptrs,
            g_sizes_w1,
            g_lds_w1,
            E,
        )

        # In-place: g1_sorted = silu(g1_sorted) * g3_sorted
        n_elem = Npairs * self.moe_inter_dim
        grid_elt = lambda META: (triton.cdiv(n_elem, META["BLOCK"]),)
        _silu_mul_inplace_kernel[grid_elt](g1_sorted, g3_sorted, n_elem, BLOCK=1024)  # pyright: ignore[reportArgumentType]

        # o_sorted = h @ w2^T  (w2 stored as [dim, inter] => B is [N=dim, K=inter])
        o_sorted = torch.empty((Npairs, self.dim), device=device, dtype=torch.float16)
        o_base = o_sorted.data_ptr()
        o_stride0 = int(o_sorted.stride(0))
        o_ptrs = (starts.to(torch.int64) * o_stride0 * o_sorted.element_size()) + int(o_base)

        gn_w2 = torch.full((E,), self.dim, device=device, dtype=torch.int32)
        gk_w2 = torch.full((E,), self.moe_inter_dim, device=device, dtype=torch.int32)
        g_sizes_w2 = torch.stack([gm, gn_w2, gk_w2], dim=1).reshape(-1).contiguous()

        lda2 = torch.full((E,), int(g1_sorted.stride(0)), device=device, dtype=torch.int32)
        ldb2 = torch.full((E,), int(self._w2_ldb), device=device, dtype=torch.int32)
        ldc2 = torch.full((E,), int(o_sorted.stride(0)), device=device, dtype=torch.int32)
        g_lds_w2 = torch.stack([lda2, ldb2, ldc2], dim=1).reshape(-1).contiguous()

        _grouped_matmul_a_bt_kernel[grid](
            (starts.to(torch.int64) * int(g1_sorted.stride(0)) * g1_sorted.element_size()) + int(g1_sorted.data_ptr()),
            self._w2_ptrs,
            o_ptrs,
            g_sizes_w2,
            g_lds_w2,
            E,
        )

        # Write back to token-major (pair-major) order (no atomics), then reduce over top-k.
        o_pairs = torch.empty((Npairs, self.dim), device=device, dtype=torch.float16)
        grid_write = (Npairs, triton.cdiv(self.dim, 256))
        _write_pairs_weighted_kernel[grid_write](
            o_sorted,
            pair_id_sorted,
            weight_sorted,
            o_pairs,
            Npairs,
            self.dim,
            STRIDE_O0=o_sorted.stride(0),  # pyright: ignore[reportArgumentType]
            STRIDE_O1=o_sorted.stride(1),  # pyright: ignore[reportArgumentType]
            STRIDE_P0=o_pairs.stride(0),  # pyright: ignore[reportArgumentType]
            STRIDE_P1=o_pairs.stride(1),  # pyright: ignore[reportArgumentType]
            BLOCK_N=256,  # pyright: ignore[reportArgumentType]
        )

        y = o_pairs.view(T, Ksel, self.dim).sum(dim=1)

        return (y + z).view(shape)


# =============================================================================
# mHC Model
# Ref: arXiv:2512.24880 - Manifold-Constrained Hyper-Connections
# =============================================================================


class Model(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) Layer.
    Ref: arXiv:2512.24880

    Key features:
    - Expands residual stream width by factor n
    - Uses doubly stochastic matrices for stable signal propagation
    - Maintains identity mapping property via Sinkhorn-Knopp projection
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,  # n (Table 5)
        sinkhorn_iters: int = 20,  # t_max (Eq.9, Table 5)
        sinkhorn_eps: float = 1e-8,  # Not in paper, engineering choice
        norm_eps: float = 1e-20,  # RMSNorm epsilon (Table 5)
        alpha_init: float = 0.01,  # Gating factor init (Table 5)
        moe_config: Optional[MoEConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps
        self.norm_eps = norm_eps

        n = expansion_rate
        C = hidden_dim

        if moe_config is None:
            moe_config = MoEConfig()
        self.F_layer = DeepSeekMoE(moe_config)

        # RMSNorm for layer input (Table 3)
        self.norm_layer = nn.RMSNorm(C, eps=norm_eps)

        # ===== Eq.(10): φ_l : tfloat32 [nC, n² + 2n] =====
        # Stored as fp32, computed with TF32 on Tensor Cores
        self.phi_pre = nn.Parameter(torch.randn(n * C, n, dtype=torch.float32) * 0.02)
        self.phi_post = nn.Parameter(torch.randn(n * C, n, dtype=torch.float32) * 0.02)
        self.phi_res = nn.Parameter(torch.randn(n * C, n * n, dtype=torch.float32) * 0.02)

        # ===== Eq.(13): b_l : float32 [1, n² + 2n] =====
        self.b_pre = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.b_post = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.b_res = nn.Parameter(torch.zeros(n, n, dtype=torch.float32))

        # ===== Eq.(12): α : float32 scalars =====
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def sinkhorn(self, log_M: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm for doubly stochastic projection.
        Ref: Eq.(9) & Eq.(19)

        M^(0) = exp(H̃^res)
        M^(t) = T_r(T_c(M^(t-1)))
        Returns H^res = M^(t_max) on the Birkhoff polytope.
        """
        # Numerical stability: log-sum-exp trick (not in paper, but necessary)
        log_M = log_M - log_M.amax(dim=(-2, -1), keepdim=True)
        M = torch.exp(log_M)

        for _ in range(self.sinkhorn_iters):
            # T_c: Column normalization (sum over rows)
            M = M / (M.sum(dim=-2, keepdim=True) + self.sinkhorn_eps)
            # T_r: Row normalization (sum over cols)
            M = M / (M.sum(dim=-1, keepdim=True) + self.sinkhorn_eps)

        return M

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mHC layer.

        Args:
            x_l: Input hidden states [Batch, n, C] in float16
        Returns:
            x_{l+1}: Output hidden states [Batch, n, C] in float16
        """
        input_dtype = x_l.dtype  # bf16
        B, n, C = x_l.shape

        # =====================================================================
        # Part 1: Coefficient Generation (Eq.10-19) — All in float32
        # =====================================================================

        # ===== Eq.(11): x̄_l : float16 [1, nC] =====
        x_flat = x_l.reshape(B, n * C)
        x_flat_fp32 = x_flat.float()

        # ===== Eq.(14): [H̃̃^pre, H̃̃^post, H̃̃^res] = x̄_l @ φ_l : float32 =====
        # GEMM with TF32 compute
        tilde_tilde_pre = x_flat_fp32 @ self.phi_pre  # [B, n]
        tilde_tilde_post = x_flat_fp32 @ self.phi_post  # [B, n]
        tilde_tilde_res = x_flat_fp32 @ self.phi_res  # [B, n²]

        # ===== Eq.(15): r = ||x̄_l||_2 / √(nC) : float32 =====
        r = x_flat_fp32.norm(p=2, dim=-1, keepdim=True) / math.sqrt(n * C)

        # ===== Eq.(16): H̃ = (1/r)[α · H̃̃] + b : float32 =====
        tilde_pre = (self.alpha_pre * tilde_tilde_pre) / (r + self.norm_eps) + self.b_pre
        tilde_post = (self.alpha_post * tilde_tilde_post) / (r + self.norm_eps) + self.b_post
        tilde_res = ((self.alpha_res * tilde_tilde_res) / (r + self.norm_eps)).reshape(B, n, n) + self.b_res

        # ===== Eq.(17): H^pre = σ(H̃^pre) : float32 =====
        H_pre = torch.sigmoid(tilde_pre)

        # ===== Eq.(18): H^post = 2σ(H̃^post) : float32 =====
        H_post = 2.0 * torch.sigmoid(tilde_post)

        # ===== Eq.(19): H^res = Sinkhorn-Knopp(H̃^res) : float32 =====
        H_res = self.sinkhorn(tilde_res)

        # =====================================================================
        # Part 2: Propagation (Eq.3 / Fig.1c)
        # Small-scale einsum uses fp32 for precision.
        # F_layer uses bf16 (standard practice for large GEMM).
        # =====================================================================

        x_l_fp32 = x_l.float()

        # --- Pre Mapping: h^in = H^pre @ x_l (Eq.3 inner term) ---
        h_in = torch.einsum("bi,bic->bc", H_pre, x_l_fp32)  # [B, C] fp32

        # --- Layer F: h^out = F(RMSNorm(h^in)) (Eq.3 / Table 3) ---
        h_in_norm = self.norm_layer(h_in)
        h_in_norm = h_in_norm.to(input_dtype)
        h_out = self.F_layer(h_in_norm)  # [B, C] bf16

        # --- Res Mapping: h^res = H^res @ x_l (Eq.3 term 1) ---
        h_res = torch.einsum("bij,bjc->bic", H_res, x_l_fp32)  # [B, n, C] fp32

        # --- Post Mapping: h^post = H^post^T @ h^out (Eq.3 term 2) ---
        h_out_fp32 = h_out.float()
        h_post = H_post.unsqueeze(-1) * h_out_fp32.unsqueeze(1)  # [B, n, C] fp32

        # --- Final: x_{l+1} = h^res + h^post (Eq.3) ---
        x_l_next = (h_res + h_post).to(input_dtype)

        return x_l_next


# =============================================================================
# Configuration & Testing
# =============================================================================

batch_size = 320
expansion_rate = 4

# Configurations based on Appendix A.1 Table 5 (3B model)
hidden_dim = 1280

sinkhorn_iters = 20
sinkhorn_eps = 1e-8
norm_eps = 1e-20
alpha_init = 0.01

# MoE configuration matching Table 5
moe_config = MoEConfig()


def get_inputs():
    """Input should be float16 per Eq.(11)."""
    return [torch.randn(batch_size, expansion_rate, hidden_dim, dtype=torch.float16)]


def get_init_inputs():
    """Arguments for Model.__init__"""
    return [hidden_dim, expansion_rate, sinkhorn_iters, sinkhorn_eps, norm_eps, alpha_init, moe_config]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = Model(*init_inputs)
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    print(output.shape)
    print(output.sum())

