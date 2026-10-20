"""
Optimized: Mamba-2 SSD optimized (merged from model_lego.py + split_0/)

Original structure: model_lego.py imports from split_0/{chunk,scan,output}/model_new.py
This file merges all 4 files into one self-contained module.

Source: 
"""
from __future__ import annotations

from einops import rearrange
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# === scan/model_new.py (Triton SSD kernel) ===



@triton.autotune(
    configs=[
        triton.Config({"P_BLOCK": 16}, num_warps=4, num_stages=3),
        triton.Config({"P_BLOCK": 32}, num_warps=4, num_stages=3),
        triton.Config({"P_BLOCK": 32}, num_warps=8, num_stages=3),
        triton.Config({"P_BLOCK": 32}, num_warps=8, num_stages=4),
    ],
    key=[],
)
@triton.jit
def ssd_scan_fused_kernel(
    # X: (B, C, L, H, P) fp16
    x_ptr,
    # B/C: (B, C, L, H, N) fp16
    b_ptr,
    c_ptr,
    # dt: (B, C, L, H) fp16  (already softplus-ed)
    dt_ptr,
    # params
    a_log_ptr,  # (H,) fp32
    d_ptr,  # (H,) fp16
    # output: Y_inner (B, C, L, H, P) fp16
    y_ptr,
    # shapes
    batch: tl.constexpr,
    n_chunks: tl.constexpr,
    block_len: tl.constexpr,
    n_heads: tl.constexpr,
    d_state: tl.constexpr,
    d_head: tl.constexpr,
    # strides (element strides)
    sx_b: tl.constexpr,
    sx_c: tl.constexpr,
    sx_l: tl.constexpr,
    sx_h: tl.constexpr,
    sx_p: tl.constexpr,
    sb_b: tl.constexpr,
    sb_c: tl.constexpr,
    sb_l: tl.constexpr,
    sb_h: tl.constexpr,
    sb_n: tl.constexpr,
    sc_b: tl.constexpr,
    sc_c: tl.constexpr,
    sc_l: tl.constexpr,
    sc_h: tl.constexpr,
    sc_n: tl.constexpr,
    sdt_b: tl.constexpr,
    sdt_c: tl.constexpr,
    sdt_l: tl.constexpr,
    sdt_h: tl.constexpr,
    sy_b: tl.constexpr,
    sy_c: tl.constexpr,
    sy_l: tl.constexpr,
    sy_h: tl.constexpr,
    sy_p: tl.constexpr,
    # meta
    P_BLOCK: tl.constexpr,
):
    """
    Fused SSD core for the production ssd config.

    Program mapping:
      - pid0: (batch, head)
      - pid1: p_block (tile over d_head)

    For each (b, h, p_block), this kernel scans chunks sequentially, computing:
      - Y_diag (intra-chunk)
      - Y_off  (state-to-output, off-diagonal)
      - skip   (x * D)
      - and updates scan state for the next chunk.
    """
    pid_bh = tl.program_id(0)
    pid_pb = tl.program_id(1)

    b = pid_bh // n_heads
    h = pid_bh - b * n_heads

    # Specialize for benchmark shapes for performance and to keep the kernel manageable.
    tl.static_assert(block_len == 64, "This kernel targets block_len=64.")
    tl.static_assert(d_state == 128, "This kernel targets d_state=128.")
    tl.static_assert(d_head == 64, "This kernel targets d_head=64.")

    # p tile
    p0 = pid_pb * P_BLOCK
    p = p0 + tl.arange(0, P_BLOCK)
    p_mask = p < d_head

    l = tl.arange(0, block_len)
    n = tl.arange(0, d_state)

    LOG2E: tl.constexpr = 1.4426950408889634

    # Load per-head params.
    a_log = tl.load(a_log_ptr + h).to(tl.float32)
    A = -tl.exp(a_log)  # f32
    D = tl.load(d_ptr + h).to(tl.float16)

    # Running state: (N, P_BLOCK) in fp32.
    state = tl.zeros((d_state, P_BLOCK), dtype=tl.float32)

    # Scan chunks in-kernel.
    for c in range(n_chunks):
        # -----------------------------
        # Load chunk inputs
        # -----------------------------
        # X: (L, P_BLOCK)
        x_offs = (
            b * sx_b
            + c * sx_c
            + l[:, None] * sx_l
            + h * sx_h
            + p[None, :] * sx_p
        )
        X = tl.load(x_ptr + x_offs, mask=p_mask[None, :], other=0.0).to(tl.float16)

        # dt: (L,)
        dt_offs = b * sdt_b + c * sdt_c + l * sdt_l + h * sdt_h
        dt = tl.load(dt_ptr + dt_offs).to(tl.float32)  # f32

        # B/C: (L, N)
        # NOTE: these are large loads (64x128); keep fp16 to use tensor cores.
        b_offs = (
            b * sb_b
            + c * sb_c
            + l[:, None] * sb_l
            + h * sb_h
            + n[None, :] * sb_n
        )
        B = tl.load(b_ptr + b_offs).to(tl.float16)
        c_offs = (
            b * sc_b
            + c * sc_c
            + l[:, None] * sc_l
            + h * sc_h
            + n[None, :] * sc_n
        )
        C = tl.load(c_ptr + c_offs).to(tl.float16)

        # -----------------------------
        # Compute A_dt cumsum and decays
        # -----------------------------
        A_dt = A * dt  # (L,) f32
        A_cum = tl.cumsum(A_dt, axis=0)  # (L,) f32

        # exp(A_cum) and exp(sum A_dt)
        expA = tl.math.exp2(A_cum * LOG2E)  # (L,) f32
        a_end = tl.sum(A_dt, axis=0)  # scalar f32
        total_decay = tl.math.exp2(a_end * LOG2E)  # scalar f32

        # decay_to_end[l] = exp(a_end - A_cum[l])
        decay_to_end = tl.math.exp2((a_end - A_cum) * LOG2E).to(tl.float16)  # (L,) f16

        # X_scaled: (L,P) = X * dt
        X_scaled = (X.to(tl.float32) * dt[:, None]).to(tl.float16)

        # -----------------------------
        # Y_diag: intra-chunk (diagonal)
        # -----------------------------
        # M = C @ B^T  => (L,L)
        M = tl.dot(C, tl.trans(B)).to(tl.float16)

        tri = l[:, None] >= l[None, :]
        diff = A_cum[:, None] - A_cum[None, :]  # f32
        Lw = tl.where(tri, tl.math.exp2(diff * LOG2E), 0.0).to(tl.float16)  # (L,L) f16

        # SSD = Lw * M
        SSD = Lw * M  # (L,L) f16
        Y_diag = tl.dot(SSD, X_scaled).to(tl.float32)  # (L,P) f32

        # -----------------------------
        # Y_off: state-to-output (off-diagonal)
        # -----------------------------
        # Keep state in fp32, but cast to fp16 for tensor-core dot (matches ref strategy of fp16 storage for states).
        Y_state = tl.dot(C, state.to(tl.float16)).to(tl.float32)  # (L,P) f32
        Y_off = Y_state * expA[:, None]  # (L,P) f32

        # -----------------------------
        # Combine + skip, store
        # -----------------------------
        Y = Y_diag + Y_off + X.to(tl.float32) * D.to(tl.float32)
        y_offs = (
            b * sy_b
            + c * sy_c
            + l[:, None] * sy_l
            + h * sy_h
            + p[None, :] * sy_p
        )
        tl.store(y_ptr + y_offs, Y.to(tl.float16), mask=p_mask[None, :])

        # -----------------------------
        # Update scan state for next chunk
        # -----------------------------
        X_decayed = (X_scaled.to(tl.float32) * decay_to_end[:, None].to(tl.float32)).to(tl.float16)
        block_state = tl.dot(tl.trans(B), X_decayed).to(tl.float32)  # (N,P)
        state = state * total_decay + block_state



# === chunk/model_new.py (prepare_ssd_inputs) ===



def prepare_ssd_inputs(
    *,
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    n_heads: int,
    d_head: int,
    n_groups: int,
    d_state: int,
    block_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare chunked tensors for the fused Triton SSD scan kernel.

    Inputs are post-conv `x` (B, S, d_inner) and linear-projected `B/C/dt` as in the reference.
    Returns:
      X_b:  (B, C, L, H, P) fp16
      B_b:  (B, C, L, H, N) fp16
      C_b:  (B, C, L, H, N) fp16
      dt_b: (B, C, L, H)    fp16
    """
    bsz, seq_len, d_inner = x.shape
    assert d_inner == n_heads * d_head
    assert seq_len % block_len == 0
    n_chunks = seq_len // block_len

    # Reshape X -> (B, S, H, P)
    X = rearrange(x, "b s (h p) -> b s h p", h=n_heads, p=d_head)

    # Reshape B/C -> (B, S, H, N) with GQA broadcast.
    B = rearrange(B, "b s (g n) -> b s g n", g=n_groups, n=d_state)
    C = rearrange(C, "b s (g n) -> b s g n", g=n_groups, n=d_state)
    ratio = n_heads // n_groups
    B = repeat(B, "b s g n -> b s (g r) n", r=ratio)
    C = repeat(C, "b s g n -> b s (g r) n", r=ratio)

    # dt: (B,S,H)
    dt = F.softplus(dt)

    # Chunk views (no copies).
    X_b = X.reshape(bsz, n_chunks, block_len, n_heads, d_head)
    B_b = B.reshape(bsz, n_chunks, block_len, n_heads, d_state)
    C_b = C.reshape(bsz, n_chunks, block_len, n_heads, d_state)
    dt_b = dt.reshape(bsz, n_chunks, block_len, n_heads)
    return X_b, B_b, C_b, dt_b



# === output/model_new.py (finish_ssd_output) ===



def finish_ssd_output(
    *,
    y_inner: torch.Tensor,
    z: torch.Tensor,
    out_proj: torch.nn.Linear,
) -> torch.Tensor:
    """
    Finish SSD forward:
      - y_inner is (B, C, L, H, P) fp16 from the fused core
      - z is (B, S, d_inner) fp16 gate pre-activation from in_proj split
    """
    bsz, n_chunks, block_len, n_heads, d_head = y_inner.shape
    seq_len = n_chunks * block_len
    y = y_inner.reshape(bsz, seq_len, n_heads * d_head)
    y = y * F.silu(z)
    return out_proj(y)



# === model_lego.py (main model) ===
"""
Optimized Mamba-2 SSD implementation for the production (1.3B) config (full 24-layer model).

Notes:
- Parameter structure (state_dict keys) matches `run/mamba_full/model.py` exactly so the
  local `run/mamba_full/verify_model_new.py` can strict-load weights.
- The SSD core inside each layer is replaced with a Triton fused kernel (intra-chunk + scan
  + off-diagonal), organized under `run/mamba_full/split_0/*` for readability.
"""






class SSDLayer(nn.Module):
    """
    Single Mamba-2 SSD block (optimized forward).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_state: int,
        d_head: int,
        n_groups: int,
        block_len: int = 64,
        conv_kernel: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = d_head
        self.n_groups = n_groups
        self.block_len = block_len

        if self.n_heads % self.n_groups != 0:
            raise ValueError("n_heads must be divisible by n_groups")

        self.d_inner = self.n_heads * self.d_head
        self.d_proj = (
            self.d_inner  # x
            + self.d_inner  # z (gate)
            + self.n_groups * self.d_state  # B
            + self.n_groups * self.d_state  # C
            + self.n_heads  # dt
        )

        self.in_proj = nn.Linear(self.d_model, self.d_proj, bias=False, dtype=torch.float16)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.d_inner,
            bias=True,
            dtype=torch.float16,
        )

        # Keep A in fp32 for stability.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.n_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.n_heads, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float16) - 4.0)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, dtype=torch.float16)
        self.norm = nn.RMSNorm(self.d_model, dtype=torch.float16)

    def _native_forward(self, u: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """Non-CUDA (e.g. MLU) fallback: reference SSD core in pure PyTorch.

        The fused ``ssd_scan_fused_kernel`` is CUDA/Triton only, so on devices
        like MLU we fall back to the equivalent native implementation to keep
        the benchmark runnable and correctness-preserving.
        """
        u = self.norm(u)
        zxbcdt = self.in_proj(u)
        x, z, B, C, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state, self.n_heads],
            dim=-1,
        )

        # Short depthwise conv (causal) + activation.
        x = rearrange(x, "b s d -> b d s")
        x = self.conv1d(x)[..., :seq_len]
        x = F.silu(x)
        x = rearrange(x, "b d s -> b s d")

        # dt discretization (same semantics as the Triton path / ref).
        dt = dt + self.dt_bias
        X_b, B_b, C_b, dt_b = prepare_ssd_inputs(
            x=x,
            B=B,
            C=C,
            dt=dt,
            n_heads=self.n_heads,
            d_head=self.d_head,
            n_groups=self.n_groups,
            d_state=self.d_state,
            block_len=self.block_len,
        )
        A = -torch.exp(self.A_log)  # fp32
        A_dt_b = A * dt_b.float()

        Y = self._ssd_core(X_b, B_b, C_b, dt_b, A_dt_b, batch)
        Y = rearrange(Y, "b c l h p -> b (c l) h p")
        Y = Y + rearrange(x, "b l (h p) -> b l h p", h=self.n_heads, p=self.d_head) * self.D.view(1, 1, -1, 1)
        Y = rearrange(Y, "b l h p -> b l (h p)")
        Y = Y * F.silu(z)
        return self.out_proj(Y)

    def _ssd_core(self, X_b, B_b, C_b, dt_b, A_dt_b, batch):
        """Native SSD core (ported from the reference implementation)."""
        device = X_b.device
        n_chunks = X_b.shape[1]

        A_cumsum = torch.cumsum(A_dt_b, dim=2)  # [b, c, l, h]
        A_cumsum = rearrange(A_cumsum, "b c l h -> b c h l")

        # Intra-chunk (diagonal).
        L = torch.exp(A_cumsum[..., :, None] - A_cumsum[..., None, :])
        mask = torch.tril(torch.ones(self.block_len, self.block_len, device=device, dtype=torch.bool))
        L = L.masked_fill(~mask, 0.0)

        M = torch.einsum("bclhn, bcshn -> bchls", C_b, B_b)
        X_scaled = X_b * dt_b.unsqueeze(-1)
        Y_diag = torch.einsum("bchls, bcshp -> bclhp", (L * M).to(torch.float16), X_scaled)

        # Inter-chunk (recurrence).
        decay_to_end = torch.exp(A_cumsum[..., -1:] - A_cumsum).to(torch.float16)
        block_state = torch.einsum("bclhn, bchl, bclhp -> bchnp", B_b, decay_to_end, X_scaled)
        chunk_decay = torch.exp(A_cumsum[..., -1])  # fp32

        states = []
        h = torch.zeros(batch, self.n_heads, self.d_state, self.d_head, device=device, dtype=torch.float32)
        for i in range(n_chunks):
            states.append(h.to(torch.float16))
            h = h * chunk_decay[:, i, :, None, None] + block_state[:, i].float()
        states = torch.stack(states, dim=1)

        # State-to-output (off-diagonal).
        decay_from_start = torch.exp(A_cumsum).to(torch.float16)
        Y_off = torch.einsum("bclhn, bchnp, bchl -> bclhp", C_b, states, decay_from_start)
        return Y_diag + Y_off

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        assert u.dtype == torch.float16, "u must be float16"
        batch, seq_len, _ = u.shape
        assert seq_len % self.block_len == 0

        # The fused Triton kernel is CUDA-only; fall back on other devices (e.g. MLU).
        if u.device.type != "cuda":
            return self._native_forward(u, batch, seq_len)

        # Normalize.
        u = self.norm(u)

        # Project.
        zxbcdt = self.in_proj(u)
        x, z, B, C, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state, self.n_heads],
            dim=-1,
        )

        # Short depthwise conv (causal) + activation.
        x = rearrange(x, "b s d -> b d s")
        x = self.conv1d(x)[..., :seq_len]
        x = F.silu(x)
        x = rearrange(x, "b d s -> b s d")

        # dt discretization (keep same semantics as ref: softplus(dt + dt_bias)).
        dt = dt + self.dt_bias

        # Prepare chunked inputs (views, no copies except the GQA repeat which matches ref semantics).
        X_b, B_b, C_b, dt_b = prepare_ssd_inputs(
            x=x,
            B=B,
            C=C,
            dt=dt,
            n_heads=self.n_heads,
            d_head=self.d_head,
            n_groups=self.n_groups,
            d_state=self.d_state,
            block_len=self.block_len,
        )
        n_chunks = X_b.shape[1]

        # Allocate output in chunked layout then fuse core.
        Y_b = torch.empty_like(X_b, dtype=torch.float16)

        grid = lambda meta: (batch * self.n_heads, triton.cdiv(self.d_head, meta["P_BLOCK"]))
        ssd_scan_fused_kernel[grid](
            X_b,
            B_b,
            C_b,
            dt_b,
            self.A_log,
            self.D,
            Y_b,
            batch=batch,
            n_chunks=n_chunks,
            block_len=self.block_len,
            n_heads=self.n_heads,
            d_state=self.d_state,
            d_head=self.d_head,
            sx_b=X_b.stride(0),
            sx_c=X_b.stride(1),
            sx_l=X_b.stride(2),
            sx_h=X_b.stride(3),
            sx_p=X_b.stride(4),
            sb_b=B_b.stride(0),
            sb_c=B_b.stride(1),
            sb_l=B_b.stride(2),
            sb_h=B_b.stride(3),
            sb_n=B_b.stride(4),
            sc_b=C_b.stride(0),
            sc_c=C_b.stride(1),
            sc_l=C_b.stride(2),
            sc_h=C_b.stride(3),
            sc_n=C_b.stride(4),
            sdt_b=dt_b.stride(0),
            sdt_c=dt_b.stride(1),
            sdt_l=dt_b.stride(2),
            sdt_h=dt_b.stride(3),
            sy_b=Y_b.stride(0),
            sy_c=Y_b.stride(1),
            sy_l=Y_b.stride(2),
            sy_h=Y_b.stride(3),
            sy_p=Y_b.stride(4),
        )

        return finish_ssd_output(y_inner=Y_b, z=z, out_proj=self.out_proj)


class Model(nn.Module):
    """
    Multi-layer Mamba-2 SSD model (stacked SSDLayer with residuals) - optimized forward.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_state: int,
        d_head: int,
        n_groups: int,
        block_len: int = 64,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [
                SSDLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_state=d_state,
                    d_head=d_head,
                    n_groups=n_groups,
                    block_len=block_len,
                    conv_kernel=conv_kernel,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(d_model, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return self.final_norm(x)


# ============================================================
# Production config: 1.3B
# ============================================================

d_model = 2048
n_layers = 24
n_heads = 128
d_state = 128
d_head = 64
n_groups = 8
block_len = 64
conv_kernel = 4

batch_size = 8
seq_length = 2048


def get_inputs():
    return [torch.randn(batch_size, seq_length, d_model, dtype=torch.float16)]


def get_init_inputs():
    return [d_model, n_layers, n_heads, d_state, d_head, n_groups, block_len, conv_kernel]


if __name__ == "__main__":
    print("=" * 60)
    print("Mamba-2 SSD clean model (full 1.3B config) - optimized model_new")
    print("=" * 60)

    device = "mlu" if torch.mlu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(*get_init_inputs()).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Config: d={d_model}, layers={n_layers}, heads={n_heads}, "
        f"state={d_state}, head_dim={d_head}, groups={n_groups}"
    )
    print(f"Input:  batch={batch_size}, seq={seq_length}")
    print(f"Params: {total_params:,} ({total_params/1e9:.3f}B)")
    print("-" * 60)

    with torch.no_grad():
        x = torch.randn(batch_size, seq_length, d_model, dtype=torch.float16, device=device)
        y = model(x)
    print(f"Output: {tuple(y.shape)}, sum={y.float().sum().item():.4f}")
    print("=" * 60)

