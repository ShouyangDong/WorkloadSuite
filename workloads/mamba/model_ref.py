"""
Mamba-2 Structured State Space Duality (SSD) implementation.

Clean version: single 1.3B config in a simple file layout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class SSDLayer(nn.Module):
    """
    Single Mamba-2 SSD block.
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
            bias=True,  # Official impl uses conv bias
            dtype=torch.float16,
        )

        # Keep A in fp32 for stability.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.n_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.n_heads, dtype=torch.float16))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float16) - 4.0)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, dtype=torch.float16)
        self.norm = nn.RMSNorm(self.d_model, dtype=torch.float16)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = u.shape

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
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :seq_len]
        x = F.silu(x)
        x = rearrange(x, "b d l -> b l d")

        # Reshape.
        x = rearrange(x, "b l (h p) -> b l h p", h=self.n_heads, p=self.d_head)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.n_groups, n=self.d_state)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.n_groups, n=self.d_state)

        # dt discretization.
        dt = F.softplus(dt + self.dt_bias)

        # GQA-style broadcast.
        ratio = self.n_heads // self.n_groups
        B = repeat(B, "b l g n -> b l (g r) n", r=ratio)
        C = repeat(C, "b l g n -> b l (g r) n", r=ratio)

        # Discretize A.
        A = -torch.exp(self.A_log)  # fp32
        A_dt = A * dt.float()

        # Chunk.
        X_b = rearrange(x, "b (c l) h p -> b c l h p", l=self.block_len)
        B_b = rearrange(B, "b (c l) h n -> b c l h n", l=self.block_len)
        C_b = rearrange(C, "b (c l) h n -> b c l h n", l=self.block_len)
        dt_b = rearrange(dt, "b (c l) h -> b c l h", l=self.block_len)
        A_dt_b = rearrange(A_dt, "b (c l) h -> b c l h", l=self.block_len)

        Y = self._ssd_core(X_b, B_b, C_b, dt_b, A_dt_b, batch)

        # Skip + gate + projection.
        Y = rearrange(Y, "b c l h p -> b (c l) h p")
        Y = Y + x * self.D.view(1, 1, -1, 1)
        Y = rearrange(Y, "b l h p -> b l (h p)")
        Y = Y * F.silu(z)
        Y = self.out_proj(Y)

        return Y

    def _ssd_core(
        self,
        X_b: torch.Tensor,
        B_b: torch.Tensor,
        C_b: torch.Tensor,
        dt_b: torch.Tensor,
        A_dt_b: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
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


class Model(nn.Module):
    """
    Multi-layer Mamba-2 SSD model (stacked SSDLayer with residuals).
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
    return [torch.randn(batch_size, seq_length, d_model, dtype=torch.float16, device="cuda")]


def get_init_inputs():
    return [d_model, n_layers, n_heads, d_state, d_head, n_groups, block_len, conv_kernel]


if __name__ == "__main__":
    print("=" * 60)
    print("Mamba-2 SSD clean model (single 1.3B config)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    # Note: fp16 reduction may overflow for large tensors; accumulate in fp32 for printing.
    print(f"Output: {tuple(y.shape)}, sum={y.float().sum().item():.4f}")
    print("=" * 60)