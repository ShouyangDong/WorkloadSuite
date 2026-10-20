import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import random
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
    dim: int = 1280                    # Hidden dimension
    moe_inter_dim: int = 896           # Expert intermediate dimension
    n_routed_experts: int = 64         # Total routed experts
    n_shared_experts: int = 2          # Shared experts (always active)
    n_activated_experts: int = 6       # Top-K active experts
    route_scale: float = 1.0           # Routing weight scale


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
        self.weight = nn.Parameter(
            torch.randn(config.n_routed_experts, config.dim, dtype=torch.float16) * config.dim ** -0.5
        )

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
        
        # Gating mechanism
        self.gate = Gate(config)
        
        # Routed experts
        self.experts = nn.ModuleList([
            Expert(config.dim, config.moe_inter_dim) 
            for _ in range(config.n_routed_experts)
        ])
        
        # Shared experts (fused as single larger MLP)
        # Hidden dim = n_shared_experts * moe_inter_dim
        self.shared_experts = MLP(
            config.dim, 
            config.n_shared_experts * config.moe_inter_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] or [B, S, D] input tensor
        Returns:
            output: Same shape as input
        """
        shape = x.size()
        x = x.view(-1, self.dim)  # [B*S, D]
        
        # Routing
        weights, indices = self.gate(x)  # [B*S, K], [B*S, K]
        
        # Routed experts computation
        y = torch.zeros_like(x)
        counts = torch.bincount(
            indices.flatten(), 
            minlength=self.n_routed_experts
        ).tolist()
        
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        
        # Shared experts (always active)
        z = self.shared_experts(x)
        
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
        expansion_rate: int = 4,       # n (Table 5)
        sinkhorn_iters: int = 20,      # t_max (Eq.9, Table 5)
        sinkhorn_eps: float = 1e-8,    # Not in paper, engineering choice
        norm_eps: float = 1e-20,       # RMSNorm epsilon (Table 5)
        alpha_init: float = 0.01,      # Gating factor init (Table 5)
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

        # # Enable TF32 matmul to emulate "tfloat32" compute (Eq.10)
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True

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
        tilde_tilde_pre = x_flat_fp32 @ self.phi_pre    # [B, n]
        tilde_tilde_post = x_flat_fp32 @ self.phi_post  # [B, n]
        tilde_tilde_res = x_flat_fp32 @ self.phi_res    # [B, n²]

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