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

    def forward(self, x, cos, sin, mask=None):
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

        n_rep = self.num_heads // self.num_kv_heads
        k = k[:, :, None, :, :].expand(batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v[:, :, None, :, :].expand(batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim).reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # Scaled Dot Product Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
        output = torch.matmul(attn_weights, v)

        # Reshape and Output Projection
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Qwen3Attention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)

    def forward(self, x, cos, sin, mask=None):
        """
        Block structure:
        x = x + Attn(LN(x))
        x = x + MLP(LN(x))
        """
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

class Model(nn.Module):
    def __init__(self, vocab_size=151936, hidden_size=2048, num_layers=16, 
                 num_heads=16, num_kv_heads=16, intermediate_size=5504, 
                 max_position_embeddings=40960, rope_theta=1000000.0):
        """
        Implementation of Qwen3ForCausalLM.
        Default params roughly match a smaller Qwen configuration.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.float16)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(hidden_size, intermediate_size, num_heads, num_kv_heads)
            for _ in range(num_layers)
        ])
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
        _, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        # Causal Attention Mask
        # (batch, 1, seq, seq) - Lower triangular
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=torch.float16)
        mask = torch.triu(mask, diagonal=1)
        mask = mask[None, None, :, :]

        # Retrieve RoPE caches
        cos = self.cos.to(torch.float16)
        sin = self.sin.to(torch.float16)

        for layer in self.layers:
            x = layer(x, cos, sin, mask)

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


if __name__ == '__main__':
    torch.set_default_device("mlu" if torch.mlu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Model()
    model.eval()
    inputs = get_inputs()
    out = model(*inputs)
    print("max:", out.max().item(), "min:", out.min().item())
    print(out.shape, out.dtype)

