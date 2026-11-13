"""Conformer encoder inference workload for speech sequences.

Architecture references:
- Paper: https://arxiv.org/abs/2005.08100
- NVIDIA NeMo Conformer documentation:
  https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html

The implementation preserves the Conformer ordering: half-step feed-forward,
self-attention, convolution module, second half-step feed-forward and final
normalization.  It is self-contained and has no torchaudio/NeMo dependency.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3, dtype=torch.float16)
        self.output = nn.Linear(d_model, d_model, dtype=torch.float16)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = x.shape
        qkv = self.qkv(x).view(batch, sequence, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        probability = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        context = torch.matmul(probability, value)
        context = context.transpose(1, 2).reshape(batch, sequence, channels)
        return self.output(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, hidden_dim, dtype=torch.float16)
        self.output = nn.Linear(hidden_dim, d_model, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.input(x)))


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self.pointwise_in = nn.Conv1d(d_model, d_model * 2, kernel_size=1, dtype=torch.float16)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            dtype=torch.float16,
        )
        self.norm = nn.BatchNorm1d(d_model, dtype=torch.float16)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.glu(self.pointwise_in(x), dim=1)
        x = F.silu(self.norm(self.depthwise(x)))
        return self.pointwise_out(x).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(d_model, dtype=torch.float16)
        self.ffn1 = FeedForward(d_model, ffn_dim)
        self.attention_norm = nn.LayerNorm(d_model, dtype=torch.float16)
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.conv_norm = nn.LayerNorm(d_model, dtype=torch.float16)
        self.convolution = ConvolutionModule(d_model, kernel_size)
        self.ffn2_norm = nn.LayerNorm(d_model, dtype=torch.float16)
        self.ffn2 = FeedForward(d_model, ffn_dim)
        self.final_norm = nn.LayerNorm(d_model, dtype=torch.float16)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        x = x + self.attention(self.attention_norm(x), valid)
        x = x + self.convolution(self.conv_norm(x))
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 31,
        num_classes: int = 1024,
    ) -> None:
        super().__init__()
        reduced_frequency = (input_dim + 3) // 4
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1, dtype=torch.float16),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1, dtype=torch.float16),
            nn.ReLU(),
        )
        self.input_projection = nn.Linear(
            d_model * reduced_frequency, d_model, dtype=torch.float16
        )
        self.layers = nn.ModuleList(
            ConformerBlock(d_model, num_heads, ffn_dim, conv_kernel_size)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model, dtype=torch.float16)
        self.classifier = nn.Linear(d_model, num_classes, dtype=torch.float16)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.subsample(features.unsqueeze(1))
        batch, channels, time, frequency = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, time, channels * frequency)
        x = self.input_projection(x)
        reduced_lengths = (lengths + 3) // 4
        positions = torch.arange(time, device=x.device).unsqueeze(0)
        valid = positions < reduced_lengths.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, valid)
        x = self.output_norm(x)
        weights = valid.to(x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


def get_inputs() -> list[torch.Tensor]:
    features = torch.randn(4, 1024, 80, dtype=torch.float16)
    lengths = torch.tensor([1024, 901, 777, 640], dtype=torch.int64)
    return [features, lengths]


def get_init_inputs() -> list[int]:
    return [80, 256, 12, 4, 1024, 31, 1024]
