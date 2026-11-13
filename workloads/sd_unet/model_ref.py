"""Stable-Diffusion-style conditional U-Net inference workload.

Architecture references:
- Latent Diffusion paper: https://arxiv.org/abs/2112.10752
- Original Stable Diffusion project: https://github.com/CompVis/stable-diffusion

This is a compact clean-room PyTorch workload rather than copied upstream
source.  It retains the characteristic multi-scale residual convolution,
GroupNorm, skip concatenation, up/down sampling and cross-attention paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half, 1)
    )
    phase = timestep.float().unsqueeze(1) * frequency.unsqueeze(0)
    embedding = torch.cat((torch.cos(phase), torch.sin(phase)), dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, dtype=torch.float16)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dtype=torch.float16)
        self.time = nn.Linear(time_dim, out_channels, dtype=torch.float16)
        self.norm2 = nn.GroupNorm(32, out_channels, dtype=torch.float16)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, dtype=torch.float16)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, dtype=torch.float16)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time(F.silu(time)).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class CrossAttention2d(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels, dtype=torch.float16)
        self.query = nn.Linear(channels, channels, dtype=torch.float16)
        self.key = nn.Linear(context_dim, channels, dtype=torch.float16)
        self.value = nn.Linear(context_dim, channels, dtype=torch.float16)
        self.output = nn.Linear(channels, channels, dtype=torch.float16)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x
        spatial = self.norm(x).flatten(2).transpose(1, 2)
        query = self.query(spatial)
        key = self.key(context)
        value = self.value(context)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, tensor.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probability = torch.softmax(score, dim=-1, dtype=torch.float32).to(value.dtype)
        attended = torch.matmul(probability, value).transpose(1, 2)
        attended = attended.reshape(batch, height * width, channels)
        attended = self.output(attended).transpose(1, 2).reshape(batch, channels, height, width)
        return residual + attended


class Downsample(nn.Module):
    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, 3, stride=2, padding=1, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, 3, padding=1, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Model(nn.Module):
    def __init__(self, base_channels: int = 128, context_dim: int = 768) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim, dtype=torch.float16),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim, dtype=torch.float16),
        )
        self.input = nn.Conv2d(4, base_channels, 3, padding=1, dtype=torch.float16)
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = Downsample(base_channels, base_channels * 2)
        self.down2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.attention2 = CrossAttention2d(base_channels * 2, context_dim)
        self.downsample2 = Downsample(base_channels * 2, base_channels * 4)
        self.middle1 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.middle_attention = CrossAttention2d(base_channels * 4, context_dim)
        self.middle2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.upsample2 = Upsample(base_channels * 4, base_channels * 2)
        self.up2 = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up_attention2 = CrossAttention2d(base_channels * 2, context_dim)
        self.upsample1 = Upsample(base_channels * 2, base_channels)
        self.up1 = ResBlock(base_channels * 2, base_channels, time_dim)
        self.output_norm = nn.GroupNorm(32, base_channels, dtype=torch.float16)
        self.output = nn.Conv2d(base_channels, 4, 3, padding=1, dtype=torch.float16)

    def forward(
        self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        time = timestep_embedding(timestep, self.time_dim // 4).to(latent.dtype)
        time = self.time_mlp(time)
        x = self.input(latent)
        skip1 = self.down1(x, time)
        x = self.downsample1(skip1)
        skip2 = self.attention2(self.down2(x, time), context)
        x = self.downsample2(skip2)
        x = self.middle2(self.middle_attention(self.middle1(x, time), context), time)
        x = self.upsample2(x)
        x = self.up_attention2(self.up2(torch.cat((x, skip2), dim=1), time), context)
        x = self.upsample1(x)
        x = self.up1(torch.cat((x, skip1), dim=1), time)
        return self.output(F.silu(self.output_norm(x)))


def get_inputs() -> list[torch.Tensor]:
    latent = torch.randn(2, 4, 64, 64, dtype=torch.float16)
    timestep = torch.tensor([10.0, 731.0], dtype=torch.float32)
    context = torch.randn(2, 77, 768, dtype=torch.float16)
    return [latent, timestep, context]


def get_init_inputs() -> list[int]:
    return [128, 768]
