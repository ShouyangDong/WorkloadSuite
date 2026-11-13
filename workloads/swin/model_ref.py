"""Swin-Tiny style hierarchical vision Transformer workload.

Architecture references:
- Paper: https://arxiv.org/abs/2103.14030
- Official project: https://github.com/microsoft/Swin-Transformer

This clean-room PyTorch implementation retains patch embedding, shifted-window
attention, relative-position bias, patch merging and hierarchical stages.  It
avoids torchvision/timm dependencies so the workload remains self-contained.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)


def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int, batch: int
) -> torch.Tensor:
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, dtype=torch.float16)
        self.projection = nn.Linear(dim, dim, dtype=torch.float16)
        table_size = (2 * window_size - 1) ** 2
        self.relative_position_bias = nn.Parameter(
            torch.zeros(table_size, num_heads, dtype=torch.float16)
        )
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).view(
            batch_windows, tokens, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2) * self.scale
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention = torch.matmul(query, key.transpose(-1, -2))
        bias = self.relative_position_bias[
            self.relative_position_index.reshape(-1)
        ].view(tokens, tokens, self.num_heads)
        attention = attention + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask[:, None].unsqueeze(0)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        probability = torch.softmax(attention, dim=-1, dtype=torch.float32).to(value.dtype)
        output = torch.matmul(probability, value).transpose(1, 2).reshape(
            batch_windows, tokens, channels
        )
        return self.projection(output)


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * ratio, dtype=torch.float16)
        self.fc2 = nn.Linear(dim * ratio, dim, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SwinBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, window_size: int, shift_size: int
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim, dtype=torch.float16)
        self.attention = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim, dtype=torch.float16)
        self.mlp = Mlp(dim)

    def _mask(
        self, height: int, width: int, padded_height: int, padded_width: int, device
    ) -> torch.Tensor | None:
        if not self.shift_size:
            return None
        mask = torch.zeros((1, padded_height, padded_width, 1), device=device)
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = height_slices
        count = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                mask[:, height_slice, width_slice, :] = count
                count += 1
        windows = window_partition(mask, self.window_size).squeeze(-1)
        attention_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return attention_mask.masked_fill(attention_mask != 0, -100.0).masked_fill(
            attention_mask == 0, 0.0
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_width, 0, pad_height))
        padded_height, padded_width = height + pad_height, width + pad_width
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size)
        windows = self.attention(
            windows,
            self._mask(height, width, padded_height, padded_width, x.device),
        )
        x = window_reverse(
            windows, self.window_size, padded_height, padded_width, batch
        )
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x[:, :height, :width].reshape(batch, tokens, channels)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim * 4, dtype=torch.float16)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False, dtype=torch.float16)

    def forward(self, x: torch.Tensor, height: int, width: int):
        batch, _, channels = x.shape
        x = x.view(batch, height, width, channels)
        if height % 2 or width % 2:
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
            height, width = x.shape[1:3]
        x = torch.cat(
            (
                x[:, 0::2, 0::2],
                x[:, 1::2, 0::2],
                x[:, 0::2, 1::2],
                x[:, 1::2, 1::2],
            ),
            dim=-1,
        )
        return self.reduction(self.norm(x)).flatten(1, 2), height // 2, width // 2


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinBlock(
                dim,
                num_heads,
                window_size,
                0 if index % 2 == 0 else window_size // 2,
            )
            for index in range(depth)
        )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, height: int, width: int):
        for block in self.blocks:
            x = block(x, height, width)
        if self.downsample is not None:
            x, height, width = self.downsample(x, height, width)
        return x, height, width


class Model(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
    ) -> None:
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=4, stride=4, dtype=torch.float16
        )
        self.stages = nn.ModuleList()
        dim = embed_dim
        for index, (depth, heads) in enumerate(zip(depths, num_heads, strict=True)):
            self.stages.append(
                SwinStage(
                    dim,
                    depth,
                    heads,
                    window_size,
                    downsample=index < len(depths) - 1,
                )
            )
            if index < len(depths) - 1:
                dim *= 2
        self.norm = nn.LayerNorm(dim, dtype=torch.float16)
        self.head = nn.Linear(dim, num_classes, dtype=torch.float16)
        self.img_size = img_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(image)
        height, width = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        for stage in self.stages:
            x, height, width = stage(x, height, width)
        return self.head(self.norm(x).mean(dim=1))


def get_inputs() -> list[torch.Tensor]:
    return [torch.randn(2, 3, 224, 224, dtype=torch.float16)]


def get_init_inputs() -> list:
    return [224, 1000, 96, (2, 2, 6, 2), (3, 6, 12, 24), 7]
