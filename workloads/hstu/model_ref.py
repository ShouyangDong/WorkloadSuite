# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact APPRENTICE HSTU inference workload."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class HSTUBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attn_alpha = 1.0 / math.sqrt(attention_dim)
        projected_dim = 2 * num_heads * (hidden_dim + attention_dim)
        self.input_norm = RMSNorm(embedding_dim)
        self.uvqk = nn.Linear(embedding_dim, projected_dim)
        self.output_norm = RMSNorm(num_heads * hidden_dim)
        self.output = nn.Linear(num_heads * hidden_dim * 2 + embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        uvqk = self.uvqk(self.input_norm(x))
        u, v, q, k = torch.split(
            uvqk,
            [
                self.num_heads * self.hidden_dim,
                self.num_heads * self.hidden_dim,
                self.num_heads * self.attention_dim,
                self.num_heads * self.attention_dim,
            ],
            dim=-1,
        )
        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = F.silu(scores).masked_fill(~causal, 0.0) / float(seq_len)
        attention = torch.matmul(scores, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        gated = self.output_norm(attention) * F.silu(u)
        return residual + self.dropout(self.output(torch.cat((gated, u, residual), dim=-1)))


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 32,
        attention_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_user_groups: int = 32,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_user_groups + 1, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    attention_dim=attention_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output.weight = self.item_embedding.weight

    def forward(self, input_ids: torch.Tensor, user_group: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = x + self.user_embedding(user_group).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


BATCH_SIZE = 8
SEQUENCE_LENGTH = 32
VOCAB_SIZE = 256
EMBEDDING_DIM = 32
NUM_HEADS = 4
HIDDEN_DIM = 8
ATTENTION_DIM = 8
NUM_LAYERS = 1
DROPOUT = 0.0


def get_inputs() -> list[torch.Tensor]:
    return [
        torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH)),
        torch.randint(1, 32, (BATCH_SIZE,)),
    ]


def get_init_inputs() -> list[int | float]:
    return [
        VOCAB_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        HIDDEN_DIM,
        ATTENTION_DIM,
        NUM_LAYERS,
        DROPOUT,
    ]


if __name__ == "__main__":
    model = Model(*get_init_inputs()).eval()
    with torch.inference_mode():
        print(model(*get_inputs()).shape)
