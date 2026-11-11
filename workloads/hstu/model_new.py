# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""HSTU candidate that hoists static position and causal-mask construction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_SOURCE = Path(__file__).with_name("model_ref.py")
_SPEC = importlib.util.spec_from_file_location("workloadsuite_hstu_reference", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load HSTU reference workload: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class HSTUBlock(_MODULE.HSTUBlock):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        attention_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embedding_dim, num_heads, hidden_dim, attention_dim, dropout)
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

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
        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha
        causal = self._causal_mask[:seq_len, :seq_len]
        scores = F.silu(scores).masked_fill(~causal, 0.0) / float(seq_len)
        attention = (
            torch.matmul(scores, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        )
        gated = self.output_norm(attention) * F.silu(u)
        return residual + self.dropout(
            self.output(torch.cat((gated, u, residual), dim=-1))
        )


class Model(_MODULE.Model):
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
        super().__init__(
            vocab_size,
            max_seq_len,
            embedding_dim,
            num_heads,
            hidden_dim,
            attention_dim,
            num_layers,
            dropout,
            num_user_groups,
        )
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(
                    embedding_dim,
                    num_heads,
                    hidden_dim,
                    attention_dim,
                    max_seq_len,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.register_buffer(
            "_position_ids", torch.arange(max_seq_len), persistent=False
        )

    def forward(
        self, input_ids: torch.Tensor, user_group: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = self._position_ids[:seq_len].unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = x + self.user_embedding(user_group).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


OPTIMIZATION_METADATA = {
    "candidate_id": "hstu-static-index-mask-v1",
    "techniques": ("hoist_position_ids", "hoist_causal_mask"),
    "execution_mode": "eager",
    "preserves_state_dict": True,
    "requires_runtime_admission": True,
}


def get_optimization_metadata() -> dict[str, object]:
    return dict(OPTIMIZATION_METADATA)


get_inputs = _MODULE.get_inputs
get_init_inputs = _MODULE.get_init_inputs
