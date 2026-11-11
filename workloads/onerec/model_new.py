# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""OneRec candidate with static indices and graph-safe all-padding handling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SOURCE = Path(__file__).with_name("model_ref.py")
_SPEC = importlib.util.spec_from_file_location(
    "workloadsuite_onerec_reference", _SOURCE
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load OneRec reference workload: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class Model(_MODULE.Model):
    def __init__(
        self,
        num_classes: int,
        sid_depth: int,
        max_hist_len: int,
        user_feat_dim: int,
        dim: int = 128,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        value_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_classes,
            sid_depth,
            max_hist_len,
            user_feat_dim,
            dim,
            num_heads,
            encoder_layers,
            decoder_layers,
            value_layers,
            dropout,
        )
        self.register_buffer(
            "_depth_positions",
            torch.arange(sid_depth).view(1, 1, sid_depth),
            persistent=False,
        )
        self.register_buffer(
            "_history_positions",
            torch.arange(max_hist_len).view(1, max_hist_len),
            persistent=False,
        )
        self.register_buffer(
            "_decoder_positions",
            torch.arange(sid_depth).view(1, sid_depth),
            persistent=False,
        )
        self.register_buffer(
            "_decoder_causal",
            torch.triu(torch.ones(sid_depth, sid_depth, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def encode_history(
        self,
        hist_sid: torch.Tensor,
        hist_len: torch.Tensor,
        user_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, hist_size, sid_depth = hist_sid.shape
        depth_positions = self._depth_positions[:, :, :sid_depth]
        sid_embedding = self.sid_embedding(hist_sid) + self.sid_position(
            depth_positions
        )
        item_vectors = self.sid_merge(
            sid_embedding.reshape(batch_size, hist_size, sid_depth * self.dim)
        )
        history_positions = self._history_positions[:, :hist_size]
        item_vectors = (
            item_vectors
            + self.hist_position(history_positions)
            + self.user_proj(user_feat).unsqueeze(1)
        )
        mask = (hist_sid == 0).all(dim=-1)
        length_mask = self._history_positions[:, :hist_size] >= hist_len.view(
            batch_size, 1
        )
        mask = mask | length_mask
        mask[:, -1] = mask[:, -1] & ~mask.all(dim=1)
        return self.history_encoder(item_vectors, src_key_padding_mask=mask), mask

    def build_decoder_input(self, target_sid: torch.Tensor) -> torch.Tensor:
        batch_size, sid_depth = target_sid.shape
        shifted = target_sid.new_zeros(batch_size, sid_depth)
        if sid_depth > 1:
            shifted[:, 1:] = target_sid[:, :-1]
        embedded = self.sid_embedding(shifted)
        embedded[:, 0, :] = self.bos
        return embedded + self.decoder_position(self._decoder_positions[:, :sid_depth])

    def causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        del device
        return self._decoder_causal[:length, :length]


OPTIMIZATION_METADATA = {
    "candidate_id": "onerec-graph-safe-static-indices-v1",
    "techniques": (
        "remove_data_dependent_mask_branch",
        "hoist_static_indices_and_causal_mask",
    ),
    "execution_mode": "eager",
    "preserves_state_dict": True,
    "requires_runtime_admission": True,
}


def get_optimization_metadata() -> dict[str, object]:
    return dict(OPTIMIZATION_METADATA)


get_inputs = _MODULE.get_inputs
get_init_inputs = _MODULE.get_init_inputs
