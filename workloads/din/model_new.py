"""DIN candidate with a library-backed batched weighted reduction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SOURCE = Path(__file__).with_name("model_ref.py")
_SPEC = importlib.util.spec_from_file_location("workloadsuite_din_reference", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load DIN reference workload: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class Model(_MODULE.Model):
    """Preserve DIN parameters while lowering the weighted sum to batched GEMM."""

    def __init__(self, vocab_size: int, dim: int, dense_dim: int) -> None:
        super().__init__(vocab_size, dim, dense_dim)
        self.register_buffer(
            "_history_positions",
            torch.arange(_MODULE.SEQUENCE_LENGTH).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        dense: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        history_embedding = self.embedding(history)
        target_embedding = self.embedding(target)
        expanded = target_embedding.unsqueeze(1).expand_as(history_embedding)
        features = torch.cat(
            (
                history_embedding,
                expanded,
                history_embedding - expanded,
                history_embedding * expanded,
            ),
            dim=-1,
        )
        scores = self.attention(features).squeeze(-1)
        positions = self._history_positions[:, : history.size(1)]
        mask = positions < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        interest = torch.bmm(weights.unsqueeze(1), history_embedding).squeeze(1)
        return self.head(
            torch.cat((interest, target_embedding, dense), dim=-1)
        ).squeeze(-1)


OPTIMIZATION_METADATA = {
    "candidate_id": "din-library-weighted-reduction-v1",
    "techniques": ("batched_library_weighted_reduction", "hoist_history_positions"),
    "execution_mode": "eager",
    "preserves_state_dict": True,
    "requires_runtime_admission": True,
}


def get_optimization_metadata() -> dict[str, object]:
    return dict(OPTIMIZATION_METADATA)


get_inputs = _MODULE.get_inputs
get_init_inputs = _MODULE.get_init_inputs
