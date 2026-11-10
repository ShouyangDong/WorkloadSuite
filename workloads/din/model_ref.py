"""Compact APPRENTICE DIN inference workload."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Deep Interest Network with target-aware history attention."""

    def __init__(self, vocab_size: int, dim: int, dense_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.PReLU(),
            nn.Linear(dim * 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(dim * 2 + dense_dim, dim * 2),
            nn.PReLU(),
            nn.Linear(dim * 2, 1),
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
        positions = torch.arange(history.size(1), device=history.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        interest = torch.sum(history_embedding * weights.unsqueeze(-1), dim=1)
        return self.head(torch.cat((interest, target_embedding, dense), dim=-1)).squeeze(-1)


BATCH_SIZE = 64
SEQUENCE_LENGTH = 50
VOCAB_SIZE = 10_000
EMBEDDING_DIM = 32
DENSE_DIM = 8


def get_inputs() -> list[torch.Tensor]:
    return [
        torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH)),
        torch.randint(1, VOCAB_SIZE, (BATCH_SIZE,)),
        torch.randn(BATCH_SIZE, DENSE_DIM),
        torch.full((BATCH_SIZE,), SEQUENCE_LENGTH, dtype=torch.long),
    ]


def get_init_inputs() -> list[int]:
    return [VOCAB_SIZE, EMBEDDING_DIM, DENSE_DIM]


if __name__ == "__main__":
    model = Model(*get_init_inputs()).eval()
    with torch.inference_mode():
        print(model(*get_inputs()).shape)
