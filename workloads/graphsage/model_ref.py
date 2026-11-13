"""Pure-PyTorch GraphSAGE inference workload.

Architecture references:
- Paper: https://arxiv.org/abs/1706.02216
- Original project: https://github.com/williamleif/GraphSAGE

This is a compact clean-room implementation of mean-aggregator GraphSAGE.  It
intentionally uses gather, ``index_add_`` and degree normalization so compiler
evaluation covers irregular graph memory access without requiring PyG/DGL.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanGraphSAGELayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim * 2, output_dim, dtype=torch.float16)
        self.norm = nn.LayerNorm(output_dim, dtype=torch.float16)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, destination = edge_index[0], edge_index[1]
        neighbors = x.index_select(0, source)
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, destination, neighbors)
        degree = torch.bincount(destination, minlength=x.shape[0]).clamp_min(1)
        aggregate = aggregate / degree.to(x.dtype).unsqueeze(-1)
        return F.relu(self.norm(self.projection(torch.cat((x, aggregate), dim=-1))))


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        dimensions = [input_dim, *([hidden_dim] * (num_layers - 1)), output_dim]
        self.layers = nn.ModuleList(
            MeanGraphSAGELayer(dimensions[index], dimensions[index + 1])
            for index in range(num_layers)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


NUM_NODES = 32768
NUM_EDGES = 262144
INPUT_DIM = 128


def get_inputs() -> list[torch.Tensor]:
    features = torch.randn(NUM_NODES, INPUT_DIM, dtype=torch.float16)
    source = torch.arange(NUM_EDGES, dtype=torch.int64) % NUM_NODES
    destination = (source * 1103515245 + 12345) % NUM_NODES
    edge_index = torch.stack((source, destination), dim=0)
    return [features, edge_index]


def get_init_inputs() -> list[int]:
    return [INPUT_DIM, 256, 128, 3]
