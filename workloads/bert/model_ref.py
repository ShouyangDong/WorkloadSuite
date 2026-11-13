"""BERT-Base style bidirectional Transformer encoder workload.

Architecture references:
- Paper: https://arxiv.org/abs/1810.04805
- Original project: https://github.com/google-research/bert

This clean-room PyTorch implementation targets inference classification and
retains token/position/type embeddings, bidirectional masked attention, GELU
feed-forward blocks and post-residual LayerNorm.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, dtype=torch.float16)
        self.output = nn.Linear(hidden_size, hidden_size, dtype=torch.float16)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = x.shape
        qkv = self.qkv(x).view(batch, sequence, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        valid = attention_mask[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).reshape(batch, sequence, hidden)
        return self.output(context)


class BertLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_heads)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12, dtype=torch.float16)
        self.intermediate = nn.Linear(hidden_size, intermediate_size, dtype=torch.float16)
        self.output = nn.Linear(intermediate_size, hidden_size, dtype=torch.float16)
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12, dtype=torch.float16)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.attention_norm(x + self.attention(x, attention_mask))
        feed_forward = self.output(F.gelu(self.intermediate(x)))
        return self.output_norm(x + feed_forward)


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_seq_len: int = 512,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, dtype=torch.float16)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size, dtype=torch.float16)
        self.type_embeddings = nn.Embedding(2, hidden_size, dtype=torch.float16)
        self.embedding_norm = nn.LayerNorm(hidden_size, eps=1e-12, dtype=torch.float16)
        self.layers = nn.ModuleList(
            BertLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)
        )
        self.pooler = nn.Linear(hidden_size, hidden_size, dtype=torch.float16)
        self.classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float16)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.type_embeddings(token_type_ids)
        )
        x = self.embedding_norm(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        pooled = torch.tanh(self.pooler(x[:, 0]))
        return self.classifier(pooled)


BATCH_SIZE = 4
SEQUENCE_LENGTH = 512


def get_inputs() -> list[torch.Tensor]:
    input_ids = torch.randint(0, 30522, (BATCH_SIZE, SEQUENCE_LENGTH), dtype=torch.int64)
    attention_mask = torch.ones(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.bool)
    attention_mask[1, -32:] = False
    attention_mask[2, -96:] = False
    attention_mask[3, -160:] = False
    token_type_ids = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.int64)
    token_type_ids[:, SEQUENCE_LENGTH // 2 :] = 1
    return [input_ids, attention_mask, token_type_ids]


def get_init_inputs() -> list[int]:
    return [30522, 768, 12, 12, 3072, 512, 2]
