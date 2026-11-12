# WorkloadSuite

Benchmark model workloads for MASTER. Each workload ships one canonical
PyTorch implementation and compares eager execution with `torch.compile`.

## Workloads

| Workload    | Type               | Model        | Implementation |
|-------------|--------------------|--------------|----------------|
| `resnet`    | CNN                | ResNet-18    | `model_ref.py` |
| `qwen`      | LLM decoder        | Qwen3-8B     | `model_ref.py` |
| `sd3_mmdit` | Diffusion transformer | SD3-MMDiT | `model_ref.py` |
| `mamba`     | State space model  | Mamba-2 SSD  | `model_ref.py` |
| `ds_mhc_moe`| Mixture of experts | mHC-MoE      | `model_ref.py` |
| `din`       | Recommender        | DIN          | `model_ref.py` |
| `dien`      | Recommender        | DIEN         | `model_ref.py` |
| `hstu`      | Sequential recommender | HSTU     | `model_ref.py` |
| `onerec`    | Generative recommender | OneRec   | `model_ref.py` |

Each workload is self-contained in `model_ref.py`, which exposes `Model`,
`get_init_inputs()` and `get_inputs()`. The suite intentionally does not bundle
pre-optimized `model_new.py` implementations: benchmark differences must come
from the execution mode, not from comparing different source programs.

## Requirements

```
pip install numpy einops
# PyTorch >= 2.x with Triton bundled
```

## Provenance

The original five workload definitions are derived from the
[LEGO](https://github.com/mmt-at/LEGO) artifact of the paper *"LEGO: An
LLM-Enabled Hierarchical Optimizer for Tensor Computation Graphs"* (Table 1).
Only their reference workload implementations are retained here; LEGO's
optimized `model_new.py` artifacts are not part of this suite. See the upstream
repository for license and citation details.

DIN, DIEN, HSTU and OneRec are adapted from
[APPRENTICE](https://github.com/ShouyangDong/APPRENTICE) revision
`c7f96ceec6feac3514550dcac4e6066f1b3d4a41`, using the same deterministic
integration shapes as its recommendation-training examples. DIN and DIEN are
compact APPRENTICE reference implementations. HSTU and OneRec retain the
Apache-2.0 NVIDIA model headers and semantics preserved by APPRENTICE from
`recsys-examples` revision `c6e2bb2`. WorkloadSuite benchmarks their inference
forward paths; it does not claim the full APPRENTICE training/admission contract.
OneRec disables PyTorch's optional eval-mode NestedTensor encoder fast path
because the dense-equivalent path is required for valid all-padding histories
on MLU. This is a portability fix in the canonical workload.

## Benchmark

`bench.py` times two execution modes of the same model instance and writes a
Markdown summary:

```bash
python bench.py --device cuda --bench-runs 30 --warmup 5
# writes results/benchmark.md
```

Variants measured per workload:

- `eager` — `model_ref.py` executed directly
- `compile` — the same `model_ref.py` model wrapped with `torch.compile`

## Layout

```
bench.py
workloads/
  resnet/
    model_ref.py
  qwen/
    model_ref.py
  din/
    model_ref.py
  dien/
    model_ref.py
  hstu/
    model_ref.py
  onerec/
    model_ref.py
```
