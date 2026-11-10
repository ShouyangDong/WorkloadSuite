# WorkloadSuite

Benchmark model workloads for MASTER: each workload ships a reference
(PyTorch eager) implementation and an optimization-target implementation of a
representative inference model, used to benchmark and validate the optimizer.

## Workloads

| Workload    | Type               | Model        | `model_ref.py`       | `model_new.py`       |
|-------------|--------------------|--------------|----------------------|----------------------|
| `resnet`    | CNN                | ResNet-18    | eager baseline       | torch-only optimised |
| `qwen`      | LLM decoder        | Qwen3-8B     | eager baseline       | Triton kernels       |
| `sd3_mmdit` | Diffusion transformer | SD3-MMDiT | eager baseline       | Triton-free `einops` |
| `mamba`     | State space model  | Mamba-2 SSD  | eager baseline       | Triton + `einops`    |
| `ds_mhc_moe`| Mixture of experts | mHC-MoE      | eager baseline       | Triton kernels       |
| `din`       | Recommender        | DIN          | APPRENTICE reference | source-equivalent    |
| `dien`      | Recommender        | DIEN         | APPRENTICE reference | source-equivalent    |
| `hstu`      | Sequential recommender | HSTU     | preserved reference  | source-equivalent    |
| `onerec`    | Generative recommender | OneRec   | preserved reference  | source-equivalent    |

Each workload is a self-contained pair:

- `model_ref.py` — PyTorch eager baseline
- `model_new.py` — optimization target; it may contain handwritten kernels and
  system-level optimizations, or a source-equivalent entry point for workloads
  newly added for compiler optimization

## Requirements

```
pip install numpy einops
# PyTorch >= 2.x with Triton bundled
```

## Provenance

The original five workloads are reproduced from the
[LEGO](https://github.com/mmt-at/LEGO) artifact of the paper *"LEGO: An
LLM-Enabled Hierarchical Optimizer for Tensor Computation Graphs"* (Table 1).
See the upstream repository for license and citation details.

DIN, DIEN, HSTU and OneRec are adapted from
[APPRENTICE](https://github.com/ShouyangDong/APPRENTICE) revision
`c7f96ceec6feac3514550dcac4e6066f1b3d4a41`, using the same deterministic
integration shapes as its recommendation-training examples. DIN and DIEN are
compact APPRENTICE reference implementations. HSTU and OneRec retain the
Apache-2.0 NVIDIA model headers and semantics preserved by APPRENTICE from
`recsys-examples` revision `c6e2bb2`. WorkloadSuite benchmarks their inference
forward paths; it does not claim the full APPRENTICE training/admission contract.

## Benchmark

`bench.py` times three variants per workload and writes a Markdown summary:

```bash
python bench.py --device cuda --bench-runs 30 --warmup 5
# writes results/benchmark.md
```

Variants measured per workload:

- `ref` — `model_ref.py`, eager
- `eager` — `model_new.py`, eager (no `torch.compile`)
- `compile` — `model_new.py`, wrapped with `torch.compile`

## Layout

```
bench.py
workloads/
  resnet/
    model_ref.py
    model_new.py
  qwen/
    ...
  din/
    model_ref.py
    model_new.py
  dien/
    ...
  hstu/
    ...
  onerec/
    ...
```
