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
| `din`       | Recommender        | DIN          | APPRENTICE reference | library weighted reduction |
| `dien`      | Recommender        | DIEN         | APPRENTICE reference | library weighted reduction |
| `hstu`      | Sequential recommender | HSTU     | preserved reference  | static index/mask hoisting |
| `onerec`    | Generative recommender | OneRec   | preserved reference  | graph-safe static inputs |

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
Their `model_new.py` files expose explicit optimization candidates and
`get_optimization_metadata()`. A declared candidate is not an accepted winner:
the benchmark must still validate equivalent outputs and measure the complete
model on the target device.

OneRec disables PyTorch's optional eval-mode NestedTensor encoder fast path in
both variants because the dense-equivalent path is required for valid
all-padding histories on MLU. This is a portability fix, not a candidate gain.

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

The four recommender candidates also provide a paired admission harness. It
loads identical weights, checks representative and held-out outputs, alternates
baseline/candidate order, synchronizes the target device, and reports rejected
candidates instead of collapsing them into `no candidate`:

```bash
MLU_VISIBLE_DEVICES=0 python bench_candidates.py --device mlu \
  --output results/recsys_candidates.json
```

The validated 2026-11-11 run is recorded in
[`results/recsys_candidates_mlu_20261111.md`](results/recsys_candidates_mlu_20261111.md)
with a machine-readable JSON summary beside it.

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
