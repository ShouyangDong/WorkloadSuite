# WorkloadSuite

Benchmark model workloads for InferArch: each workload ships a reference
(PyTorch eager) implementation and an optimized (Triton) implementation of a
representative inference model, used to benchmark and validate the optimizer.

## Workloads

| Workload    | Type               | Model        | `model_ref.py`       | `model_new.py`       |
|-------------|--------------------|--------------|----------------------|----------------------|
| `resnet`    | CNN                | ResNet-18    | eager baseline       | torch-only optimised |
| `qwen`      | LLM decoder        | Qwen3-8B     | eager baseline       | Triton kernels       |
| `sd3_mmdit` | Diffusion transformer | SD3-MMDiT | eager baseline       | Triton-free `einops` |
| `mamba`     | State space model  | Mamba-2 SSD  | eager baseline       | Triton + `einops`    |
| `ds_mhc_moe`| Mixture of experts | mHC-MoE      | eager baseline       | Triton kernels       |

Each workload is a self-contained pair:

- `model_ref.py` — PyTorch eager baseline
- `model_new.py` — optimized variant (Triton kernels and/or system-level opts)

## Requirements

```
pip install numpy einops
# PyTorch >= 2.x with Triton bundled
```

## Provenance

These five workloads are reproduced from the
[LEGO](https://github.com/mmt-at/LEGO) artifact of the paper *"LEGO: An
LLM-Enabled Hierarchical Optimizer for Tensor Computation Graphs"* (Table 1).
See the upstream repository for license and citation details.

## Layout

```
workloads/
  resnet/
    model_ref.py
    model_new.py
  qwen/
    ...
```