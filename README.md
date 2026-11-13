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
| `swin`      | Vision transformer | [Swin-Tiny](https://github.com/microsoft/Swin-Transformer) | `model_ref.py` |
| `bert`      | Encoder transformer | [BERT-Base](https://github.com/google-research/bert) | `model_ref.py` |
| `sd_unet`   | Convolutional diffusion model | [Stable Diffusion U-Net](https://github.com/CompVis/stable-diffusion) | `model_ref.py` |
| `conformer` | Speech encoder | [Conformer-Small](https://arxiv.org/abs/2005.08100) | `model_ref.py` |
| `graphsage` | Graph neural network | [GraphSAGE](https://snap.stanford.edu/graphsage/) | `model_ref.py` |
| `fno2d`     | Scientific neural operator | [FNO-2D](https://github.com/neuraloperator/neuraloperator) | `model_ref.py` |

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

The additional diversity workloads are clean-room, compact PyTorch
implementations of their published architectures rather than copied upstream
source. Their architecture references are:

- Swin Transformer: [paper](https://arxiv.org/abs/2103.14030) and
  [official repository](https://github.com/microsoft/Swin-Transformer).
- BERT: [paper](https://arxiv.org/abs/1810.04805) and
  [original repository](https://github.com/google-research/bert).
- Latent Diffusion / Stable Diffusion U-Net:
  [paper](https://arxiv.org/abs/2112.10752) and
  [original repository](https://github.com/CompVis/stable-diffusion).
- Conformer: [paper](https://arxiv.org/abs/2005.08100) and
  [NVIDIA NeMo model documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html).
- GraphSAGE: [paper](https://arxiv.org/abs/1706.02216),
  [project page](https://snap.stanford.edu/graphsage/), and
  [original repository](https://github.com/williamleif/GraphSAGE).
- Fourier Neural Operator: [paper](https://arxiv.org/abs/2010.08895),
  [official repository](https://github.com/neuraloperator/neuraloperator), and
  [documentation](https://neuraloperator.github.io/dev/).

On the current torch-MLU stack, FNO's spectral FFT submodule is an explicit
`torch.compile` graph boundary because the backend fake/meta FFT stride does not
match its runtime MLU stride. The FFT still executes on MLU without CPU
fallback; lift, pointwise residual and projection regions remain compilable.

### Diversity workload scale

| Workload | Default representative input | Parameters | Distinct operator coverage |
|----------|------------------------------|-----------:|----------------------------|
| `swin` | `2 x 3 x 224 x 224` FP16 images | 28.3M | shifted windows, relative bias, patch merging, layout transforms |
| `bert` | batch 4, sequence 512 | 109.5M | bidirectional masked attention, GELU FFN, token/type/position embedding |
| `sd_unet` | latent `2 x 4 x 64 x 64`, context `2 x 77 x 768` | 19.9M | GroupNorm, multi-scale convolution, resize, skip concat, cross attention |
| `conformer` | speech features `4 x 1024 x 80` | 20.4M | subsampling, attention, GLU, depthwise convolution, length masks |
| `graphsage` | 32768 nodes, 262144 edges, feature 128 | 0.26M | gather, `index_add_`, `bincount`, irregular degree normalization |
| `fno2d` | `4 x 3 x 64 x 64` FP32 fields | 16.8M | rFFT/iFFT, explicit complex multiply, spectral truncation |

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
  swin/
    model_ref.py
  bert/
    model_ref.py
  sd_unet/
    model_ref.py
  conformer/
    model_ref.py
  graphsage/
    model_ref.py
  fno2d/
    model_ref.py
```
