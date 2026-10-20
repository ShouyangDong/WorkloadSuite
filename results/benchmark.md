# WorkloadSuite benchmark

- device: `mlu`
- bench runs: 30
- warmup: 5
- compile mode: `default`
- generated: 2026-10-20 19:12:06

## Latency (ms)

| workload | variant | median | mean | p90 | p99 |
|---|---|---|---|---|---|
| resnet | ref | 2.025 | 2.027 | 2.061 | 2.093 |
| resnet | eager | 0.859 | 0.861 | 0.869 | 0.896 |
| resnet | compile | 1.185 | 1.196 | 1.229 | 1.309 |
| qwen | ref | 666.050 | 666.016 | 667.023 | 667.236 |
| qwen | eager | 700.174 | 700.172 | 700.383 | 700.454 |
| qwen | compile | 672.836 | 672.829 | 673.124 | 673.674 |
| sd3_mmdit | ref | 992.682 | 992.667 | 992.887 | 993.303 |
| sd3_mmdit | eager | 496.281 | 496.295 | 496.354 | 497.101 |
| sd3_mmdit | compile | 462.818 | 462.915 | 463.115 | 465.189 |
| mamba | ref | 813.711 | 813.726 | 813.885 | 814.152 |
| mamba | eager | 813.680 | 813.705 | 813.830 | 813.979 |
| mamba | compile | 1072.436 | 1073.593 | 1091.816 | 1094.022 |
| ds_mhc_moe | ref | 218.014 | 138.822 | 256.809 | 271.191 |
| ds_mhc_moe | eager | 15.910 | 15.964 | 16.012 | 17.263 |
| ds_mhc_moe | compile | 22.987 | 23.075 | 23.331 | 24.493 |

## Summary vs reference

| workload | ref (ms) | eager (ms) | compile (ms) | eager vs ref | compile vs ref |
|---|---|---|---|---|---|
| resnet | 2.025 | 0.859 | 1.185 | 2.36x | 1.71x |
| qwen | 666.050 | 700.174 | 672.836 | 0.95x | 0.99x |
| sd3_mmdit | 992.682 | 496.281 | 462.818 | 2.00x | 2.14x |
| mamba | 813.711 | 813.680 | 1072.436 | 1.00x | 0.76x |
| ds_mhc_moe | 218.014 | 15.910 | 22.987 | 13.70x | 9.48x |
