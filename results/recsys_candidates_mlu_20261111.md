# Recommendation candidate admission on MLU (2026-11-11)

The previous four `model_new.py` files only re-exported their reference models,
so the suite could not identify an implementation candidate. In addition, the
dynamic loader did not register modules for Dynamo replay, and OneRec's
reference enabled an optional NestedTensor path that fails for valid
all-padding histories on MLU.

This revision adds explicit, state-dict-compatible candidates and reports a
measured rejection separately from `no candidate`.

| workload | candidate | result | baseline | candidate | aggregate | paired median |
|---|---|---|---:|---:|---:|---:|
| DIN | library weighted reduction + static positions | accepted | 132.530 us | 119.558 us | +9.79% | +9.29% |
| DIEN | target-invariant projection + library weighted reduction | rejected | 1386.063 us | 1369.520 us | +1.19% | +0.07% |
| HSTU | static position and causal masks | accepted | 265.112 us | 249.269 us | +5.98% | +5.98% |
| OneRec | graph-safe padding mask + static indices/mask | accepted | 934.568 us | 918.737 us | +1.69% | +1.74% |

The run used physical MLU580-X6 card 2, identical weights, 20 warmups, 500
whole-model calls per sample, seven alternating pairs, and synchronized wall
time. Both aggregate median improvement and the median of paired improvements
had to reach 1%. Representative and held-out outputs passed `rtol=atol=1e-5`;
held-out inputs include zero lengths for DIN/DIEN, fresh HSTU IDs, and an
all-padding zero-length OneRec history.

DIEN now has a concrete, executable and precision-valid candidate. It remains
rejected because the gain is not stable across pairs; it must not be reported
as accepted and must not be collapsed back into `no candidate`.

Reproduce with:

```bash
MLU_VISIBLE_DEVICES=2 ./bench_candidates.py --device mlu \
  --warmup 20 --iterations 500 --repeats 7 \
  --min-improvement-pct 1.0 --output /tmp/recsys-candidates.json
```
