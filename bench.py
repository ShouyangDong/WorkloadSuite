#!/usr/bin/env python3
"""Benchmark eager / torch.compile / ref across the WorkloadSuite workloads.

For each workload under ``workloads/``, load the reference (``model_ref.py``)
and the optimized (``model_new.py``) implementations and time three variants:

* ``ref``     — ``model_ref.py``, eager
* ``eager``   — ``model_new.py``, eager (no ``torch.compile``)
* ``compile`` — ``model_new.py``, wrapped with ``torch.compile``

Results are printed to stdout and written to a Markdown summary (default
``results/benchmark.md``).

The harness is self-contained: it uses the ``Model`` / ``get_inputs`` /
``get_init_inputs`` convention shared by every workload file and does not
depend on any external LEGO tooling.

Example::

    python bench.py --device cuda --bench-runs 30 --warmup 5
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import random
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DEFAULT_WORKLOADS = ["resnet", "qwen", "sd3_mmdit", "mamba", "ds_mhc_moe"]


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _set_f32_config() -> None:
    """Disable TF32 so timing matches the LEGO reference setup."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def _sync(device_type: str) -> None:
    """Synchronize the given device if it exposes a synchronize() callable."""
    module = getattr(torch, device_type, None)
    sync = getattr(module, "synchronize", None)
    if callable(sync):
        sync()


def _event_class(device_type: str) -> type[Any] | None:
    """Return the timing Event class for a device (e.g. torch.cuda.Event)."""
    module = getattr(torch, device_type, None)
    event_cls = getattr(module, "Event", None)
    return event_cls if event_cls is not None else None


# ---------------------------------------------------------------------------
# Module loading (isolated, so ref/new files with identical basenames coexist)
# ---------------------------------------------------------------------------


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _summarize(samples: list[float]) -> dict[str, float]:
    samples_sorted = sorted(samples)

    def quantile(q: float) -> float:
        idx = min(len(samples_sorted) - 1, max(0, round(q * (len(samples_sorted) - 1))))
        return samples_sorted[idx]

    return {
        "median_ms": float(statistics.median(samples_sorted)),
        "mean_ms": float(statistics.mean(samples_sorted)),
        "p90_ms": float(quantile(0.90)),
        "p99_ms": float(quantile(0.99)),
    }


def _benchmark(
    model: torch.nn.Module,
    inputs: list[Any],
    *,
    device_type: str,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    """Run ``model(*inputs)`` and return per-call latency statistics (ms)."""
    for _ in range(warmup):
        model(*inputs)

    event_cls = _event_class(device_type)
    if event_cls is not None:
        _sync(device_type)
        starts = [event_cls(enable_timing=True) for _ in range(runs)]
        ends = [event_cls(enable_timing=True) for _ in range(runs)]
        for start, end in zip(starts, ends):
            start.record()
            model(*inputs)
            end.record()
        _sync(device_type)
        samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    else:
        samples = []
        for _ in range(runs):
            _sync(device_type)
            t0 = time.perf_counter()
            model(*inputs)
            _sync(device_type)
            samples.append((time.perf_counter() - t0) * 1000.0)

    return _summarize(samples)


def _compile_and_bench(
    model: torch.nn.Module,
    inputs: list[Any],
    *,
    device_type: str,
    warmup: int,
    runs: int,
    compile_mode: str,
    dynamic: bool,
) -> dict[str, float | str]:
    """Trace with torch.compile and benchmark; record the error if it breaks."""
    try:
        compiled = torch.compile(model, mode=compile_mode, dynamic=dynamic, fullgraph=False)
        return _benchmark(
            compiled,
            inputs,
            device_type=device_type,
            warmup=warmup,
            runs=runs,
        )
    except Exception as exc:  # noqa: BLE001 - report rather than crash the run
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            torch._dynamo.reset()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-workload driver
# ---------------------------------------------------------------------------


def _bench_workload(name: str, *, device_type: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    workload_dir = ROOT / "workloads" / name
    ref_path = workload_dir / "model_ref.py"
    new_path = workload_dir / "model_new.py"

    rows: list[dict[str, Any]] = []

    def add_row(variant: str, stats: dict[str, float | str]) -> None:
        rows.append({"workload": name, "variant": variant, **stats})

    try:
        ref_mod = _load_module(ref_path, f"ws_{name}_ref")
        new_mod = _load_module(new_path, f"ws_{name}_new")
        init_inputs = ref_mod.get_init_inputs()
        inputs = ref_mod.get_inputs()
    except Exception as exc:  # noqa: BLE001
        print(f"!! load failed for {name}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        add_row("load", {"error": f"{type(exc).__name__}: {exc}"})
        return rows

    common = dict(device_type=device_type, warmup=args.warmup, runs=args.bench_runs)

    # ref (reference implementation, eager)
    try:
        model = ref_mod.Model(*init_inputs)
        model.eval()
        with torch.inference_mode():
            add_row("ref", _benchmark(model, inputs, **common))
    except Exception as exc:  # noqa: BLE001
        print(f"!! ref failed for {name}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        add_row("ref", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            del model
        except NameError:
            pass
        gc.collect()

    # eager (optimized implementation, no compile)
    model = None
    try:
        model = new_mod.Model(*init_inputs)
        model.eval()
        with torch.inference_mode():
            add_row("eager", _benchmark(model, inputs, **common))
    except Exception as exc:  # noqa: BLE001
        print(f"!! eager failed for {name}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        add_row("eager", {"error": f"{type(exc).__name__}: {exc}"})

    # compile (optimized implementation under torch.compile)
    if model is not None:
        try:
            with torch.inference_mode():
                stats = _compile_and_bench(
                    model,
                    inputs,
                    device_type=device_type,
                    warmup=args.warmup,
                    runs=args.bench_runs,
                    compile_mode=args.compile_mode,
                    dynamic=args.dynamic,
                )
            add_row("compile", stats)
        except Exception as exc:  # noqa: BLE001
            print(f"!! compile failed for {name}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            add_row("compile", {"error": f"{type(exc).__name__}: {exc}"})
    else:
        add_row("compile", {"error": "skipped (eager model not available)"})

    try:
        if model is not None:
            del model
    except NameError:
        pass
    gc.collect()

    return rows


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def _fmt(value: float | str | None, nd: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:.{nd}f}"


def _write_markdown(
    rows: list[dict[str, Any]],
    workloads: list[str],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_workload: dict[str, dict[str, dict[str, float | str]]] = {w: {} for w in workloads}
    for row in rows:
        by_workload[row["workload"]][row["variant"]] = {
            k: v for k, v in row.items() if k not in {"workload", "variant"}
        }

    lines: list[str] = []
    lines.append("# WorkloadSuite benchmark\n")
    lines.append(f"- device: `{args.device}`")
    lines.append(f"- bench runs: {args.bench_runs}")
    lines.append(f"- warmup: {args.warmup}")
    lines.append(f"- compile mode: `{args.compile_mode}`{' (dynamic)' if args.dynamic else ''}")
    lines.append(f"- generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Latency (ms)\n")
    lines.append("| workload | variant | median | mean | p90 | p99 |")
    lines.append("|---|---|---|---|---|---|")

    for workload in workloads:
        variants = by_workload[workload]
        if not variants:
            lines.append(f"| {workload} | — | FAILED (no data) | — | — | — |")
            continue
        # If the workload failed to load, report the error explicitly
        if "load" in variants:
            err = variants["load"].get("error", "unknown load error")
            lines.append(f"| {workload} | load | {err} | — | — | — |")
            continue

        for variant in ("ref", "eager", "compile"):
            stats = variants.get(variant, {})
            if not stats:
                lines.append(f"| {workload} | {variant} | missing | — | — | — |")
                continue
            if "error" in stats:
                lines.append(
                    f"| {workload} | {variant} | {stats['error']} | — | — | — |"
                )
                continue
            lines.append(
                f"| {workload} | {variant} "
                f"| {_fmt(stats['median_ms'])} | {_fmt(stats['mean_ms'])} "
                f"| {_fmt(stats['p90_ms'])} | {_fmt(stats['p99_ms'])} |"
            )

    lines.append("\n## Summary vs reference\n")
    lines.append("| workload | ref (ms) | eager (ms) | compile (ms) | eager vs ref | compile vs ref |")
    lines.append("|---|---|---|---|---|---|")

    for workload in workloads:
        variants = by_workload[workload]
        if not variants or "load" in variants:
            lines.append(f"| {workload} | — | — | — | FAILED | FAILED |")
            continue

        ref = variants.get("ref", {})
        eager = variants.get("eager", {})
        compiled = variants.get("compile", {})

        ref_ms = ref.get("median_ms")
        eager_ms = eager.get("median_ms")
        compile_ms = compiled.get("median_ms")

        def speedup(candidate: float | None, baseline: float | None) -> str:
            if candidate is None or baseline is None or not baseline:
                return "—"
            return f"{baseline / candidate:.2f}x" if candidate else "—"

        lines.append(
            f"| {workload} | {_fmt(ref_ms)} | {_fmt(eager_ms)} | {_fmt(compile_ms)} "
            f"| {speedup(eager_ms, ref_ms)} | {speedup(compile_ms, ref_ms)} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="device to run and time on (default: mlu if available, else cuda)")
    parser.add_argument("--bench-runs", type=int, default=30, help="timed runs per variant")
    parser.add_argument("--warmup", type=int, default=5, help="warmup runs per variant")
    parser.add_argument("--compile-mode", default="default", help="torch.compile mode")
    parser.add_argument("--dynamic", action="store_true", help="enable dynamic shapes in torch.compile")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/benchmark.md", help="output Markdown path")
    parser.add_argument(
        "--workloads",
        default=",".join(DEFAULT_WORKLOADS),
        help="comma-separated workload names",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "mlu" if torch.mlu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    torch.set_default_device(device)
    _set_seed(args.seed)
    _set_f32_config()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    rows: list[dict[str, Any]] = []
    for name in workloads:
        print(f"\n== benchmarking {name} ==")
        rows.extend(_bench_workload(name, device_type=device.type, args=args))

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    _write_markdown(rows, workloads, args, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())