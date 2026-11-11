#!/usr/bin/env python3
"""Run paired correctness and whole-model admission for declared candidates."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

ROOT = Path(__file__).resolve().parent
DEFAULT_WORKLOADS = ("din", "dien", "hstu", "onerec")


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _workloads(
    name: str, module: Any, device: torch.device
) -> list[tuple[str, tuple[Any, ...]]]:
    torch.manual_seed(20261111)
    representative = tuple(_to_device(item, device) for item in module.get_inputs())
    rows = [("representative", representative)]
    if name in {"din", "dien"}:
        varied = list(representative)
        varied[-1] = torch.arange(representative[0].shape[0], device=device) % (
            representative[0].shape[1] + 1
        )
        rows.append(("varying_lengths_including_zero", tuple(varied)))
    elif name == "hstu":
        rows.append(
            (
                "fresh_ids",
                (
                    torch.randint(1, 256, representative[0].shape, device=device),
                    torch.randint(1, 32, representative[1].shape, device=device),
                ),
            )
        )
    elif name == "onerec":
        all_padding = [item.clone() for item in representative]
        all_padding[2].zero_()
        all_padding[3].zero_()
        rows.append(("all_padding_zero_length_history", tuple(all_padding)))
    return rows


def _precision(
    reference: Any, candidate: Any, workloads: list[tuple[str, tuple[Any, ...]]]
):
    reports = []
    with torch.inference_mode():
        for workload_id, inputs in workloads:
            expected, expected_spec = tree_flatten(reference(*inputs))
            actual, actual_spec = tree_flatten(candidate(*inputs))
            if actual_spec != expected_spec or len(actual) != len(expected):
                raise ValueError(f"output structure changed for {workload_id}")
            max_abs = 0.0
            for actual_item, expected_item in zip(actual, expected, strict=True):
                if not isinstance(actual_item, torch.Tensor) or not isinstance(
                    expected_item, torch.Tensor
                ):
                    if actual_item != expected_item:
                        raise ValueError(f"non-tensor output changed for {workload_id}")
                    continue
                torch.testing.assert_close(
                    actual_item, expected_item, rtol=1e-5, atol=1e-5, equal_nan=True
                )
                if actual_item.numel():
                    difference = (
                        actual_item.ne(expected_item)
                        if actual_item.dtype == torch.bool
                        else (actual_item - expected_item).abs()
                    )
                    max_abs = max(max_abs, float(difference.max().cpu()))
            reports.append(
                {"workload_id": workload_id, "passed": True, "max_abs": max_abs}
            )
    return reports


def _synchronize(device: torch.device) -> None:
    if device.type == "mlu":
        torch.mlu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    module: Any,
    inputs: tuple[Any, ...],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            module(*inputs)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            module(*inputs)
        _synchronize(device)
    return (time.perf_counter() - started) * 1e6 / iterations


def _evaluate(
    name: str, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    directory = ROOT / "workloads" / name
    reference_module = _load(directory / "model_ref.py", f"candidate_bench_{name}_ref")
    candidate_module = _load(directory / "model_new.py", f"candidate_bench_{name}_new")
    metadata = candidate_module.get_optimization_metadata()
    torch.manual_seed(args.seed)
    reference = (
        reference_module.Model(*reference_module.get_init_inputs()).eval().to(device)
    )
    candidate = (
        candidate_module.Model(*candidate_module.get_init_inputs()).eval().to(device)
    )
    if set(reference.state_dict()) != set(candidate.state_dict()):
        raise ValueError("candidate state_dict contract changed")
    candidate.load_state_dict(reference.state_dict(), strict=True)
    execution_mode = metadata["execution_mode"]
    executable = candidate
    if execution_mode == "compile":
        executable = torch.compile(
            candidate, mode=args.compile_mode, dynamic=False, fullgraph=False
        )
    workloads = _workloads(name, reference_module, device)
    precision = _precision(reference, executable, workloads)
    representative = workloads[0][1]
    baseline_samples = []
    candidate_samples = []
    for repeat in range(args.repeats):
        if repeat % 2:
            candidate_us = _measure(
                executable,
                representative,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            baseline_us = _measure(
                reference,
                representative,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        else:
            baseline_us = _measure(
                reference,
                representative,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            candidate_us = _measure(
                executable,
                representative,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        baseline_samples.append(baseline_us)
        candidate_samples.append(candidate_us)
    baseline_median = statistics.median(baseline_samples)
    candidate_median = statistics.median(candidate_samples)
    improvement = (baseline_median - candidate_median) / baseline_median * 100.0
    paired_improvements = [
        (baseline_us - candidate_us) / baseline_us * 100.0
        for baseline_us, candidate_us in zip(
            baseline_samples, candidate_samples, strict=True
        )
    ]
    paired_improvement = statistics.median(paired_improvements)
    passed_performance = (
        min(improvement, paired_improvement) >= args.min_improvement_pct
    )
    return {
        "workload": name,
        "candidate": metadata,
        "status": "accepted" if passed_performance else "rejected",
        "reason": (
            "precision and performance gates passed"
            if passed_performance
            else (
                "performance gate failed: "
                f"aggregate={improvement:.3f}%, paired_median={paired_improvement:.3f}%"
            )
        ),
        "precision": precision,
        "timing": {
            "method": "synchronized_wall_clock_whole_model",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "alternating_order": True,
            "baseline_samples_us": baseline_samples,
            "candidate_samples_us": candidate_samples,
            "baseline_median_us": baseline_median,
            "candidate_median_us": candidate_median,
            "improvement_pct": improvement,
            "paired_improvement_samples_pct": paired_improvements,
            "paired_improvement_median_pct": paired_improvement,
            "required_improvement_pct": args.min_improvement_pct,
            "gate": "min(aggregate_median_improvement, paired_improvement_median)",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-improvement-pct", type=float, default=1.0)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations and repeats must be positive")
    if not math.isfinite(args.min_improvement_pct) or args.min_improvement_pct < 0:
        parser.error("min improvement must be finite and nonnegative")
    names = tuple(name.strip() for name in args.workloads.split(",") if name.strip())
    if not names or set(names) - set(DEFAULT_WORKLOADS):
        parser.error("candidate workloads must be din,dien,hstu,onerec")
    device = torch.device(args.device)
    if device.type == "mlu":
        import torch_mlu  # noqa: F401

    report = {
        "schema_version": "workloadsuite_candidate_admission_v1",
        "device": str(device),
        "mlu_visible_devices": os.environ.get("MLU_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            result = _evaluate(name, args, device)
        except Exception as error:  # noqa: BLE001 - preserve rejection evidence
            result = {
                "workload": name,
                "status": "rejected",
                "reason": f"{type(error).__name__}: {error}",
            }
        report["results"].append(result)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        torch._dynamo.reset()
        gc.collect()
        if device.type == "mlu":
            torch.mlu.empty_cache()
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
