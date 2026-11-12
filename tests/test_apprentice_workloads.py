from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ("din", "dien", "hstu", "onerec")


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


def _assert_tree_close(test: unittest.TestCase, actual: Any, expected: Any) -> None:
    if isinstance(expected, tuple):
        test.assertIsInstance(actual, tuple)
        test.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_close(test, actual_item, expected_item)
        return
    torch.testing.assert_close(actual, expected)


class ApprenticeWorkloadTests(unittest.TestCase):
    def test_reference_models_cover_recommendation_edge_cases(self) -> None:
        for workload in WORKLOADS:
            with self.subTest(workload=workload):
                directory = ROOT / "workloads" / workload
                reference = _load(
                    directory / "model_ref.py", f"test_{workload}_reference"
                )

                torch.manual_seed(10)
                reference_model = reference.Model(*reference.get_init_inputs()).eval()
                if workload == "onerec":
                    self.assertIs(
                        reference_model.history_encoder.enable_nested_tensor, False
                    )
                torch.manual_seed(30)
                inputs = reference.get_inputs()
                cases = [inputs]
                if workload in {"din", "dien"}:
                    varied = [item.clone() for item in inputs]
                    varied[-1] = torch.arange(varied[-1].numel()) % (
                        varied[0].shape[1] + 1
                    )
                    cases.append(varied)
                elif workload == "hstu":
                    cases.append([inputs[0][:3, :17], inputs[1][:3]])
                else:
                    all_padding = [item.clone() for item in inputs]
                    all_padding[2].zero_()
                    all_padding[3].zero_()
                    cases.append(all_padding)
                with torch.inference_mode():
                    for case in cases:
                        expected = reference_model(*case)
                        actual = reference_model(*case)
                        _assert_tree_close(self, actual, expected)

    def test_each_workload_has_only_one_model_source(self) -> None:
        directories = sorted(
            path for path in (ROOT / "workloads").iterdir() if path.is_dir()
        )
        self.assertTrue(directories)
        for directory in directories:
            with self.subTest(workload=directory.name):
                self.assertTrue((directory / "model_ref.py").is_file())
                self.assertFalse((directory / "model_new.py").exists())

    def test_cpu_uses_wall_clock_timing(self) -> None:
        bench = _load(ROOT / "bench.py", "test_workloadsuite_bench")
        self.assertIsNone(bench._event_class("cpu"))

    def test_benchmark_loader_registers_modules_for_compiler_replay(self) -> None:
        bench = _load(ROOT / "bench.py", "test_workloadsuite_bench_loader")
        for workload in ("hstu", "onerec"):
            name = f"test_bench_registered_{workload}"
            module = bench._load_module(
                ROOT / "workloads" / workload / "model_ref.py", name
            )
            self.assertIs(sys.modules[name], module)
            model = module.Model(*module.get_init_inputs()).eval()
            inputs = module.get_inputs()
            with torch.inference_mode():
                expected = model(*inputs)
                actual = torch.compile(model, backend="eager", fullgraph=False)(*inputs)
            _assert_tree_close(self, actual, expected)


if __name__ == "__main__":
    unittest.main()
