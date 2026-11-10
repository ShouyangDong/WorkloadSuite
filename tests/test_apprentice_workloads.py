from __future__ import annotations

import importlib.util
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
    spec.loader.exec_module(module)
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
    def test_reference_and_optimization_target_are_source_equivalent(self) -> None:
        for workload in WORKLOADS:
            with self.subTest(workload=workload):
                directory = ROOT / "workloads" / workload
                reference = _load(directory / "model_ref.py", f"test_{workload}_reference")
                candidate = _load(directory / "model_new.py", f"test_{workload}_candidate")

                torch.manual_seed(10)
                reference_model = reference.Model(*reference.get_init_inputs()).eval()
                torch.manual_seed(20)
                candidate_model = candidate.Model(*candidate.get_init_inputs()).eval()
                candidate_model.load_state_dict(reference_model.state_dict(), strict=True)

                torch.manual_seed(30)
                inputs = reference.get_inputs()
                with torch.inference_mode():
                    expected = reference_model(*inputs)
                    actual = candidate_model(*inputs)
                _assert_tree_close(self, actual, expected)

    def test_cpu_uses_wall_clock_timing(self) -> None:
        bench = _load(ROOT / "bench.py", "test_workloadsuite_bench")
        self.assertIsNone(bench._event_class("cpu"))


if __name__ == "__main__":
    unittest.main()
