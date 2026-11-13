from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ("swin", "bert", "sd_unet", "conformer", "graphsage", "fno2d")


def _load(name: str) -> Any:
    path = ROOT / "workloads" / name / "model_ref.py"
    spec = importlib.util.spec_from_file_location(f"test_diverse_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case(name: str):
    module = _load(name)
    if name == "swin":
        return module.Model(32, 10, 16, (1, 1), (2, 4), 4), (
            torch.randn(2, 3, 32, 32, dtype=torch.float16),
        )
    if name == "bert":
        model = module.Model(64, 32, 2, 4, 64, 16, 3)
        ids = torch.randint(0, 64, (2, 16))
        return model, (ids, torch.ones_like(ids, dtype=torch.bool), torch.zeros_like(ids))
    if name == "sd_unet":
        return module.Model(32, 16), (
            torch.randn(1, 4, 16, 16, dtype=torch.float16),
            torch.tensor([10.0]),
            torch.randn(1, 5, 16, dtype=torch.float16),
        )
    if name == "conformer":
        return module.Model(16, 32, 2, 4, 64, 7, 8), (
            torch.randn(2, 32, 16, dtype=torch.float16),
            torch.tensor([32, 25]),
        )
    if name == "graphsage":
        source = torch.arange(32, dtype=torch.int64) % 16
        destination = (source * 5 + 1) % 16
        return module.Model(8, 16, 4, 2), (
            torch.randn(16, 8, dtype=torch.float16),
            torch.stack((source, destination)),
        )
    if name == "fno2d":
        return module.Model(3, 1, 8, 4, 4, 2), (torch.randn(2, 3, 16, 16),)
    raise AssertionError(name)


class DiverseWorkloadTests(unittest.TestCase):
    def test_contract_and_reference_forward(self) -> None:
        for name in WORKLOADS:
            with self.subTest(workload=name):
                module = _load(name)
                self.assertTrue(callable(module.get_inputs))
                self.assertTrue(callable(module.get_init_inputs))
                model, inputs = _case(name)
                model.eval()
                with torch.inference_mode():
                    output = model(*inputs)
                self.assertIsInstance(output, torch.Tensor)
                self.assertTrue(torch.isfinite(output).all())

    def test_small_cases_support_torch_compile_contract(self) -> None:
        for name in WORKLOADS:
            with self.subTest(workload=name):
                model, inputs = _case(name)
                model.eval()
                with torch.inference_mode():
                    expected = model(*inputs)
                    actual = torch.compile(model, backend="eager", fullgraph=False)(*inputs)
                torch.testing.assert_close(actual, expected)
                torch._dynamo.reset()

    def test_readme_contains_architecture_links(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for link in (
            "https://github.com/microsoft/Swin-Transformer",
            "https://github.com/google-research/bert",
            "https://github.com/CompVis/stable-diffusion",
            "https://arxiv.org/abs/2005.08100",
            "https://snap.stanford.edu/graphsage/",
            "https://github.com/neuraloperator/neuraloperator",
        ):
            self.assertIn(link, readme)

    def test_benchmark_default_catalog_includes_diversity_workloads(self) -> None:
        spec = importlib.util.spec_from_file_location("test_diverse_bench", ROOT / "bench.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        bench = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bench
        spec.loader.exec_module(bench)
        for workload in WORKLOADS:
            self.assertIn(workload, bench.DEFAULT_WORKLOADS)


if __name__ == "__main__":
    unittest.main()
