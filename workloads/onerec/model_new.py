"""Source-equivalent OneRec entry point for eager and ``torch.compile`` benchmarking."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE = Path(__file__).with_name("model_ref.py")
_SPEC = importlib.util.spec_from_file_location("workloadsuite_onerec_reference", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load OneRec reference workload: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

Model = _MODULE.Model
get_inputs = _MODULE.get_inputs
get_init_inputs = _MODULE.get_init_inputs
