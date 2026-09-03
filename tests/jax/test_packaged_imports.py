from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_jax_backend_imports_outside_repository(tmp_path: Path) -> None:
    """Catches packaged modules that still depend on checkout-root imports."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OMP_NUM_THREADS"] = "1"
    env["KMP_USE_SHM"] = "0"
    modules = [
        "drifting_jax.checkpointing.initialize",
        "drifting_jax.data.latent",
        "drifting_jax.models.generator",
        "drifting_jax.models.mae",
        "drifting_jax.models.convnext",
        "drifting_jax.data.datasets",
        "drifting_jax.training.generator",
        "drifting_jax.training.mae",
        "drifting_jax.evaluation.evaluator",
        "drifting_jax.checkpointing.checkpoint",
        "drifting_jax.distributed",
        "drifting_jax.inference",
    ]
    code = (
        "import importlib, json; "
        f"names={modules!r}; "
        "print(json.dumps([importlib.import_module(name).__name__ for name in names]))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == modules


def test_legacy_generator_import_resolves_to_packaged_class() -> None:
    """Catches a compatibility shim diverging from the packaged implementation."""
    from drifting_jax.models.generator import DitGen
    from models.generator import DitGen as LegacyDitGen

    assert LegacyDitGen is DitGen
