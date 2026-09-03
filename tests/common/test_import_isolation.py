from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_common_import_does_not_load_tensor_backends() -> None:
    """Catches accidental JAX/PyTorch imports in the common package."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import drifting_common; "
                "print(json.dumps(sorted(name for name in sys.modules "
                "if name == 'jax' or name.startswith('jax.') "
                "or name == 'torch' or name.startswith('torch.'))))"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []

