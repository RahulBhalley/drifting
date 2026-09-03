from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_backend_entrypoints_are_explicit_and_complete():
    scripts = project()["project"]["scripts"]
    assert set(scripts) == {
        "drifting-jax-train",
        "drifting-jax-infer",
        "drifting-jax-cache",
        "drifting-torch-train",
        "drifting-torch-train-mae",
        "drifting-torch-infer",
        "drifting-torch-cache",
    }
    assert not any(name in scripts for name in ("drifting-train", "drifting-infer"))


def test_backend_extras_and_shared_config_payload_are_declared():
    metadata = project()
    extras = metadata["project"]["optional-dependencies"]
    assert {"jax", "torch", "parity", "dev"} <= set(extras)
    data_files = metadata["tool"]["setuptools"]["data-files"]
    assert any("configs/gen" in target for target in data_files)
    assert any("configs/runtime/torch" in target for target in data_files)
