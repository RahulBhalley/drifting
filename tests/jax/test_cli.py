from __future__ import annotations

from importlib import metadata

from drifting_jax import cli


def _project_scripts() -> set[str]:
    return {
        entry.name
        for entry in metadata.entry_points(group="console_scripts")
        if entry.dist is not None and entry.dist.name == "drifting-models"
    }


def test_installed_commands_name_jax_backend_explicitly() -> None:
    """Catches a generic CLI silently choosing JAX as the default backend."""
    scripts = _project_scripts()

    assert {"drifting-jax-train", "drifting-jax-infer", "drifting-jax-cache"} <= scripts
    assert "drifting-train" not in scripts
    assert "drifting-infer" not in scripts


def test_cli_help_does_not_initialize_jax_runtime(capsys) -> None:
    """Catches eager runtime imports during argument discovery."""
    for entrypoint in (cli.train_main, cli.infer_main, cli.cache_main):
        try:
            entrypoint(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
    output = capsys.readouterr().out
    assert "--help" in output
