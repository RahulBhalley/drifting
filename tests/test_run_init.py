from __future__ import annotations

import utils.misc as misc


def test_run_init_skips_distributed_initialization_for_explicit_cpu(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(misc, "_did_run_init", False)

    def unexpected_initialize() -> None:
        raise AssertionError("single-process CPU must not initialize distributed JAX")

    monkeypatch.setattr(misc.jax.distributed, "initialize", unexpected_initialize)

    misc.run_init()

    assert misc._did_run_init is True


def test_run_init_preserves_distributed_initialization_for_tpu(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "tpu,cpu")
    monkeypatch.setattr(misc, "_did_run_init", False)
    calls = []
    monkeypatch.setattr(misc.jax.distributed, "initialize", lambda: calls.append(True))

    misc.run_init()

    assert calls == [True]
    assert misc._did_run_init is True
