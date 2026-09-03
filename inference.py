"""Compatibility entry point for packaged JAX inference."""

from drifting_jax.inference import *  # noqa: F403
from drifting_jax.inference import main as _main


if __name__ == "__main__":
    _main()
