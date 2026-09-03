"""Compatibility entry point for the packaged JAX latent cache."""

from drifting_jax.data.latent import *  # noqa: F403
from drifting_jax.data.latent import main as _main


if __name__ == "__main__":
    _main()
