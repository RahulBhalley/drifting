"""Compatibility entry point for packaged JAX generator training."""

import argparse

from drifting_jax.training.generator import *  # noqa: F403
from drifting_jax.training.generator import _create_memory_banks, _should_evaluate, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", default="runs")
    args = parser.parse_args()
    main(args)
