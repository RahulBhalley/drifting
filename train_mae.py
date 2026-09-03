"""Compatibility entry point for packaged JAX MAE training."""

import argparse

from drifting_jax.training.mae import *  # noqa: F403
from drifting_jax.training.mae import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", default="runs")
    args = parser.parse_args()
    main(args)
