"""Command line entry point for native PyTorch latent-cache construction."""

from __future__ import annotations

import argparse

from .latent import build_latent_cache
from .vae import VAECodec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    codec = VAECodec.from_pretrained(
        cache_dir=args.hf_cache_dir, local_files_only=args.local_files_only
    )
    manifest = build_latent_cache(
        args.data_root,
        args.cache_root,
        codec,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"cached {sum(manifest.split_counts.values())} entries")


if __name__ == "__main__":
    main()
