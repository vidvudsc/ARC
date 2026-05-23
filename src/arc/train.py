from __future__ import annotations

import argparse

from .config import load_config
from .data import describe_mixture, read_mixture
from .model import ArcModel, estimate_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arc training entrypoint scaffold")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mixture", default="configs/stage1_text_mix.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    mixture = read_mixture(args.mixture)
    params = estimate_parameters(cfg)
    print(f"model: {cfg.model_name}")
    print(f"total params: {params['total']:,}")
    print(f"text + embeddings params: {params['text_plus_embeddings']:,}")
    print(f"vision params: {params['vision']:,}")
    print("\nmixture:")
    print(describe_mixture(mixture))
    if args.dry_run:
        return
    _ = ArcModel(cfg)
    raise NotImplementedError("Distributed training loop is not implemented yet.")


if __name__ == "__main__":
    main()

