#!/usr/bin/env python3
from __future__ import annotations

import argparse

from arc.throughput import hours_for_tokens, recommend_token_budget


def main() -> None:
    parser = argparse.ArgumentParser(description="Arc v0.3 throughput gate calculator")
    parser.add_argument("tokens_per_second", type=float)
    parser.add_argument("--target_tokens", type=int, default=30_000_000_000)
    args = parser.parse_args()
    print(f"throughput: {args.tokens_per_second:,.0f} tok/s")
    print(f"recommendation: {recommend_token_budget(args.tokens_per_second)}")
    print(f"{args.target_tokens:,} tokens would take {hours_for_tokens(args.target_tokens, args.tokens_per_second):.2f} hours")


if __name__ == "__main__":
    main()

