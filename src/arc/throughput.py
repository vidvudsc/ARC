from __future__ import annotations


def recommend_token_budget(tokens_per_second: float) -> str:
    if tokens_per_second < 2_000_000:
        return "debug only; do not full-send"
    if tokens_per_second < 2_500_000:
        return "train 15B-20B tokens"
    if tokens_per_second < 3_500_000:
        return "train 20B-25B tokens"
    return "train full 30B tokens"


def hours_for_tokens(tokens: int, tokens_per_second: float) -> float:
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    return tokens / tokens_per_second / 3600

