from __future__ import annotations


def recommend_token_budget(tokens_per_second: float) -> str:
    if tokens_per_second < 50_000:
        return "debug only; train <=100M tokens"
    if tokens_per_second < 150_000:
        return "train 100M-300M tokens"
    if tokens_per_second < 300_000:
        return "train 300M-700M tokens"
    return "train full 1B Arc-Mini run"


def hours_for_tokens(tokens: int, tokens_per_second: float) -> float:
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    return tokens / tokens_per_second / 3600
