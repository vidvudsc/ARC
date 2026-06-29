from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import ByteLevelBPETokenizer


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|image_start|>",
    "<|image_end|>",
]


class ArcTokenizer:
    def __init__(self, tokenizer: ByteLevelBPETokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_dir(cls, path: str | Path) -> "ArcTokenizer":
        path = Path(path)
        tok = ByteLevelBPETokenizer(str(path / "vocab.json"), str(path / "merges.txt"))
        tok.add_special_tokens(SPECIAL_TOKENS)
        return cls(tok)

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Missing token: {token}")
        return int(token_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id("<|pad|>")

    @property
    def bos_id(self) -> int:
        return self.token_to_id("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self.token_to_id("<|eos|>")

    @property
    def image_start_id(self) -> int:
        return self.token_to_id("<|image_start|>")

    @property
    def image_end_id(self) -> int:
        return self.token_to_id("<|image_end|>")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return [int(x) for x in ids]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)


def train_byte_bpe(
    texts: Iterable[str],
    out_dir: str | Path,
    vocab_size: int = 16384,
    min_frequency: int = 2,
    limit: int | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        length=limit,
    )
    tokenizer.save_model(str(out))
