from __future__ import annotations
from dataclasses import dataclass
from unicodedata import normalize

import regex

__all__ = [
    'Token',
    'tokenize',
    'normalize_token',
]

PATTERN = regex.compile(r'[\p{L}\p{M}\p{N}]+')


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    normalized: str

    def __eq__(self, other: Token) -> bool:
        return other.normalized == self.normalized

    def __hash__(self) -> int:
        return hash(self.normalized)


def normalize_token(token: str) -> str:
    return normalize('NFC', token.lower())


def tokenize(text: str) -> list[Token]:
    return [
        Token(
            start=match.start(),
            end=match.end(),
            normalized=normalize_token(match.group()),
        )
        for match in PATTERN.finditer(text)
    ]
