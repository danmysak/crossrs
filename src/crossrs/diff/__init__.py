from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher

from .tokenizer import tokenize

__all__ = [
    'compute_diff',
    'DiffSegment',
]


@dataclass
class DiffSegment:
    old_start: int
    old_end: int
    old_text: str | None
    new_start: int
    new_end: int
    new_text: str | None


def compute_diff(old: str, new: str) -> list[DiffSegment]:
    """Compute the diff between two texts, returning segments where they differ."""
    segments: list[DiffSegment] = []
    old_tokens = tokenize(old)
    new_tokens = tokenize(new)
    for (tag, i1, i2, j1, j2) in SequenceMatcher(
        a=old_tokens, b=new_tokens, autojunk=False,
    ).get_opcodes():
        if tag != 'equal':
            segments.append(DiffSegment(
                old_start=old_tokens[i1].start if i2 > i1 else (old_tokens[i1 - 1].end if i1 > 0 else 0),
                old_end=old_tokens[i2 - 1].end if i2 > i1 else (old_tokens[i1 - 1].end if i1 > 0 else 0),
                old_text=old[old_tokens[i1].start:old_tokens[i2 - 1].end] if i2 > i1 else None,
                new_start=new_tokens[j1].start if j2 > j1 else (new_tokens[j1 - 1].end if j1 > 0 else 0),
                new_end=new_tokens[j2 - 1].end if j2 > j1 else (new_tokens[j1 - 1].end if j1 > 0 else 0),
                new_text=new[new_tokens[j1].start:new_tokens[j2 - 1].end] if j2 > j1 else None,
            ))
    return segments
