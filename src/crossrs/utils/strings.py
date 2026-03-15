import unicodedata

__all__ = [
    'normalize',
]


def normalize(s: str) -> str:
    return unicodedata.normalize('NFC', s.strip())
