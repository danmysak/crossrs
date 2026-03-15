from time import time

__all__ = [
    'get_timestamp',
]


def get_timestamp() -> int:
    return int(time())
