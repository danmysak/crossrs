from __future__ import annotations
from typing import Never

import typer

__all__ = [
    'typer_raise',
]


def typer_raise(message: str) -> Never:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)
