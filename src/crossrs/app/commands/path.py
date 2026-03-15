from typing import Annotated

from typer import Argument

from crossrs.app.app import app
from crossrs.db import get_path
from crossrs.utils.typer import typer_raise

__all__ = [
    'path',
]


@app.command()
def path(
        language: Annotated[
            str,
            Argument(help='Target language code whose data file should be printed.'),
        ],
) -> None:
    """Print the absolute path to CrossRS's data file for `language`."""
    language_path = get_path(language)
    if language_path.exists():
        print(language_path.absolute())
    else:
        typer_raise(f'Language "{language}" is not initialized.')
