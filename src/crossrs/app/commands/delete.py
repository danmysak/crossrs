from typing import Annotated

from typer import Argument, Option

from crossrs.app.app import app
from crossrs.db import get_path
from crossrs.utils.typer import typer_raise

__all__ = [
    'delete',
]


@app.command()
def delete(
        language: Annotated[
            str,
            Argument(help='Target language code whose data should be removed.'),
        ],
        force: Annotated[
            bool | None,
            Option(
                '--force',
                '-f',
                help='Skip the confirmation prompt and delete immediately.',
            ),
        ] = False,
) -> None:
    """Delete all stored data for `language`."""
    path = get_path(language)
    if not path.exists():
        typer_raise(f'Language "{language}" is not initialized.')

    if not force and not input(
            f'Are you sure you want to delete all data for "{language}"? (y/N): ',
    ).lower().startswith('y'):
        typer_raise('Operation canceled.')

    path.unlink(missing_ok=True)
    print(f'Deleted all data for "{language}".')
