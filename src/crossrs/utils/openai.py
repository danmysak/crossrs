from __future__ import annotations

import time
from typing import TypeVar, Callable

from openai import APIError, APIConnectionError, AuthenticationError, RateLimitError

from crossrs.utils.typer import typer_raise

__all__ = [
    'api_call_with_retries',
]

T = TypeVar('T')

MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]


def api_call_with_retries(fn: Callable[[], T], on_error: Callable[[str], None] | None = None) -> T:
    """Call fn() with automatic retries on transient OpenAI errors.

    on_error is called with a message before each retry (e.g. to print to console).
    Raises on non-transient errors or after exhausting retries.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except AuthenticationError:
            typer_raise('Invalid API key. Check --api-key or CROSSRS_API_KEY.')
        except RateLimitError as e:
            _handle_retry(attempt, f'Rate limited: {e}', on_error)
        except APIConnectionError as e:
            _handle_retry(attempt, f'Connection error: {e}', on_error)
        except APIError as e:
            if e.status_code and e.status_code >= 500:
                _handle_retry(attempt, f'Server error: {e}', on_error)
            else:
                typer_raise(f'OpenAI API error: {e.message}')


def _handle_retry(attempt: int, message: str, on_error: Callable[[str], None] | None) -> None:
    if attempt >= MAX_RETRIES:
        raise RuntimeError(f'OpenAI API failed after {MAX_RETRIES + 1} attempts: {message}')
    delay = RETRY_DELAYS[attempt]
    full_message = f'{message} — retrying in {delay}s...'
    if on_error:
        on_error(full_message)
    time.sleep(delay)
