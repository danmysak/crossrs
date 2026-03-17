from dataclasses import dataclass
from typing import Callable

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.text import Text

from crossrs.utils.console import clear_previous

__all__ = [
    'ask',
    'ExtraOption',
]

OPTION_DELIMITER = ' | '


@dataclass
class ExtraOption:
    title: str
    action: Callable[[], bool | None]
    key: str | None = None  # Custom key; defaults to first character of title
    returns: bool = False  # If True, ask() returns empty string after action


def normalize_key(text: str) -> str:
    return text.lower()


def build_options(extra_options: list[ExtraOption]) -> tuple[list[list[Text]], dict[str, ExtraOption]]:
    """Build the list of option prompts and a mapping from option keys to their actions."""
    prompt = []
    mapping: dict[str, ExtraOption] = {}
    for option in extra_options:
        if not option.title:
            raise ValueError('Option title cannot be empty.')
        key = option.key or option.title[0]
        key_lower = normalize_key(key)
        if key_lower in mapping:
            raise ValueError(f'Duplicate key for options: {key_lower}')
        mapping[key_lower] = option
        # Bold the key characters: if key matches first letters of words, bold those
        if len(key) > 1:
            words = option.title.split()
            parts: list[Text] = []
            for i, word in enumerate(words):
                if i < len(key):
                    parts.append(Text(word[0], style='bold'))
                    parts.append(Text(word[1:]))
                else:
                    parts.append(Text(word))
                if i < len(words) - 1:
                    parts.append(Text(' '))
            prompt.append(parts)
        else:
            prompt.append([Text(option.title[0], style='bold'), Text(option.title[1:])])
    return prompt, mapping


def ask(
        console: Console,
        prompt: Text,
        extra_options: list[ExtraOption],
        voice_input: Callable[[], str | None] | None = None,
) -> str:
    """Prompt the user for input or to choose one of the extra options.

    If voice_input is provided, pressing Enter (empty input) triggers voice recording.
    """
    options_prompt, options_mapping = build_options(extra_options)

    def print_prompt() -> None:
        console.print(prompt, end='')
        for option_prompt in options_prompt:
            console.print(Text(OPTION_DELIMITER, style='dim'), end='')
            console.print(*option_prompt, sep='', end='')
        console.print()

    print_prompt()
    while True:
        response = PromptSession().prompt()
        normalized = response.strip()
        if not normalized and voice_input is not None:
            clear_previous()  # Clear just the empty input line; keep the prompt
            text = voice_input()
            if text is not None:
                # Print transcription below prompt so caller sees 2 lines (prompt + text)
                console.print(text)
                return text
            print_prompt()
            continue
        if option := options_mapping.get(normalize_key(normalized)):
            clear_previous(2)  # Clear both user input and prompt line
            result = option.action()
            if option.returns or result is True:
                return ''
            print_prompt()
        elif normalized:
            return normalized
        else:
            return ''
