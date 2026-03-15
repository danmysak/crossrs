from openai import OpenAI
from typing import Generator

from crossrs.utils.openai import api_call_with_retries

__all__ = [
    'explain',
]

TEMPERATURE = 0.0


def generate_prompt(target_lang: str, source_lang: str,
                    source_translation: str, user_translation: str) -> str:
    """Generate a prompt instructing the LLM to explain why the user's translation was incorrect."""
    return '\n'.join([
        f'A learner is practicing the language with code `{target_lang}` by translating '
        f'sentences from `{source_lang}` back into `{target_lang}`.',
        f'',
        f'The sentence in `{source_lang}`:',
        f'`{source_translation}`',
        f'',
        f"The learner's translation into `{target_lang}`:",
        f'`{user_translation}`',
        f'',
        f"Explain to the learner why their translation is not fully correct. "
        f"Focus on the specific grammatical, lexical, or meaning issues. "
        f"Be concise and helpful. Do not mention the language codes "
        f"`{target_lang}` or `{source_lang}` in your explanation.",
    ])


def explain(source_translation: str, user_translation: str,
            target_lang: str, source_lang: str, model: str,
            api_key: str) -> Generator[str, None, None]:
    """Stream an explanation for why the user's translation was incorrect."""
    stream = api_call_with_retries(lambda: OpenAI(api_key=api_key).responses.create(
        model=model,
        temperature=TEMPERATURE,
        input=generate_prompt(target_lang, source_lang,
                              source_translation, user_translation),
        stream=True,
    ))
    for event in stream:
        if event.type == 'response.output_text.delta':
            yield event.delta
