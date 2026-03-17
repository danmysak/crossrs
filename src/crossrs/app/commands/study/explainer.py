from __future__ import annotations

from dataclasses import dataclass
from openai import OpenAI
from pydantic import BaseModel

from crossrs.utils.openai import api_call_with_retries

__all__ = [
    'explain',
    'ExplanationResult',
]


class ExplanationOutput(BaseModel):
    is_user_correct: bool
    explanation: str


@dataclass
class ExplanationResult:
    is_user_correct: bool
    explanation: str


def generate_prompt(target_lang: str, source_lang: str,
                    source_translation: str, user_translation: str,
                    corrected_translation: str) -> str:
    """Generate a prompt for the model to judge the correction and explain."""
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
        f'The minimally corrected version:',
        f'`{corrected_translation}`',
        f'',
        f'Compare the two translations carefully. Determine whether the corrections '
        f'are justified — do they actually fix grammatical, lexical, or meaning issues, '
        f'or was the original translation already fully correct?',
        f'',
        f'Set `is_user_correct` to true if the learner\'s original translation was '
        f'already fully grammatical, natural, and conveying the same meaning. '
        f'Set it to false if the corrections were justified.',
        f'',
        f'If `is_user_correct` is false, provide a concise explanation of the '
        f'grammatical, lexical, or meaning issues in the learner\'s translation. '
        f'Do not start with phrases like "The correction is justified" — just '
        f'explain the issues directly. If `is_user_correct` is true, set '
        f'`explanation` to an empty string.',
        f'',
        f'Do not mention the language codes `{target_lang}` or `{source_lang}` '
        f'in your explanation.',
    ])


def explain(source_translation: str, user_translation: str,
            corrected_translation: str,
            target_lang: str, source_lang: str, model: str,
            api_key: str) -> ExplanationResult:
    """Ask the model to judge the correction and explain if needed."""
    output = api_call_with_retries(lambda: OpenAI(api_key=api_key).responses.parse(
        model=model,
        reasoning={"effort": "medium"},
        text_format=ExplanationOutput,
        input=generate_prompt(target_lang, source_lang,
                              source_translation, user_translation,
                              corrected_translation),
    ).output_parsed)
    return ExplanationResult(
        is_user_correct=output.is_user_correct,
        explanation=output.explanation,
    )
