from __future__ import annotations
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel

from crossrs.db import get_session
from crossrs.db.models import EvaluationCache, Sentence, TranslationCache
from crossrs.utils.openai import api_call_with_retries
from crossrs.utils.strings import normalize

__all__ = [
    'evaluate',
    'invalidate_evaluation_cache',
    'translate_to_source',
    'Evaluation',
]

TEMPERATURE = 0.0


@dataclass
class Evaluation:
    is_correct: bool
    corrected_translation: str | None


class TranslationOutput(BaseModel):
    translation: str


class EvaluationOutput(BaseModel):
    is_correct: bool
    corrected_translation: str | None


def generate_translation_prompt(target_lang: str, source_lang: str, sentence: str) -> str:
    """Generate a prompt to translate a sentence from target language to source language."""
    return '\n'.join([
        f'Translate the following sentence from the language with code `{target_lang}` '
        f'into the language with code `{source_lang}`. '
        f'Provide a natural, accurate translation.',
        f'',
        f'`{sentence}`',
    ])


def generate_evaluation_prompt(
        target_lang: str, source_lang: str,
        source_translation: str, user_translation: str,
) -> str:
    """Generate a prompt to evaluate the user's translation back into the target language."""
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
        f"Determine whether the learner's translation is fully grammatical, natural, "
        f'and conveys the same meaning as the `{source_lang}` sentence.',
        f'',
        f"If the learner's translation is correct (or differs only in trivial ways like "
        f'punctuation), set `is_correct` to true and `corrected_translation` to null.',
        f'',
        f'If the translation has any grammatical, lexical, or meaning issues, set '
        f'`is_correct` to false and provide a `corrected_translation` that is a '
        f"**minimally edited** version of the learner's attempt that makes it fully "
        f'grammatical, natural, and meaning-preserving. Change as few words as possible.',
    ])


def request_translation(sentence: str, target_lang: str, source_lang: str, model: str, api_key: str) -> str:
    """Request translation of a sentence from target to source language."""
    return api_call_with_retries(lambda: OpenAI(api_key=api_key).responses.parse(
        model=model,
        temperature=TEMPERATURE,
        text_format=TranslationOutput,
        input=generate_translation_prompt(target_lang, source_lang, sentence),
    ).output_parsed.translation)


def translate_to_source(sentence: Sentence, target_lang: str, source_lang: str,
                        model: str, api_key: str) -> str:
    """Translate a sentence to the source language, using cache if available."""
    with get_session(target_lang) as session:
        if cached := session.query(TranslationCache).filter(
            TranslationCache.sentence_id == sentence.id,
            TranslationCache.source_lang == source_lang,
        ).limit(1).first():
            return cached.translation
        else:
            translation = request_translation(sentence.sentence, target_lang, source_lang,
                                              model, api_key)
            session.add(TranslationCache(
                sentence_id=sentence.id,
                source_lang=source_lang,
                translation=translation,
                model=model,
            ))
            return translation


def request_evaluation(source_translation: str, user_translation: str,
                       target_lang: str, source_lang: str, model: str,
                       api_key: str) -> EvaluationOutput:
    """Request LLM evaluation of the user's translation."""
    return api_call_with_retries(lambda: OpenAI(api_key=api_key).responses.parse(
        model=model,
        temperature=TEMPERATURE,
        text_format=EvaluationOutput,
        input=generate_evaluation_prompt(target_lang, source_lang,
                                         source_translation, user_translation),
    ).output_parsed)


def cached_evaluation(sentence: Sentence, source_translation: str, user_translation: str,
                      target_lang: str, source_lang: str, model: str,
                      api_key: str) -> EvaluationOutput:
    """Request evaluation with caching."""
    with get_session(target_lang) as session:
        if cached := session.query(EvaluationCache).filter(
            EvaluationCache.sentence_id == sentence.id,
            EvaluationCache.source_lang == source_lang,
            EvaluationCache.model == model,
            EvaluationCache.translation == user_translation,
        ).limit(1).first():
            return EvaluationOutput.model_validate_json(cached.evaluation)
        else:
            output = request_evaluation(source_translation,
                                        user_translation, target_lang, source_lang,
                                        model, api_key)
            session.add(EvaluationCache(
                sentence_id=sentence.id,
                source_lang=source_lang,
                model=model,
                translation=user_translation,
                evaluation=output.model_dump_json(),
            ))
            return output


def build_evaluation(output: EvaluationOutput, sentence: Sentence,
                     user_translation: str) -> Evaluation:
    """Build an Evaluation from the LLM output."""
    is_correct = (output.is_correct
                  or normalize(output.corrected_translation or '') == normalize(user_translation)
                  or normalize(user_translation) == normalize(sentence.sentence))
    return Evaluation(
        is_correct=is_correct,
        corrected_translation=None if is_correct else output.corrected_translation,
    )


def evaluate(sentence: Sentence, source_translation: str, user_translation: str,
             target_lang: str, source_lang: str, model: str,
             api_key: str) -> Evaluation:
    """Evaluate the user's translation of a sentence."""
    if normalize(user_translation) == normalize(sentence.sentence):
        return Evaluation(is_correct=True, corrected_translation=None)
    output = cached_evaluation(sentence, source_translation, user_translation,
                               target_lang, source_lang, model, api_key)
    return build_evaluation(output, sentence, user_translation)


def invalidate_evaluation_cache(sentence: Sentence, user_translation: str,
                                source_lang: str, model: str,
                                target_lang: str, session) -> None:
    """Delete cached evaluation for a sentence+translation marked as correct."""
    session.query(EvaluationCache).filter(
        EvaluationCache.sentence_id == sentence.id,
        EvaluationCache.source_lang == source_lang,
        EvaluationCache.model == model,
        EvaluationCache.translation == user_translation,
    ).delete()
