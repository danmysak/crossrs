from __future__ import annotations
from datetime import timedelta

from crossrs.db import Session
from crossrs.db.models import Metadata, Word, Sentence
from crossrs.diff.tokenizer import tokenize
from crossrs.utils.time import get_timestamp

__all__ = [
    'update_sentence',
    'undo_learned',
]

INTERVALS = [
    timedelta(days=1, hours=-4),   # 20 hours
    timedelta(days=7, hours=-4),   # 164 hours
    timedelta(days=30, hours=-4),  # ~716 hours
]


def extract_words_from_text(text: str) -> set[str]:
    """Extract all words from a text."""
    return {t.normalized for t in tokenize(text)}


def mark_seen(sentence: Sentence, user_translation: str, session: Session) -> None:
    """Set learnedness to 1 for any words at 0 in the sentence or user's translation."""
    sentence_word_ids = {word.id for word in sentence.words}
    translation_words = extract_words_from_text(user_translation)

    if translation_words:
        matching = session.query(Word).filter(Word.word.in_(translation_words)).all()
        translation_word_ids = {w.id for w in matching}
    else:
        translation_word_ids = set()

    all_word_ids = sentence_word_ids | translation_word_ids

    if all_word_ids:
        session.query(Word).filter(
            Word.id.in_(all_word_ids),
            Word.learnedness == 0,
        ).update({Word.learnedness: 1}, synchronize_session=False)


def mark_learned(sentence: Sentence, user_translation: str | None, session: Session) -> None:
    """Mark a sentence as learned and update learnedness for relevant words."""
    sentence.status = 2

    # Collect words from the original sentence
    sentence_word_ids = {word.id for word in sentence.words}

    # Collect words from the user's last translation
    translation_words = extract_words_from_text(user_translation) if user_translation else set()

    # Find word IDs that match the translation word strings
    if translation_words:
        matching = session.query(Word).filter(Word.word.in_(translation_words)).all()
        translation_word_ids = {w.id for w in matching}
    else:
        translation_word_ids = set()

    # Union of both sets
    all_word_ids = sentence_word_ids | translation_word_ids

    # Increment learnedness
    for word_id in all_word_ids:
        session.query(Word).filter(Word.id == word_id).update(
            {Word.learnedness: Word.learnedness + 1},
        )


def undo_learned(sentence: Sentence, session: Session) -> None:
    """Revert learnedness increments if a learned sentence is removed."""
    if sentence.status != 2:
        return

    sentence_word_ids = {word.id for word in sentence.words}

    for word_id in sentence_word_ids:
        session.query(Word).filter(Word.id == word_id, Word.learnedness > 0).update(
            {Word.learnedness: Word.learnedness - 1},
        )


def update_sentence(sentence: Sentence, is_correct: bool, is_first_attempt: bool,
                    target_word: Word | None, user_translation: str,
                    session: Session) -> None:
    """Update sentence state based on evaluation result."""
    now = get_timestamp()
    sentence.rounds += 1
    session.query(Metadata).filter(Metadata.id == 1).update(
        {Metadata.total_rounds: Metadata.total_rounds + 1},
    )

    if is_correct:
        mark_seen(sentence, user_translation, session)

    if is_first_attempt:
        # First time seeing this sentence
        if target_word is not None:
            sentence.target_word_id = target_word.id
        if is_correct:
            # Schedule to the last review stage (30 days - 4 hours)
            sentence.status = 1
            sentence.review_stage = len(INTERVALS) - 1
            sentence.due = now + int(INTERVALS[-1].total_seconds())
        else:
            # Enter the queue at the first stage
            sentence.status = 1
            sentence.review_stage = 0
            sentence.due = now + int(INTERVALS[0].total_seconds())
    else:
        # Review of an in-queue sentence
        if is_correct:
            next_stage = sentence.review_stage + 1
            if next_stage >= len(INTERVALS):
                # All review stages passed — learned
                mark_learned(sentence, user_translation, session)
            else:
                sentence.review_stage = next_stage
                sentence.due = now + int(INTERVALS[next_stage].total_seconds())
        else:
            # Failed review — reset schedule
            sentence.review_stage = 0
            sentence.due = now + int(INTERVALS[0].total_seconds())
