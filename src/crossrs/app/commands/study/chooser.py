from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import case, func

from crossrs.db import Session
from crossrs.db.models import Word, Sentence, SentenceWord
from crossrs.utils.time import get_timestamp
from crossrs.utils.typer import typer_raise

__all__ = [
    'choose_next',
]

MAX_CANDIDATE_SENTENCES = 10000


@dataclass
class RoundCounts:
    due: int = 0
    in_queue: int = 0
    learned: int = 0


def get_round_counts(session: Session) -> RoundCounts:
    """Return due, in-queue, and learned sentence counts in a single query."""
    now = get_timestamp()
    row = session.query(
        func.sum(case((
            (Sentence.status == 1) & Sentence.due.is_not(None) & (Sentence.due <= now), 1,
        ), else_=0)),
        func.sum(case((Sentence.status == 1, 1), else_=0)),
        func.sum(case((Sentence.status == 2, 1), else_=0)),
    ).one()
    return RoundCounts(
        due=int(row[0] or 0),
        in_queue=int(row[1] or 0),
        learned=int(row[2] or 0),
    )


def choose_due_sentence(session: Session) -> Sentence | None:
    """Return the sentence with the earliest due timestamp, or None."""
    return session.query(Sentence).filter(
        Sentence.status == 1,
        Sentence.due.is_not(None),
        Sentence.due <= get_timestamp(),
    ).order_by(Sentence.due).limit(1).first()


def get_next_due_time(session: Session) -> int | None:
    """Return the earliest future due timestamp, or None if no sentences are in the queue."""
    result = session.query(func.min(Sentence.due)).filter(
        Sentence.status == 1,
        Sentence.due.is_not(None),
    ).scalar()
    return result


def get_queue_target_word_ids(session: Session) -> set[int]:
    """Return the set of word IDs that are targets of sentences currently in the queue."""
    rows = session.query(Sentence.target_word_id).filter(
        Sentence.status == 1,
        Sentence.target_word_id.is_not(None),
    ).all()
    return {row[0] for row in rows}


def choose_target_word(session: Session, threshold: int) -> Word | None:
    """
    Choose the next target word using a single SQL query with EXISTS subquery:
    1. learnedness < threshold
    2. Not already a target of any in-queue sentence
    3. Has at least one unseen sentence
    4. Highest learnedness first
    5. Highest occurrences (lowest id) among ties
    """
    queue_targets = get_queue_target_word_ids(session)

    # Correlated EXISTS subquery: does this word appear in any unseen sentence?
    unseen_exists = session.query(SentenceWord.sentence_id).join(
        Sentence, SentenceWord.sentence_id == Sentence.id,
    ).filter(
        SentenceWord.word_id == Word.id,
        Sentence.status == 0,
    ).exists()

    query = session.query(Word).filter(
        Word.learnedness < threshold,
        unseen_exists,
    )
    if queue_targets:
        query = query.filter(~Word.id.in_(queue_targets))

    return query.order_by(Word.learnedness.desc(), Word.id).limit(1).first()


def choose_sentence_for_word(session: Session, word: Word, threshold: int) -> Sentence:
    """
    Choose a sentence for the given word. Samples MAX_CANDIDATE_SENTENCES random unseen
    sentences, then ranks them entirely in SQL:
    1. Fewest unseen words (learnedness == 0)
    2. Most seen-but-not-learned words (0 < learnedness < threshold)
    3. Smallest sum of word ids (frequency rankings)
    """
    candidates_sq = (
        session.query(SentenceWord.sentence_id)
        .join(Sentence, SentenceWord.sentence_id == Sentence.id)
        .filter(
            SentenceWord.word_id == word.id,
            Sentence.status == 0,
        )
        .order_by(SentenceWord.random_key)
        .limit(MAX_CANDIDATE_SENTENCES)
        .subquery()
    )

    best = (
        session.query(candidates_sq.c.sentence_id)
        .join(SentenceWord, candidates_sq.c.sentence_id == SentenceWord.sentence_id)
        .join(Word, SentenceWord.word_id == Word.id)
        .group_by(candidates_sq.c.sentence_id)
        .order_by(
            func.sum(case((Word.learnedness == 0, 1), else_=0)),
            func.sum(case((Word.learnedness.between(1, threshold - 1), 1), else_=0)).desc(),
            func.sum(Word.id),
        )
        .limit(1)
        .first()
    )

    return session.get(Sentence, best[0])


def choose_next(session: Session, threshold: int) -> tuple[Sentence, Word | None, RoundCounts]:
    """
    Return the next sentence to study, the target word (None for review sentences),
    and the current round counts.
    """
    counts = get_round_counts(session)

    # Priority 1: due review sentence
    if due_sentence := choose_due_sentence(session):
        return due_sentence, None, counts

    # Priority 2: new sentence via word algorithm
    if target_word := choose_target_word(session, threshold):
        sentence = choose_sentence_for_word(session, target_word, threshold)
        return sentence, target_word, counts

    # Priority 3: future due sentence exists — inform user
    if next_due := get_next_due_time(session):
        now = get_timestamp()
        wait_seconds = max(0, next_due - now)
        hours = wait_seconds // 3600
        minutes = (wait_seconds % 3600) // 60
        if hours > 0:
            time_str = f'{hours}h {minutes}m'
        else:
            time_str = f'{minutes}m'
        typer_raise(f'No sentences to review right now. Next review due in {time_str}.')

    # Nothing left
    typer_raise('🎉 Congratulations! You have learned all the words in the corpus!')
