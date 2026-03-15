from __future__ import annotations
from typing import Annotated

from dataclasses import dataclass
from rich.console import Console
from rich.text import Text
from sqlalchemy import func, case
from typer import Argument, Option

from crossrs.app.app import app
from crossrs.db import get_session, Session
from crossrs.db.models import Metadata, Word, Sentence

__all__ = [
    'stats',
]

DEFAULT_THRESHOLD = 3


@dataclass
class WordStatsItem:
    learned: int = 0
    total: int = 0
    learned_occurrences: int = 0
    total_occurrences: int = 0


@dataclass
class SentenceStatsData:
    learned: int = 0
    in_queue: int = 0
    total: int = 0
    total_rounds: int = 0
    targeted_words: int = 0


def compute_word_stats(session: Session, threshold: int) -> WordStatsItem:
    """Compute word statistics using SQL aggregation."""
    row = session.query(
        func.count(Word.id),
        func.sum(Word.occurrences),
        func.sum(case((Word.learnedness >= threshold, 1), else_=0)),
        func.sum(case((Word.learnedness >= threshold, Word.occurrences), else_=0)),
    ).one()

    return WordStatsItem(
        learned=int(row[2] or 0),
        total=int(row[0] or 0),
        learned_occurrences=int(row[3] or 0),
        total_occurrences=int(row[1] or 0),
    )


def compute_sentence_stats(session: Session, threshold: int) -> SentenceStatsData:
    """Compute sentence statistics using SQL aggregation."""
    row = session.query(
        func.count(Sentence.id),
        func.sum(case((Sentence.status == 2, 1), else_=0)),
        func.sum(case((Sentence.status == 1, 1), else_=0)),
    ).one()

    meta = session.get(Metadata, 1)
    total_rounds = meta.total_rounds if meta else 0

    targeted_words = session.query(
        func.count(func.distinct(Sentence.target_word_id)),
    ).filter(
        Sentence.status == 1,
        Sentence.target_word_id.is_not(None),
    ).join(Word, Sentence.target_word_id == Word.id).filter(
        Word.learnedness < threshold,
    ).scalar()

    return SentenceStatsData(
        total=int(row[0]),
        total_rounds=total_rounds,
        learned=int(row[1] or 0),
        in_queue=int(row[2] or 0),
        targeted_words=int(targeted_words or 0),
    )


def format_section_title(title: str) -> Text:
    return Text(title, style='bold underline')


def format_stats_label(title: str) -> Text:
    return Text(f'{title}:', style='bold')


@app.command()
def stats(
        language: Annotated[
            str,
            Argument(help='Target language code to show statistics for.'),
        ],
        threshold: Annotated[
            int,
            Option(
                '--threshold', '-t',
                help='Learnedness threshold for words to be considered fully learned.',
            ),
        ] = DEFAULT_THRESHOLD,
) -> None:
    """Display study statistics for the given language."""
    with get_session(language) as session:
        word_stats = compute_word_stats(session, threshold)
        sentence_stats = compute_sentence_stats(session, threshold)

    console = Console(highlight=False)

    # Sentence statistics
    console.print(format_section_title('Sentence Statistics'))
    console.print(
        format_stats_label('Sentences'),
        Text(f'{sentence_stats.learned} learned + {sentence_stats.in_queue} in queue '
             f'/ {sentence_stats.total} total'),
    )
    console.print()

    # Word statistics
    console.print(format_section_title('Word Statistics'))
    text = Text(f'{word_stats.learned} learned / {word_stats.total} total')
    if word_stats.total_occurrences > 0:
        coverage = word_stats.learned_occurrences / word_stats.total_occurrences
        text.append(f' ')
        text.append(Text(f'(coverage: {coverage:.1%})', style='dim'))
    console.print(format_stats_label('Words'), text)
    if sentence_stats.targeted_words > 0:
        console.print(
            format_stats_label('Targeted'),
            Text(f'{sentence_stats.targeted_words} unlearned words in queue'),
        )
    console.print()

    # Total rounds
    console.print(format_stats_label('Total rounds'), sentence_stats.total_rounds)
