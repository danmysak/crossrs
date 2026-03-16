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
    in_progress: int = 0
    total: int = 0
    learned_occurrences: int = 0
    in_progress_occurrences: int = 0
    total_occurrences: int = 0


@dataclass
class SentenceStatsData:
    learned: int = 0
    in_queue: int = 0
    total: int = 0
    total_rounds: int = 0


def compute_word_stats(session: Session, threshold: int) -> WordStatsItem:
    """Compute word statistics using SQL aggregation."""
    row = session.query(
        func.count(Word.id),
        func.sum(Word.occurrences),
        func.sum(case((Word.learnedness >= threshold, 1), else_=0)),
        func.sum(case((Word.learnedness >= threshold, Word.occurrences), else_=0)),
        func.sum(case((Word.learnedness.between(1, threshold - 1), 1), else_=0)),
        func.sum(case((Word.learnedness.between(1, threshold - 1), Word.occurrences), else_=0)),
    ).one()

    return WordStatsItem(
        learned=int(row[2] or 0),
        in_progress=int(row[4] or 0),
        total=int(row[0] or 0),
        learned_occurrences=int(row[3] or 0),
        in_progress_occurrences=int(row[5] or 0),
        total_occurrences=int(row[1] or 0),
    )


def compute_sentence_stats(session: Session) -> SentenceStatsData:
    """Compute sentence statistics using SQL aggregation."""
    row = session.query(
        func.count(Sentence.id),
        func.sum(case((Sentence.status == 2, 1), else_=0)),
        func.sum(case((Sentence.status == 1, 1), else_=0)),
    ).one()

    meta = session.get(Metadata, 1)
    total_rounds = meta.total_rounds if meta else 0

    return SentenceStatsData(
        total=int(row[0]),
        total_rounds=total_rounds,
        learned=int(row[1] or 0),
        in_queue=int(row[2] or 0),
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
        sentence_stats = compute_sentence_stats(session)

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
    total_occ = word_stats.total_occurrences or 1

    def format_coverage(occurrences: int) -> str:
        return f'(coverage: {occurrences / total_occ:.1%})'

    learned_text = Text(f'{word_stats.learned} ')
    learned_text.append(Text(format_coverage(word_stats.learned_occurrences), style='dim'))
    console.print(format_stats_label('Learned'), learned_text)

    in_progress_text = Text(f'{word_stats.in_progress} ')
    in_progress_text.append(Text(format_coverage(word_stats.in_progress_occurrences), style='dim'))
    console.print(format_stats_label('In progress'), in_progress_text)

    combined = word_stats.learned + word_stats.in_progress
    combined_occ = word_stats.learned_occurrences + word_stats.in_progress_occurrences
    combined_text = Text(f'{combined} ')
    combined_text.append(Text(format_coverage(combined_occ), style='dim'))
    console.print(format_stats_label('Learned + in progress'), combined_text)

    console.print(format_stats_label('Total'), word_stats.total)
    console.print()

    # Total rounds
    console.print(format_stats_label('Total rounds'), sentence_stats.total_rounds)
