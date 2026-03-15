from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Annotated, Generator

from sqlalchemy import text
from tqdm import tqdm
from typer import Argument

from crossrs.app.app import app
from crossrs.db import get_session, Session
from crossrs.db.models import Metadata, Word, Sentence, SentenceWord
from crossrs.diff.tokenizer import tokenize
from crossrs.utils.typer import typer_raise

__all__ = [
    'init',
]


def get_sentences(corpus: Path) -> Generator[str, None, None]:
    """Yield sentences from the corpus file."""
    with corpus.open('r', encoding='utf-8') as file:
        for line in file:
            if sentence := line.strip():
                yield sentence


def extract_tokens(sentence: str) -> list[str]:
    """Extract normalized tokens from a sentence."""
    return [token.normalized for token in tokenize(sentence)]


def extract_words(tokens: list[str]) -> list[str]:
    """Extract words from a list of tokens."""
    return list(tokens)


def process_corpus(corpus: Path) -> tuple[dict[str, set[str]], Counter[str]]:
    """Process the corpus and return unique words by sentence and word frequencies."""
    words_by_sentence: dict[str, set[str]] = {}
    word_frequencies: Counter[str] = Counter()

    for sentence in tqdm(get_sentences(corpus), desc='Processing sentences'):
        if sentence not in words_by_sentence:
            tokens = extract_tokens(sentence)
            if tokens:
                words = extract_words(tokens)
                words_by_sentence[sentence] = set(words)
                word_frequencies.update(words)

    return words_by_sentence, word_frequencies


def add_words(session: Session, frequencies: Counter[str]) -> dict[str, int]:
    """Add words to the database sorted by frequency and return a mapping of words to their IDs."""
    ids_by_word: dict[str, int] = {}
    for word, occurrences in tqdm(frequencies.most_common(), desc='Adding words'):
        word_obj = Word(
            word=word,
            occurrences=occurrences,
            learnedness=0,
        )
        session.add(word_obj)
        session.flush()
        ids_by_word[word] = word_obj.id
    return ids_by_word


def add_sentences(session: Session, words_by_sentence: dict[str, set[str]],
                  ids_by_word: dict[str, int]) -> None:
    """Add sentences to the database with their word associations."""
    for sentence_text, words in tqdm(words_by_sentence.items(), desc='Adding sentences'):
        sentence_obj = Sentence(sentence=sentence_text)
        session.add(sentence_obj)
        session.flush()
        for word in words:
            session.add(SentenceWord(
                sentence_id=sentence_obj.id,
                word_id=ids_by_word[word],
            ))
        session.flush()


@app.command()
def init(
        language: Annotated[
            str,
            Argument(help='Target language code (e.g., "de", "fr", "uk").'),
        ],
        corpus: Annotated[
            Path,
            Argument(
                dir_okay=False,
                exists=True,
                readable=True,
                resolve_path=True,
                help='Plain-text file containing one sentence per line.',
            ),
        ],
) -> None:
    """Initialize CrossRS for a new target language."""
    with get_session(language) as session:
        if session.query(Sentence).limit(1).first():
            typer_raise(f'CrossRS is already initialized for language "{language}".')
        words_by_sentence, word_frequencies = process_corpus(corpus)
        if not word_frequencies:
            typer_raise('No valid sentences found in the corpus.')
        ids_by_word = add_words(session, word_frequencies)
        add_sentences(session, words_by_sentence, ids_by_word)
        session.add(Metadata(id=1, total_rounds=0))
        print('Committing changes to the database...')
        session.commit()
        print('Optimizing the database...')
        session.execute(text('vacuum'))
    print(f'Initialized CrossRS for language "{language}" '
          f'with {len(words_by_sentence)} unique sentences '
          f'and {len(word_frequencies)} unique words.')
