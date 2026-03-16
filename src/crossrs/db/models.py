from __future__ import annotations

from sqlalchemy import ForeignKey, Index, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

__all__ = [
    'EvaluationCache',
    'Metadata',
    'Word',
    'Sentence',
    'SentenceWord',
    'TranslationCache',
]


class Metadata(Base):
    __tablename__ = 'metadata'

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    total_rounds: Mapped[int] = mapped_column(default=0)


class Word(Base):
    __tablename__ = 'words'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    word: Mapped[str]
    occurrences: Mapped[int]
    learnedness: Mapped[int] = mapped_column(default=0)

    sentences: Mapped[list[Sentence]] = relationship(
        'Sentence',
        secondary='sentence_words',
        order_by=lambda: SentenceWord.random_key,
        back_populates='words',
        passive_deletes=True,
    )


Index('ix_words_learnedness_id', Word.learnedness, Word.id)


class Sentence(Base):
    __tablename__ = 'sentences'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sentence: Mapped[str]
    status: Mapped[int] = mapped_column(default=0)  # 0=unseen, 1=in_queue, 2=learned
    target_word_id: Mapped[int | None] = mapped_column(
        ForeignKey('words.id', ondelete='SET NULL', onupdate='CASCADE'),
        default=None,
    )
    due: Mapped[int | None] = mapped_column(default=None)  # Unix timestamp
    review_stage: Mapped[int] = mapped_column(default=0)
    rounds: Mapped[int] = mapped_column(default=0)

    words: Mapped[list[Word]] = relationship(
        'Word',
        secondary='sentence_words',
        order_by=Word.id,
        back_populates='sentences',
        passive_deletes=True,
    )


Index('ix_sentences_due_in_queue', Sentence.due, sqlite_where=Sentence.status == 1)
Index('ix_sentences_target_word_in_queue', Sentence.target_word_id,
      sqlite_where=(Sentence.status == 1) & (Sentence.target_word_id.is_not(None)))
Index('ix_sentences_status', Sentence.status)


class SentenceWord(Base):
    __tablename__ = 'sentence_words'

    sentence_id: Mapped[int] = mapped_column(
        ForeignKey('sentences.id', ondelete='CASCADE', onupdate='CASCADE'),
        primary_key=True,
    )
    word_id: Mapped[int] = mapped_column(
        ForeignKey('words.id', ondelete='CASCADE', onupdate='CASCADE'),
        primary_key=True,
    )
    random_key: Mapped[int] = mapped_column(server_default=text('(abs(random()) % 16384)'))

    __table_args__ = (
        Index('ix_sentence_words_word_id_random_key', 'word_id', 'random_key'),
        {'sqlite_with_rowid': False},
    )


class TranslationCache(Base):
    __tablename__ = 'translation_cache'

    sentence_id: Mapped[int] = mapped_column(
        ForeignKey('sentences.id', ondelete='CASCADE', onupdate='CASCADE'),
        primary_key=True,
    )
    source_lang: Mapped[str] = mapped_column(primary_key=True)
    translation: Mapped[str]
    model: Mapped[str]


class EvaluationCache(Base):
    __tablename__ = 'evaluation_cache'

    sentence_id: Mapped[int] = mapped_column(
        ForeignKey('sentences.id', ondelete='CASCADE', onupdate='CASCADE'),
        primary_key=True,
    )
    source_lang: Mapped[str] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(primary_key=True)
    translation: Mapped[str] = mapped_column(primary_key=True)
    evaluation: Mapped[str]
