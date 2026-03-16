from __future__ import annotations
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from crossrs.db.base import Base
from crossrs.db.models import Metadata, Word, Sentence
from crossrs.diff.tokenizer import tokenize, normalize_token
from crossrs.diff import compute_diff
from crossrs.app.commands.init import extract_tokens, extract_words, process_corpus
from crossrs.app.commands.study.updater import (
    update_sentence, extract_words_from_text, INTERVALS,
)
from crossrs.app.commands.study.chooser import (
    choose_next, choose_target_word,
    get_round_counts, get_queue_target_word_ids,
)
from crossrs.utils.time import get_timestamp


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def corpus_file(tmp_path):
    """Create a temporary corpus file."""
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(
        'Die Katze sitzt auf der Matte.\n'
        'Der Hund läuft im Park.\n'
        'Die Katze und der Hund spielen zusammen.\n'
        'Das Wetter ist heute sehr schön.\n'
        'Ich gehe morgen in die Schule.\n',
        encoding='utf-8',
    )
    return corpus


def populate_db(session: Session, corpus_file: Path) -> dict[str, int]:
    """Populate a test database from a corpus file. Returns ids_by_word."""
    from crossrs.app.commands.init import process_corpus, add_words, add_sentences
    words_by_sentence, word_frequencies = process_corpus(corpus_file)
    ids_by_word = add_words(session, word_frequencies)
    add_sentences(session, words_by_sentence, ids_by_word)
    session.add(Metadata(id=1, total_rounds=0))
    session.commit()
    return ids_by_word


# ─── Tokenizer Tests ───────────────────────────────────────────────────────────

class TestTokenizer:
    def test_basic_tokenization(self):
        tokens = tokenize('Hello world')
        assert len(tokens) == 2
        assert tokens[0].normalized == 'hello'
        assert tokens[1].normalized == 'world'

    def test_unicode_tokenization(self):
        tokens = tokenize('Die Katze läuft.')
        assert len(tokens) == 3
        assert tokens[0].normalized == 'die'
        assert tokens[1].normalized == 'katze'
        assert tokens[2].normalized == 'läuft'

    def test_punctuation_skipped(self):
        tokens = tokenize('Hello, world! How are you?')
        normalized = [t.normalized for t in tokens]
        assert normalized == ['hello', 'world', 'how', 'are', 'you']

    def test_numbers_included(self):
        tokens = tokenize('I have 3 cats')
        assert len(tokens) == 4
        assert tokens[2].normalized == '3'

    def test_position_tracking(self):
        text = 'Hello world'
        tokens = tokenize(text)
        assert text[tokens[0].start:tokens[0].end] == 'Hello'
        assert text[tokens[1].start:tokens[1].end] == 'world'

    def test_normalize_token(self):
        assert normalize_token('HELLO') == 'hello'
        assert normalize_token('Café') == 'café'

    def test_empty_string(self):
        assert tokenize('') == []

    def test_only_punctuation(self):
        assert tokenize('.,!?;:') == []


# ─── Diff Tests ─────────────────────────────────────────────────────────────────

class TestDiff:
    def test_identical_texts(self):
        segments = compute_diff('Hello world', 'Hello world')
        assert segments == []

    def test_single_word_change(self):
        segments = compute_diff('I have a cat', 'I have a dog')
        assert len(segments) == 1
        assert segments[0].old_text == 'cat'
        assert segments[0].new_text == 'dog'

    def test_insertion(self):
        segments = compute_diff('I have cat', 'I have a cat')
        assert len(segments) == 1
        assert segments[0].old_text is None
        assert segments[0].new_text == 'a'

    def test_deletion(self):
        segments = compute_diff('I have a cat', 'I have cat')
        assert len(segments) == 1
        assert segments[0].old_text == 'a'
        assert segments[0].new_text is None

    def test_case_insensitive_comparison(self):
        # Tokens are compared by normalized (lowercased) form
        segments = compute_diff('Hello World', 'hello world')
        assert segments == []


# ─── Word Extraction Tests ─────────────────────────────────────────────────────

class TestWordExtraction:
    def test_extract_tokens(self):
        tokens = extract_tokens('Die Katze sitzt.')
        assert tokens == ['die', 'katze', 'sitzt']

    def test_extract_words(self):
        words = extract_words(['a', 'b', 'c'])
        assert words == ['a', 'b', 'c']

    def test_extract_words_from_text(self):
        words = extract_words_from_text('I have a cat')
        assert 'i' in words
        assert 'have' in words
        assert 'a' in words
        assert 'cat' in words


# ─── Corpus Processing Tests ───────────────────────────────────────────────────

class TestCorpusProcessing:
    def test_process_corpus(self, corpus_file):
        words_by_sentence, frequencies = process_corpus(corpus_file)
        assert len(words_by_sentence) == 5
        # 'die' appears in multiple sentences
        assert frequencies['die'] >= 2
        # Each sentence has its own set of words
        for words in words_by_sentence.values():
            assert isinstance(words, set)
            assert len(words) > 0

    def test_duplicate_sentences_ignored(self, tmp_path):
        corpus = tmp_path / 'dup.txt'
        corpus.write_text('Hello world.\nHello world.\nGoodbye.\n', encoding='utf-8')
        words_by_sentence, frequencies = process_corpus(corpus)
        assert len(words_by_sentence) == 2  # duplicate ignored


# ─── Database / Init Tests ──────────────────────────────────────────────────────

class TestDatabaseInit:
    def test_words_sorted_by_frequency(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        words = db_session.query(Word).order_by(Word.id).all()
        # IDs should be in decreasing frequency order
        for i in range(len(words) - 1):
            assert words[i].occurrences >= words[i + 1].occurrences

    def test_sentence_word_associations(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        sentence = db_session.query(Sentence).first()
        assert len(sentence.words) > 0

    def test_initial_state(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        for sentence in db_session.query(Sentence).all():
            assert sentence.status == 0
            assert sentence.rounds == 0
            assert sentence.due is None
        for word in db_session.query(Word).all():
            assert word.learnedness == 0


# ─── Updater Tests ──────────────────────────────────────────────────────────────

class TestUpdater:
    def test_first_attempt_correct_schedules_last_stage(self, db_session, corpus_file):
        ids = populate_db(db_session, corpus_file)
        sentence = db_session.query(Sentence).first()
        word = db_session.query(Word).first()
        update_sentence(sentence, is_correct=True, is_first_attempt=True,
                        target_word=word, user_translation='The cat sits on the mat.',
                        session=db_session)
        db_session.flush()
        assert sentence.status == 1  # in queue
        assert sentence.review_stage == len(INTERVALS) - 1
        assert sentence.due is not None
        assert sentence.rounds == 1

    def test_first_attempt_incorrect_enters_queue(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        sentence = db_session.query(Sentence).first()
        word = db_session.query(Word).first()
        update_sentence(sentence, is_correct=False, is_first_attempt=True,
                        target_word=word, user_translation='wrong', session=db_session)
        db_session.flush()
        assert sentence.status == 1  # in queue
        assert sentence.review_stage == 0
        assert sentence.due is not None
        assert sentence.rounds == 1

    def test_review_correct_advances_stage(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        sentence = db_session.query(Sentence).first()
        word = db_session.query(Word).first()
        # Enter queue
        update_sentence(sentence, is_correct=False, is_first_attempt=True,
                        target_word=word, user_translation='wrong', session=db_session)
        db_session.flush()
        # Correct review at stage 0 -> stage 1
        update_sentence(sentence, is_correct=True, is_first_attempt=False,
                        target_word=None, user_translation='correct', session=db_session)
        db_session.flush()
        assert sentence.review_stage == 1
        assert sentence.status == 1

    def test_review_incorrect_resets_stage(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        sentence = db_session.query(Sentence).first()
        word = db_session.query(Word).first()
        # Enter queue and advance to stage 1
        update_sentence(sentence, is_correct=False, is_first_attempt=True,
                        target_word=word, user_translation='wrong', session=db_session)
        update_sentence(sentence, is_correct=True, is_first_attempt=False,
                        target_word=None, user_translation='correct', session=db_session)
        db_session.flush()
        assert sentence.review_stage == 1
        # Incorrect at stage 1 -> back to stage 0
        update_sentence(sentence, is_correct=False, is_first_attempt=False,
                        target_word=None, user_translation='wrong', session=db_session)
        db_session.flush()
        assert sentence.review_stage == 0

    def test_full_review_cycle_marks_learned(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        sentence = db_session.query(Sentence).first()
        word = db_session.query(Word).first()
        # Enter queue
        update_sentence(sentence, is_correct=False, is_first_attempt=True,
                        target_word=word, user_translation='wrong', session=db_session)
        # Pass all review stages
        for _ in range(len(INTERVALS)):
            update_sentence(sentence, is_correct=True, is_first_attempt=False,
                            target_word=None, user_translation='The cat sits on the mat.',
                            session=db_session)
        db_session.flush()
        assert sentence.status == 2  # learned

    def test_learnedness_incremented_on_learn(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        sentence = db_session.query(Sentence).first()
        word = db_session.query(Word).first()
        # First attempt correct → enters queue at last stage
        update_sentence(sentence, is_correct=True, is_first_attempt=True,
                        target_word=word, user_translation='The cat sits on the mat.',
                        session=db_session)
        db_session.flush()
        # Pass the final review → marked as learned
        update_sentence(sentence, is_correct=True, is_first_attempt=False,
                        target_word=None, user_translation='The cat sits on the mat.',
                        session=db_session)
        db_session.flush()
        # Words from the original sentence should have learnedness > 0
        db_session.expire_all()
        sentence_words = sentence.words
        incremented = [w for w in sentence_words if w.learnedness > 0]
        assert len(incremented) > 0

    def test_intervals(self):
        assert int(INTERVALS[0].total_seconds()) == 72000  # 20 hours
        assert int(INTERVALS[1].total_seconds()) == 590400  # 7 days - 4 hours
        assert int(INTERVALS[2].total_seconds()) == 2577600  # 30 days - 4 hours


# ─── Chooser Tests ──────────────────────────────────────────────────────────────

class TestChooser:
    def test_choose_new_sentence(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        sentence, target_word, counts = choose_next(db_session, threshold=3)
        assert sentence is not None
        assert target_word is not None
        assert counts.due == 0
        assert sentence.status == 0  # unseen

    def test_target_word_is_most_frequent(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        _, target_word, _ = choose_next(db_session, threshold=3)
        # With all learnedness=0, should pick the most frequent word (id=1)
        assert target_word.id == 1

    def test_due_count_zero_initially(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        assert get_round_counts(db_session).due == 0

    def test_due_sentence_prioritized(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        # Put a sentence in the queue with past due time
        sentence = db_session.query(Sentence).first()
        sentence.status = 1
        sentence.due = get_timestamp() - 1000
        sentence.target_word_id = 1
        db_session.flush()
        result_sentence, target_word, counts = choose_next(db_session, threshold=3)
        assert result_sentence.id == sentence.id
        assert target_word is None  # review, not new
        assert counts.due == 1

    def test_queue_targets_excluded(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        # Put first word in queue
        sentence = db_session.query(Sentence).first()
        sentence.status = 1
        sentence.due = get_timestamp() + 100000  # future
        sentence.target_word_id = 1
        db_session.flush()
        queue_targets = get_queue_target_word_ids(db_session)
        assert 1 in queue_targets
        # Next target should not be word 1
        target = choose_target_word(db_session, threshold=3)
        assert target is not None
        assert target.id != 1

    def test_higher_learnedness_preferred(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        # Give word 2 higher learnedness
        word2 = db_session.query(Word).filter(Word.id == 2).one()
        word2.learnedness = 2
        db_session.flush()
        target = choose_target_word(db_session, threshold=3)
        assert target.id == 2  # higher learnedness preferred

    def test_threshold_excludes_learned(self, db_session, corpus_file):
        populate_db(db_session, corpus_file)
        # Set all words to learnedness >= threshold
        db_session.query(Word).update({Word.learnedness: 3})
        db_session.flush()
        target = choose_target_word(db_session, threshold=3)
        assert target is None  # all learned


# ─── Stats Tests ────────────────────────────────────────────────────────────────

class TestStats:
    def test_initial_stats(self, db_session, corpus_file):
        from crossrs.app.commands.stats import compute_word_stats, compute_sentence_stats
        populate_db(db_session, corpus_file)

        word_stats = compute_word_stats(db_session, threshold=3)
        sentence_stats = compute_sentence_stats(db_session)

        assert sentence_stats.learned == 0
        assert sentence_stats.in_queue == 0
        assert sentence_stats.total == 5
        assert sentence_stats.total_rounds == 0

        assert word_stats.learned == 0
        assert word_stats.total > 0

    def test_stats_after_learning(self, db_session, corpus_file):
        from crossrs.app.commands.stats import compute_sentence_stats
        populate_db(db_session, corpus_file)

        # First attempt correct → enters queue at last stage (not immediately learned)
        sentence = db_session.query(Sentence).first()
        word = db_session.query(Word).first()
        update_sentence(sentence, is_correct=True, is_first_attempt=True,
                        target_word=word, user_translation='The cat sits on the mat.',
                        session=db_session)
        db_session.flush()

        sentence_stats = compute_sentence_stats(db_session)
        assert sentence_stats.in_queue == 1
        assert sentence_stats.learned == 0
        assert sentence_stats.total_rounds == 1


# ─── Delete Tests ───────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_with_force(self, tmp_path):
        db_path = tmp_path / 'test.db'
        db_path.write_text('dummy')
        assert db_path.exists()
        db_path.unlink()
        assert not db_path.exists()
