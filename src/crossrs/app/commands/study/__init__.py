import os
import threading
from typing import Annotated

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from typer import Argument, Option

from crossrs.app.app import app
from crossrs.db import get_session
from crossrs.db.models import Sentence, Word
from crossrs.diff import compute_diff
from crossrs.utils.console import clear_previous
from .chooser import choose_next, get_round_counts
from .evaluator import evaluate, translate_to_source
from .explainer import explain
from .interaction import ask, ExtraOption
from .updater import update_sentence, undo_learned

__all__ = [
    'study',
]

DEFAULT_THRESHOLD = 3

SECTION_DELIMITER = Text('-' * 3, style='dim')

CORRECT_MARK = '✅'
INCORRECT_MARK = '❌'

EVALUATION_IN_PROGRESS = Text('Evaluating translation...', style='dim')
PREPARING_NEXT = Text('Preparing next sentence...', style='dim')
ENTER_TO_CONTINUE = Text('Press Enter to continue...', style='dim')

QUIT_TITLE = 'Quit'
EXPLAIN_TITLE = 'Explain'
REMOVE_TITLE = 'Remove sentence'
MARK_CORRECT_TITLE = 'Mark correct'

CONSOLE = Console(highlight=False)

QUIT_OPTION = [ExtraOption(QUIT_TITLE, lambda: exit(0))]


def build_explanation_option(source_translation: str,
                             user_translation: str, target_lang: str,
                             source_lang: str, model: str,
                             api_key: str) -> list[ExtraOption]:
    """Build an option to explain why the user's translation was incorrect."""
    def do_explain():
        chunks: list[str] = []
        with CONSOLE.screen():
            with Live(console=CONSOLE, transient=True) as live:
                for chunk in explain(source_translation, user_translation,
                                     target_lang, source_lang, model, api_key):
                    chunks.append(chunk)
                    live.update(Markdown(''.join(chunks)))
        os.environ['LESS'] = '-R'
        with CONSOLE.pager(styles=True):
            CONSOLE.print(Markdown(''.join(chunks)))

    return [ExtraOption(EXPLAIN_TITLE, do_explain)]


def format_user_diff(user_text: str, segments: list) -> Text:
    """Format the user's text with changed/deleted parts highlighted in red."""
    result = Text()
    last_old_end = 0
    for seg in segments:
        if seg.old_text is not None:
            result.append(user_text[last_old_end:seg.old_start])
            result.append(seg.old_text, style='red bold')
            last_old_end = seg.old_end
        elif seg.new_text is not None:
            result.append(user_text[last_old_end:seg.old_start])
            last_old_end = seg.old_start
    result.append(user_text[last_old_end:])
    return result


def format_corrected_diff(corrected_text: str, segments: list) -> Text:
    """Format the corrected text with changed/inserted parts highlighted in green."""
    result = Text()
    last_new_end = 0
    for seg in segments:
        if seg.new_text is not None:
            result.append(corrected_text[last_new_end:seg.new_start])
            result.append(seg.new_text, style='green bold')
            last_new_end = seg.new_end
        elif seg.old_text is not None:
            result.append(corrected_text[last_new_end:seg.new_start])
            last_new_end = seg.new_start
    result.append(corrected_text[last_new_end:])
    return result


@app.command()
def study(
        language: Annotated[
            str,
            Argument(help='Target language code previously initialized with `crossrs init`.'),
        ],
        source: Annotated[
            str,
            Argument(help='Source language code to translate into (e.g., "en").'),
        ],
        threshold: Annotated[
            int,
            Option(
                '--threshold', '-t',
                help='Learnedness threshold for words to be considered fully learned.',
            ),
        ] = DEFAULT_THRESHOLD,
        model: Annotated[
            str,
            Option(
                '--model',
                envvar='CROSSRS_MODEL',
                help='GPT model for translation and evaluation.',
            ),
        ] = 'gpt-4.1-mini',
        api_key: Annotated[
            str,
            Option(
                '--api-key',
                envvar='CROSSRS_API_KEY',
                help='OpenAI API key.',
            ),
        ] = '',
) -> None:
    """Launch an interactive study session."""
    if not api_key:
        from crossrs.utils.typer import typer_raise
        typer_raise('API key is required. Set --api-key or CROSSRS_API_KEY environment variable.')

    with get_session(language) as session:
        sentence: Sentence
        target_word: Word | None
        source_translation: str

        # Background preparation results (IDs to be merged into main session)
        _bg_sentence_id: int
        _bg_target_word_id: int | None
        _bg_error: BaseException | None = None

        def prepare_next() -> None:
            """Choose the next sentence and translate it in a separate session."""
            nonlocal _bg_sentence_id, _bg_target_word_id, source_translation, _bg_error
            _bg_error = None
            try:
                with get_session(language) as bg_session:
                    bg_sentence, bg_target_word, _ = choose_next(bg_session, threshold)
                    _bg_sentence_id = bg_sentence.id
                    _bg_target_word_id = bg_target_word.id if bg_target_word else None
                    source_translation = translate_to_source(
                        bg_sentence, language, source, model, api_key,
                    )
            except BaseException as e:
                _bg_error = e

        def load_prepared() -> None:
            """Load the background-prepared sentence into the main session."""
            nonlocal sentence, target_word
            if _bg_error is not None:
                raise _bg_error
            session.expire_all()
            sentence = session.get(Sentence, _bg_sentence_id)
            target_word = session.get(Word, _bg_target_word_id) if _bg_target_word_id else None

        CONSOLE.print(Text('Preparing first sentence...', style='dim'))
        prepare_next()
        load_prepared()
        clear_previous()

        bg_ready: threading.Event | None = None

        while True:
            # Wait for background preparation if it hasn't finished yet
            if bg_ready is not None and not bg_ready.is_set():
                CONSOLE.print(PREPARING_NEXT)
                bg_ready.wait()
                clear_previous()
            if bg_ready is not None:
                try:
                    load_prepared()
                except Exception as e:
                    CONSOLE.print(Text(f'Error: {e}', style='red'))
                    CONSOLE.print(Text('Retrying...', style='dim'))
                    prepare_next()
                    load_prepared()
                    clear_previous(2)

            is_first_attempt = sentence.status == 0

            # 1) Show round status (always fresh)
            counts = get_round_counts(session)
            status_parts: list[Text] = []
            if counts.due > 0:
                status_parts.append(Text(f'{counts.due} due', style='yellow bold'))
            if counts.in_queue > 0:
                status_parts.append(Text(f'{counts.in_queue} in queue', style='dim'))
            if counts.learned > 0:
                status_parts.append(Text(f'{counts.learned} learned', style='green'))
            if status_parts:
                line = status_parts[0]
                for part in status_parts[1:]:
                    line = line + Text(' · ') + part
                CONSOLE.print(line)

            # 2) Show source translation and get user input
            formatted_prompt = Text(source_translation, style='bold')
            user_translation = ask(CONSOLE, formatted_prompt, QUIT_OPTION)
            clear_previous(2)
            CONSOLE.print(formatted_prompt)
            CONSOLE.print(user_translation)

            # 3) Evaluate
            CONSOLE.print(EVALUATION_IN_PROGRESS)
            try:
                evaluation = evaluate(
                    sentence, source_translation, user_translation,
                    language, source, model, api_key,
                )
            except Exception as e:
                clear_previous()
                CONSOLE.print(Text(f'Error: {e}', style='red'))
                CONSOLE.print(Text('Press Enter to retry...', style='dim'))
                input()
                clear_previous(4)
                continue
            clear_previous()

            # 4) Show result
            CONSOLE.print(SECTION_DELIMITER)
            if evaluation.is_correct:
                CONSOLE.print(Text(f'{CORRECT_MARK} ', style='') + Text(user_translation))
            else:
                CONSOLE.print(Text(f'{INCORRECT_MARK} '))
                if evaluation.corrected_translation:
                    segments = compute_diff(user_translation, evaluation.corrected_translation)
                    if segments:
                        CONSOLE.print(format_user_diff(user_translation, segments))
                        CONSOLE.print(format_corrected_diff(evaluation.corrected_translation, segments))
                    else:
                        CONSOLE.print(Text(evaluation.corrected_translation, style='bold'))
                CONSOLE.print(
                    Text('Original: ', style='dim')
                    + Text(sentence.sentence, style='bold italic'),
                )
            CONSOLE.print(SECTION_DELIMITER)

            # 5) Update (save state first so we can undo if user marks correct)
            saved_status = sentence.status
            saved_stage = sentence.review_stage
            saved_due = sentence.due
            saved_rounds = sentence.rounds
            update_sentence(sentence, evaluation.is_correct, is_first_attempt,
                            target_word, user_translation, session)
            session.commit()

            # Save reference to the sentence shown this round
            shown_sentence = sentence

            # 6) Start background preparation, then show continue prompt
            event = threading.Event()
            bg_ready = event
            threading.Thread(
                target=lambda e=event: (prepare_next(), e.set()),
                daemon=True,
            ).start()

            removed = False
            marked_correct = False

            def do_remove() -> None:
                nonlocal removed
                removed = True

            def do_mark_correct() -> None:
                nonlocal marked_correct
                marked_correct = True

            explain_option = (
                build_explanation_option(source_translation, user_translation,
                                        language, source, model, api_key)
                if not evaluation.is_correct else []
            )
            mark_correct_option = (
                [ExtraOption(MARK_CORRECT_TITLE, do_mark_correct, key='MC', returns=True)]
                if not evaluation.is_correct else []
            )
            remove_option = [ExtraOption(REMOVE_TITLE, do_remove, key='RS', returns=True)]
            ask(CONSOLE, ENTER_TO_CONTINUE,
                explain_option + mark_correct_option + remove_option + QUIT_OPTION)

            if marked_correct:
                # Undo incorrect update and redo as correct
                shown_sentence.status = saved_status
                shown_sentence.review_stage = saved_stage
                shown_sentence.due = saved_due
                shown_sentence.rounds = saved_rounds
                update_sentence(shown_sentence, True, is_first_attempt,
                                target_word, user_translation, session)
                session.commit()
                bg_ready.wait()
                clear_previous(2)
                CONSOLE.print(Text(f'{CORRECT_MARK}  Marked as correct.', style='dim'))
                continue

            if removed:
                undo_learned(shown_sentence, session)
                session.delete(shown_sentence)
                session.commit()
                bg_ready.wait()
                clear_previous(2)
                CONSOLE.print(Text('Sentence removed.', style='dim'))
                continue

            clear_previous(2)
            CONSOLE.print(SECTION_DELIMITER)
