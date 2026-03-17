from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import mkstemp
from typing import Callable, Generator, Mapping, TYPE_CHECKING
import re
import wave

from openai import OpenAI
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.text import Text

from crossrs.utils.console import clear_previous
from crossrs.utils.openai import api_call_with_retries

if TYPE_CHECKING:
    from pyaudio import PyAudio, Stream

__all__ = [
    'build_voice_input',
    'ensure_audio',
]

# ─── Recording parameters ───────────────────────────────────────────────────────

FILE_EXTENSION = '.wav'
CHUNK_SIZE = 1024
RECORD_CHANNELS = 1
RECORD_RATE = 16000
RECORD_BITS_PER_SAMPLE = 16

RECORDING_PROMPT = Text('🔴 Recording... (Enter to stop)', style='red')
TRANSCRIBING_PROMPT = Text('Transcribing...', style='dim')
RERECORD_KEY = 'r'


# ─── PyAudio helpers ────────────────────────────────────────────────────────────

@dataclass
class _PyAudioData:
    audio: PyAudio
    paInt16: int
    paContinue: int
    paFramesPerBufferUnspecified: int


@contextmanager
def _get_audio() -> Generator[_PyAudioData, None, None]:
    import pyaudio
    audio = pyaudio.PyAudio()
    try:
        yield _PyAudioData(
            audio,
            paInt16=pyaudio.paInt16,
            paContinue=pyaudio.paContinue,
            paFramesPerBufferUnspecified=pyaudio.paFramesPerBufferUnspecified,
        )
    finally:
        audio.terminate()


@contextmanager
def _get_stream(
        pyaudio: _PyAudioData,
        fmt: int,
        channels: int,
        rate: int,
        is_input: bool = False,
        frames_per_buffer: int | None = None,
        stream_callback: Callable | None = None,
) -> Generator[Stream, None, None]:
    stream = pyaudio.audio.open(
        format=fmt,
        channels=channels,
        rate=rate,
        input=is_input,
        frames_per_buffer=frames_per_buffer or pyaudio.paFramesPerBufferUnspecified,
        stream_callback=stream_callback,
    )
    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


# ─── WAV recording ──────────────────────────────────────────────────────────────

def _record_callback(
        wav: wave.Wave_write,
        flag_to_return: int,
        data: bytes,
        _frame_count: int,
        _time_info: Mapping[str, float],
        _status_flags: int,
) -> tuple[None, int]:
    wav.writeframes(data)
    return None, flag_to_return


@contextmanager
def _record(file: Path) -> Generator[None, None, None]:
    wav = wave.open(str(file.absolute()), 'wb')
    wav.setnchannels(RECORD_CHANNELS)
    wav.setsampwidth(RECORD_BITS_PER_SAMPLE // 8)
    wav.setframerate(RECORD_RATE)
    try:
        with (
            _get_audio() as pyaudio,
            _get_stream(
                pyaudio=pyaudio,
                fmt=pyaudio.paInt16,
                channels=RECORD_CHANNELS,
                rate=RECORD_RATE,
                is_input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=partial(_record_callback, wav, pyaudio.paContinue),
            ),
        ):
            yield
    finally:
        wav.close()


# ─── Transcription ──────────────────────────────────────────────────────────────

def _normalize_transcription(text: str) -> str:
    cleaned = ''.join(
        char for char in text.strip()
        if not (0xE000 <= ord(char) <= 0xF8FF)
    )
    cleaned = re.sub(
        r"^[a-zA-Z\u00C0-\u024F]+(?=[A-Z][a-z'\']+([.?!,;:]|\s))|[¹²³⁴-⁹]",
        '',
        cleaned,
    )
    return cleaned.strip()


def _transcribe(file: Path, language: str, model: str, api_key: str) -> str:
    return _normalize_transcription(
        api_call_with_retries(lambda: OpenAI(api_key=api_key).audio.transcriptions.create(
            file=file.open('rb'),
            model=model,
            language=language,
        ).text)
    )


# ─── Public API ─────────────────────────────────────────────────────────────────

def ensure_audio() -> None:
    """Check that PyAudio is available."""
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        from crossrs.utils.typer import typer_raise
        typer_raise(
            'Audio dependencies are not installed. '
            'Install with: pip install crossrs[audio]'
        )


def build_voice_input(console: Console, language: str,
                      asr_model: str, api_key: str) -> Callable[[], str | None]:
    """Build a voice input callable for use with ask().

    Returns a function that records audio, transcribes it, and returns the text.
    Returns None if the user wants to re-record (caller should re-prompt).
    """
    def voice_input() -> str | None:
        file = Path(mkstemp(suffix=FILE_EXTENSION)[1])
        try:
            console.print(RECORDING_PROMPT)
            with _record(file):
                PromptSession().prompt()
            clear_previous(2)

            console.print(TRANSCRIBING_PROMPT)
            try:
                text = _transcribe(file, language, asr_model, api_key)
            except Exception as e:
                clear_previous()
                console.print(Text(f'Transcription error: {e}', style='red'))
                return None
            clear_previous()

            if not text:
                console.print(Text('No speech detected.', style='dim'))
                return None

            return text
        finally:
            try:
                Path(file).unlink(missing_ok=True)
            except OSError:
                pass

    return voice_input

