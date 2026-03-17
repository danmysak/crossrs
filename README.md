# CrossRS

CrossRS is a command-line tool for improving language **production** skills through reverse translation exercises. Given a corpus in your target language, CrossRS translates sentences into a source language you already know and asks you to translate them back, reinforcing vocabulary and grammar through word-based spaced repetition. Because CrossRS uses GPT under the hood, you must have a paid [OpenAI account](https://platform.openai.com/) and an [API key](https://platform.openai.com/api-keys) to run it.

## How It Works

CrossRS focuses on **words** sorted by their frequency in the corpus. You learn the most common ones first. Each study round:

1. CrossRS picks a sentence containing the next word to learn.
2. The sentence is translated into your source language and shown to you.
3. You translate it back into the target language.
4. CrossRS evaluates your translation and provides feedback — either a ✅ confirmation or a ❌ with a highlighted diff showing the minimal corrections needed.

Sentences you translate correctly on the first try are scheduled for a single review in **29 days 20 hours**. Otherwise, they enter a spaced-repetition queue with reviews at **20 hours**, **6 days 20 hours**, and **29 days 20 hours** before being marked as learned.

## Installation

Install [Python](https://www.python.org/downloads/) **3.13 or later** and [pipx](https://pipx.pypa.io/stable/installation/), then run:

```bash
pipx install crossrs       # install
pipx upgrade crossrs       # upgrade
pipx uninstall crossrs     # uninstall
```

## Initialize a New Language

Prepare a plain-text file that contains **one sentence per line** in the language you want to learn. For example, you can download a monolingual corpus from [OPUS](https://opus.nlpl.eu/). Then run:

```bash
crossrs init <target-lang> <corpus>
```

`<target-lang>` is a language code (e.g., `de`, `fr`, `uk`), and `<corpus>` is the path to the corpus file.

## Study a Language

```bash
crossrs study <target-lang> <source-lang> [--threshold T] [--model <GPT_MODEL>] [--api-key <OPENAI_KEY>] [--listen] [--asr-model <ASR_MODEL>]
```

* **`<target-lang>`** — the language code you initialized earlier.
* **`<source-lang>`** — the language you want sentences translated into (e.g., `en`).
* **`--threshold` / `-t`** — the learnedness threshold for words (default: 3). A word is considered fully learned once it has appeared in this many learned sentences.
* **`--model`** — the GPT model to use for translation and evaluation.
* **`--api-key`** — your OpenAI API key.
* **`--listen`** — enable voice input: press Enter to record your translation via microphone instead of typing.
* **`--asr-model`** — the ASR model for speech-to-text (e.g., `gpt-4o-transcribe`). Required when `--listen` is used.

Instead of passing `--model`, `--api-key`, and `--asr-model` each time, you can set the environment variables `CROSSRS_MODEL`, `CROSSRS_API_KEY`, and `CROSSRS_ASR_MODEL`.

### Voice Input

When `--listen` is enabled, pressing Enter at the translation prompt starts recording. Speak your translation and press Enter again to stop. The recording is transcribed and submitted automatically. You can also type your translation directly — only an empty Enter triggers recording.

Voice input requires the `audio` extra:

```bash
pipx install crossrs[audio]   # or: pip install crossrs[audio]
```

## View Your Progress

```bash
crossrs stats <target-lang> [--threshold T]
```

Displays:
- **Sentences**: learned + in queue / total
- **Words**: learned / total, with word-level coverage
- **Total rounds**: the number of translation attempts so far

## Other Commands

```bash
crossrs path <target-lang>               # show the path to the language data file
crossrs delete <target-lang> [--force]   # delete the language data file; use --force to skip the confirmation prompt
```
