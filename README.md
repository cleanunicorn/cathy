# cathy

Turn any text file into narrated audio, fully locally. No cloud, no API keys —
synthesis runs on your own GPU (or CPU). Four engines are supported, with
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) as the default: it
ranks near the top of the TTS Arena leaderboard and is fast enough to narrate
a whole book in minutes.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- `espeak-ng` (phonemizer backend): `sudo apt install espeak-ng`
- `ffmpeg` (only for non-wav output formats)
- `sox` (only for the qwen engine): `sudo apt install sox`
- NVIDIA GPU recommended (~2–3 GB VRAM for Kokoro); falls back to CPU

## Install and run

There are two ways to get cathy; pick one.

### Option 1: global install from GitHub

```sh
uv tool install --python 3.12 git+https://github.com/cleanunicorn/cathy
```

That puts a `cathy` command on your PATH, runnable from any directory. All
engines are available immediately:

```sh
cathy book.mobi -o book.m4b          # kokoro, the default
cathy book.txt -e qwen               # any other engine just works
```

The non-default engines live in separate environments (their dependency pins
conflict), which cathy manages for you through uv: the first `-e qwen` /
`-e chatterbox` / `-e fish` run downloads that engine's environment (a few
GB), and later runs reuse it. To pre-download instead of waiting on first
use:

```sh
cathy setup              # all three optional engines
cathy setup fish         # just one
```

Upgrade later with `uv tool upgrade cathy`; remove with `uv tool uninstall cathy`.

### Windows and GPU

On Windows, torch from PyPI is **CPU-only** — installing the normal way
narrates on CPU. Point uv at the CUDA wheel index when installing:

```sh
uv tool install --python 3.12 --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match git+https://github.com/cleanunicorn/cathy
```

A git clone (Option 2) needs no extra flags on Windows — the repo already
selects the CUDA index for Windows. Linux installs are unaffected either way;
PyPI's Linux wheels bundle CUDA. You'll also need Windows builds of
`espeak-ng` ([release installer](https://github.com/espeak-ng/espeak-ng/releases))
and `ffmpeg` (`winget install ffmpeg`).

### Option 2: from a git clone

No install step — `uv run` creates the environment on first use:

```sh
git clone https://github.com/cleanunicorn/cathy
cd cathy
uv run cathy book.mobi -o book.m4b
```

From a clone, non-default engines are selected per run with `--extra`:

```sh
uv run --extra qwen cathy book.txt -e qwen
```

(Running `-e qwen` without the `--extra` also works — cathy falls back to a
delegated environment as in the global install, fetched from GitHub unless
`CATHY_SOURCE` points elsewhere, e.g. `CATHY_SOURCE=$PWD` for local code.)

Note the difference: an installed tool is invoked as `cathy ...` from
anywhere, a clone as `uv run cathy ...` from inside the repo. All flags below
work identically in both.

## Usage

```sh
cathy book.txt                    # writes book.wav
cathy book.mobi -o book.m4b       # audiobook with chapter markers
cathy book.txt -o book.mp3        # any format ffmpeg understands
cathy book.txt -v female          # or male (default), british-male, ...
cathy book.txt -v af_heart:2,af_bella:1   # blend voices (weighted)
cathy book.txt -s 1.2             # 20% faster; native on kokoro, ffmpeg atempo elsewhere
cathy book.epub --dry-run         # preview chapters, skips, and duration estimates
cathy book.epub --chapters 3-10   # narrate part of the book
cathy book.epub -e qwen --max-chunk-chars 1000  # bigger chunks: smoother prosody, more VRAM
cathy --list-voices
cathy convert book.wav book.m4b   # convert existing audio, no re-narration
cathy convert book.wav fast.m4b -s 1.25   # re-time while converting
```

Ebook narration carries the book's title, author, and cover art into
`.m4b`/`.m4a` output, so audiobook players show the real book, not a blank
tile.

Narrating to `.wav` also writes `book.chapters.txt` (chapters plus title and
author tags) and `book.cover.jpg`/`.png` next to it, so a later
`cathy convert book.wav book.m4b` still embeds everything. Convert works for
any ffmpeg-supported output (mp3, flac, ...).

The first run downloads the model (~330 MB) from Hugging Face; everything after
that is offline.

Input can be plain text (`.txt`, `.md`, …) or an ebook (`.mobi`, `.azw`,
`.azw3`, `.epub`) — ebooks are unpacked in memory and their chapters narrated
in reading order. Paragraphs (blank-line separated) are
narrated with a short pause between them, and audio is streamed to disk as it
is generated, so book-length inputs won't exhaust memory.

Chapters are detected from the epub spine (ebooks), h1/h2 headings (HTML), or
markdown `#`/`##` headings (plain text). Writing to `.m4b` or `.m4a` embeds
them as real chapter markers, so audiobook players show a chapter list and
remember your position. Other formats (mp3, wav) get one continuous stream.

Narration checkpoints each finished chapter into `<output>.partial/`. If a
long run is interrupted — Ctrl-C, crash, out-of-memory — rerun the same
command and it resumes from the last finished chapter (the directory is
removed once the output is written). Changing the engine, voice, speed, or
text invalidates the affected checkpoints automatically.

Front and back matter that makes no sense read aloud — title page, table of
contents, copyright page, index, endnote collections, "also by"/"praise
for" lists — is skipped
automatically (cathy prints what it skipped); pass `--all` to narrate
everything. Text is also cleaned up for narration: footnote markers, markdown
formatting, scene-separator glyphs, and stray page numbers are dropped, and
URLs are read as their bare domain.

## Engines

Pick an engine with `-e`; the first use of each downloads its environment
(pre-download with `cathy setup`):

```sh
cathy book.mobi -o book.m4b                  # kokoro (default)
cathy book.txt -e qwen -v Ryan
cathy book.txt -e chatterbox -v me.wav
cathy book.txt -e fish -v female
```

| Engine | Model | Speed (RTX 3080) | Voices | License |
|---|---|---|---|---|
| `kokoro` | Kokoro-82M | ~30× real time | presets + weighted blends | Apache-2.0 |
| `qwen` | Qwen3-TTS-0.6B-CustomVoice | ~real time | 9 preset speakers | Apache-2.0 |
| `chatterbox` | Chatterbox-Nano (110M) | ~10× real time | clones a reference .wav | MIT |
| `fish` | OpenAudio S1-mini (0.5B) | ~real time | clones a reference .wav | CC-BY-NC-SA (non-commercial) |

The fish model (`fishaudio/s1-mini`) is gated on Hugging Face: accept its
terms on the model page, then set `HF_TOKEN` to a
[token from your free account](https://huggingface.co/settings/tokens)
before first use.

`-v male` / `-v female` works on every engine. For the cloning engines
(chatterbox, fish) those generate a reference clip with Kokoro; pass a path to
a ~10 s `.wav` of any speaker to clone that voice instead.

Languages other than English: Kokoro ships Spanish, French, Italian,
Portuguese, and Hindi voices — the voice prefix picks the language
(`cathy libro.epub -v ef_dora`; see `cathy voices`). Qwen reads many
languages with any speaker: pick one with `-l` (`cathy livre.txt -e qwen -l
French`). Kokoro's Japanese/Chinese presets need the `misaki[ja]`/`misaki[zh]`
extras, which cathy doesn't install by default.

Kokoro is the right default for whole books; the others trade speed for
expressiveness or cloning. Each engine environment is a few GB on disk (uv
caches them; `uv cache clean` reclaims the space).

## Troubleshooting

**`Device: cpu` on a machine with an NVIDIA GPU** — the environment has
CPU-only torch (the PyPI default on Windows). Reinstall with the CUDA index
(see "Windows and GPU" above), then run `cathy setup` — cached engine
environments keep their old torch until setup rebuilds them.

**fish fails with `401`/gated-repo errors** — `fishaudio/s1-mini` requires a
free authenticated Hugging Face account: accept the terms on the
[model page](https://huggingface.co/fishaudio/s1-mini), create a
[token](https://huggingface.co/settings/tokens), and set it as the `HF_TOKEN`
environment variable (`export HF_TOKEN=...`; on Windows
`setx HF_TOKEN ...` and open a new terminal).

**`sox: not found` with the qwen engine** — install sox:
`sudo apt install sox` / `winget install sox`.

**CUDA out-of-memory** — qwen and fish each need ~6 GB VRAM; run one
narration at a time on a 10 GB card, and close other GPU-heavy apps. If it
still runs out, lower `--max-chunk-chars`: it caps how much text goes to the
engine in one request, which is what drives peak VRAM. Cathy halves the cap
itself and retries when the GPU runs out, so a run survives an occasional
oversized paragraph; setting it lower up front just avoids the retry.

**Engines still behave like an older cathy after upgrading** — run
`cathy setup` after `uv tool upgrade cathy`; the per-engine environments are
cached and only setup refreshes them.

## Development

```sh
git clone https://github.com/cleanunicorn/cathy
cd cathy
uv sync                      # creates .venv with dev dependencies
uv run pytest -q             # 49 tests, ~1 s
uv run cathy book.txt        # run your working copy
```

`uv sync` pins Python 3.12 (`.python-version`) and installs the dev group
(pytest). The tests cover text extraction, normalization, chunking, and audio
plumbing only — no model downloads, no GPU.

### Install your working copy as the `cathy` command

To use local changes from any directory (replaces a GitHub-installed cathy):

```sh
uv tool install --python 3.12 --editable .
```

Editable means source edits take effect immediately, no reinstall. Go back to
the published version with:

```sh
uv tool install --python 3.12 --force git+https://github.com/cleanunicorn/cathy
```

### Non-default engines against local code

The optional engines run in separate uv environments. Two ways to point them
at your clone rather than GitHub:

```sh
uv run --extra qwen cathy book.txt -e qwen              # inside the repo
CATHY_SOURCE=/path/to/cathy cathy book.txt -e qwen      # from anywhere
```

Without `CATHY_SOURCE`, a delegated engine run fetches cathy from GitHub, so
your local changes would not apply to that engine.

### Layout

| Path | Contents |
|---|---|
| `src/cathy/cli.py` | CLI, ebook parsing, chapters, chunking, audio writing |
| `src/cathy/engines.py` | per-engine synthesis backends and voice handling |
| `src/cathy/normalize.py` | text cleanup for narration (footnotes, URLs, glyphs) |
| `tests/` | pytest suite, no GPU or network needed |

CI (`.github/workflows/ci.yml`) runs the same pytest suite on Linux with only
the lightweight dependencies installed, so keep new tests free of torch and
model imports.

## Notes

- torch is capped below 2.9 everywhere: newer PyPI wheels bundle CUDA 13
  libraries that hit a cuDNN version mismatch at inference time. Repo
  development additionally pins torch 2.8.0 via uv overrides.
- Kokoro voice prefixes select the language (`a` American, `b` British
  English). Blending averages style vectors — `af_heart:2,af_bella:1` is two
  parts heart, one part bella — letting you tune a narrator between presets.
- Cadence is audiobook-style: segment edges are trimmed of stray silence, then
  explicit pauses are inserted — 0.15 s between sentences of long paragraphs,
  0.55 s between paragraphs, 1 s after a chapter heading.
- fish-speech and chatterbox are installed from pinned git commits: the last
  S1-era fish-speech commit (S2 needs 24 GB VRAM), and the Chatterbox-Nano
  release (absent from PyPI).
