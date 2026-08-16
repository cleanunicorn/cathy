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
- NVIDIA GPU recommended (~2–3 GB VRAM for Kokoro); falls back to CPU

## Install

```sh
uv tool install git+https://github.com/cleanunicorn/cathy
```

That puts `cathy` on your PATH, runnable from anywhere. To use a non-default
engine, install with its extra instead (one engine per install — their
dependency pins conflict):

```sh
uv tool install --reinstall "cathy[qwen] @ git+https://github.com/cleanunicorn/cathy"
```

Upgrade later with `uv tool upgrade cathy`; developing from a clone of this
repo, use `uv run cathy ...` as shown below without installing.

## Usage

```sh
cathy book.txt                    # writes book.wav
cathy book.mobi -o book.m4b       # audiobook with chapter markers
cathy book.txt -o book.mp3        # any format ffmpeg understands
cathy book.txt -v female          # or male (default), british-male, ...
cathy book.txt -v af_heart:2,af_bella:1   # blend voices (weighted)
cathy book.txt -s 1.2             # 20% faster speech
cathy --list-voices
```

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

## Engines

Each non-default engine lives in its own dependency environment (their pinned
torch/transformers versions conflict). From a clone of this repo, select one
with `uv run --extra <name>`; for a global install, reinstall the tool with
the matching extra (see Install above):

```sh
uv run cathy book.mobi -o book.m4b                              # kokoro (default)
uv run --extra qwen cathy book.txt -e qwen -v Ryan
uv run --extra chatterbox cathy book.txt -e chatterbox -v me.wav
uv run --extra fish cathy book.txt -e fish -v female
```

| Engine | Model | Speed (RTX 3080) | Voices | License |
|---|---|---|---|---|
| `kokoro` | Kokoro-82M | ~30× real time | presets + weighted blends | Apache-2.0 |
| `qwen` | Qwen3-TTS-0.6B-CustomVoice | ~real time | 9 preset speakers | Apache-2.0 |
| `chatterbox` | Chatterbox-Nano (110M) | ~10× real time | clones a reference .wav | MIT |
| `fish` | OpenAudio S1-mini (0.5B) | ~real time | clones a reference .wav | CC-BY-NC-SA (non-commercial) |

`-v male` / `-v female` works on every engine. For the cloning engines
(chatterbox, fish) those generate a reference clip with Kokoro; pass a path to
a ~10 s `.wav` of any speaker to clone that voice instead.

Kokoro is the right default for whole books; the others trade speed for
expressiveness or cloning. Switching engines swaps the virtualenv (a few GB of
packages), so expect the first `--extra` run to take a while.

## Notes

- All environments share torch 2.8.0 from the CUDA 12.8 wheel index (pinned in
  `pyproject.toml`): the default cu130 build hits a cuDNN version mismatch,
  chatterbox pins torch 2.6, and fish predates 2.8 — one override keeps every
  engine on a working wheel.
- Kokoro voice prefixes select the language (`a` American, `b` British
  English). Blending averages style vectors — `af_heart:2,af_bella:1` is two
  parts heart, one part bella — letting you tune a narrator between presets.
- Cadence is audiobook-style: segment edges are trimmed of stray silence, then
  explicit pauses are inserted — 0.15 s between sentences of long paragraphs,
  0.55 s between paragraphs, 1 s after a chapter heading.
- fish-speech and chatterbox are installed from pinned git commits: the last
  S1-era fish-speech commit (S2 needs 24 GB VRAM), and the Chatterbox-Nano
  release (absent from PyPI).
