# cathy

Turn any text file into narrated audio, fully locally. No cloud, no API keys —
synthesis runs on your own GPU (or CPU) using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
an 82M-parameter model that ranks near the top of the TTS Arena leaderboard.

## Requirements

- Python 3.12+ managed via [uv](https://docs.astral.sh/uv/)
- `espeak-ng` (phonemizer backend): `sudo apt install espeak-ng`
- `ffmpeg` (only for non-wav output formats)
- NVIDIA GPU recommended (~2–3 GB VRAM); falls back to CPU automatically

## Usage

```sh
uv run cathy book.txt                    # writes book.wav
uv run cathy book.mobi -o book.m4b       # audiobook with chapter markers
uv run cathy book.txt -o book.mp3        # any format ffmpeg understands
uv run cathy book.txt -v bm_fable        # pick a narrator voice
uv run cathy book.txt -s 1.2             # 20% faster speech
uv run cathy --list-voices
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

## Performance

On an RTX 3080, synthesis runs at roughly 10–30× real time — about an hour of
audio in a few minutes.

## Notes

- Torch is pinned to the CUDA 12.8 wheel index in `pyproject.toml`; the
  default cu130 build hits a cuDNN version mismatch (as of torch 2.13).
- Voices are Kokoro's built-in presets; the first letter selects the language
  (`a` American, `b` British English). No voice cloning — if you need a custom
  voice, [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) is the
  natural companion model.
