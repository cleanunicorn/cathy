"""Command-line interface: text file in, narrated audio out."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# Curated subset of Kokoro's built-in voices. Prefix encodes language/gender:
# a=American, b=British English; f=female, m=male.
VOICES = {
    "af_heart": "American female (highest rated)",
    "af_bella": "American female, warm",
    "af_nicole": "American female, soft-spoken",
    "am_michael": "American male",
    "am_fenrir": "American male, deep",
    "bf_emma": "British female",
    "bm_george": "British male",
    "bm_fable": "British male, narrator",
}

SAMPLE_RATE = 24_000
PAUSE_SECONDS = 0.35  # silence inserted between paragraphs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cathy",
        description="Turn a text file into speech, fully locally (GPU-accelerated).",
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="input .txt/.md/.mobi/.azw3/.epub file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output audio file (.wav, .mp3, .flac, ...); default: input name + .wav",
    )
    parser.add_argument(
        "-v",
        "--voice",
        default="af_heart",
        help="voice name (see --list-voices); default: af_heart",
    )
    parser.add_argument(
        "-s", "--speed", type=float, default=1.0, help="speech speed; default: 1.0"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="force CPU even if a GPU is available"
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="list available voices and exit"
    )
    args = parser.parse_args(argv)
    if not args.list_voices and args.input is None:
        parser.error("input file is required (or use --list-voices)")
    return args


BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "li", "blockquote"]


def html_to_text(html: str) -> str:
    """Extract readable text from HTML as blank-line separated paragraphs."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    blocks = [
        b for b in soup.find_all(BLOCK_TAGS) if b.find_parent(BLOCK_TAGS) is None
    ]
    if blocks:
        return "\n\n".join(b.get_text(" ", strip=True) for b in blocks)
    return soup.get_text("\n\n", strip=True)


def epub_to_text(path: Path) -> str:
    """Extract text from an epub, following the book's spine order."""
    from ebooklib import ITEM_DOCUMENT, epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    chapters = []
    for spine_id, _ in book.spine:
        item = book.get_item_with_id(spine_id)
        if item is not None and item.get_type() == ITEM_DOCUMENT:
            chapters.append(html_to_text(item.get_content().decode("utf-8", "ignore")))
    return "\n\n".join(chapters)


def mobi_to_text(path: Path) -> str:
    """Unpack a mobi/azw file and extract text from whatever is inside."""
    import shutil

    import mobi

    tempdir, extracted = mobi.extract(str(path))
    try:
        extracted = Path(extracted)
        if extracted.suffix.lower() == ".epub":
            return epub_to_text(extracted)
        if extracted.suffix.lower() in (".html", ".htm"):
            return html_to_text(extracted.read_text(encoding="utf-8", errors="ignore"))
        return extracted.read_text(encoding="utf-8", errors="ignore")
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def load_text(path: Path) -> list[str]:
    """Read a text/ebook file and return non-empty paragraphs."""
    suffix = path.suffix.lower()
    if suffix in (".mobi", ".azw", ".azw3"):
        text = mobi_to_text(path)
    elif suffix == ".epub":
        text = epub_to_text(path)
    else:
        text = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p.replace("\n", " ") for p in paragraphs if p]


def synthesize(
    paragraphs: list[str], voice: str, speed: float, device: str | None
):
    """Yield one audio array per synthesized segment."""
    import numpy as np
    from kokoro import KPipeline

    lang_code = voice[0]  # Kokoro convention: voice prefix selects the language
    pipeline = KPipeline(lang_code=lang_code, device=device)
    pause = np.zeros(int(SAMPLE_RATE * PAUSE_SECONDS), dtype=np.float32)

    from tqdm import tqdm

    for paragraph in tqdm(paragraphs, unit="para", desc="Narrating"):
        for _, _, audio in pipeline(paragraph, voice=voice, speed=speed):
            yield audio.numpy()
        yield pause


def write_output(chunks, output: Path) -> None:
    """Stream audio chunks to a wav file, converting via ffmpeg if needed."""
    import soundfile as sf

    if output.suffix.lower() == ".wav":
        wav_path = output
    else:
        wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])

    with sf.SoundFile(
        wav_path, mode="w", samplerate=SAMPLE_RATE, channels=1, format="WAV"
    ) as f:
        for chunk in chunks:
            f.write(chunk)

    if wav_path != output:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), str(output)],
            check=True,
        )
        wav_path.unlink()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_voices:
        for name, description in VOICES.items():
            print(f"  {name:12} {description}")
        return

    if not args.input.exists():
        sys.exit(f"error: {args.input} not found")

    import torch

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output or args.input.with_suffix(".wav")
    paragraphs = load_text(args.input)
    if not paragraphs:
        sys.exit("error: input file contains no text")

    print(f"Device: {device} | Voice: {args.voice} | Paragraphs: {len(paragraphs)}")
    chunks = synthesize(paragraphs, args.voice, args.speed, device)
    write_output(chunks, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
