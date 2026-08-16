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
# Graded pauses, audiobook-style: short between sentences (Kokoro chunk
# boundaries fall on sentence ends), longer between paragraphs, longest after
# a chapter heading.
SENTENCE_PAUSE = 0.15
PARAGRAPH_PAUSE = 0.55
HEADING_PAUSE = 1.0
EDGE_KEEP_SECONDS = 0.05  # padding kept when trimming silence off segment edges
CHAPTER_FORMATS = (".m4b", ".m4a")  # containers that get chapter markers

# A chapter is (title, paragraphs). Titles become m4b chapter markers.
Chapter = tuple[str, list[str]]


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
        help="output audio file (.wav, .mp3, .m4b, ...); default: input name + .wav. "
        ".m4b/.m4a outputs include chapter markers",
    )
    parser.add_argument(
        "-v",
        "--voice",
        default="af_heart",
        help="voice name, or a blend like 'af_heart:2,af_bella:1' "
        "(see --list-voices); default: af_heart",
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


HEADING_TAGS = ["h1", "h2", "h3", "h4"]
BLOCK_TAGS = ["p", "li", "blockquote", *HEADING_TAGS]


def html_blocks(html: str):
    """Return top-level readable block elements from an HTML document."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return [b for b in soup.find_all(BLOCK_TAGS) if b.find_parent(BLOCK_TAGS) is None]


def blocks_to_paragraphs(blocks) -> list[str]:
    paragraphs = [b.get_text(" ", strip=True) for b in blocks]
    return [p for p in paragraphs if p]


def html_to_chapters(html: str) -> list[Chapter]:
    """Split a single HTML document into chapters at h1/h2 headings."""
    chapters: list[Chapter] = []
    title, current = "", []
    for block in html_blocks(html):
        if block.name in ("h1", "h2"):
            if current:
                chapters.append((title, current))
            title, current = block.get_text(" ", strip=True), []
        text = block.get_text(" ", strip=True)
        if text:
            current.append(text)
    if current:
        chapters.append((title, current))
    return chapters


def epub_to_chapters(path: Path) -> list[Chapter]:
    """Extract chapters from an epub, following the book's spine order."""
    from ebooklib import ITEM_DOCUMENT, epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    chapters: list[Chapter] = []
    for spine_id, _ in book.spine:
        item = book.get_item_with_id(spine_id)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        blocks = html_blocks(item.get_content().decode("utf-8", "ignore"))
        paragraphs = blocks_to_paragraphs(blocks)
        if not paragraphs:
            continue
        heading = next((b for b in blocks if b.name in HEADING_TAGS), None)
        title = heading.get_text(" ", strip=True) if heading else ""
        chapters.append((title, paragraphs))
    return chapters


def mobi_to_chapters(path: Path) -> list[Chapter]:
    """Unpack a mobi/azw file and extract chapters from whatever is inside."""
    import shutil

    import mobi

    tempdir, extracted = mobi.extract(str(path))
    try:
        extracted = Path(extracted)
        if extracted.suffix.lower() == ".epub":
            return epub_to_chapters(extracted)
        if extracted.suffix.lower() in (".html", ".htm"):
            return html_to_chapters(
                extracted.read_text(encoding="utf-8", errors="ignore")
            )
        return text_to_chapters(extracted.read_text(encoding="utf-8", errors="ignore"))
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def text_to_chapters(text: str) -> list[Chapter]:
    """Split plain text/markdown into chapters at markdown h1/h2 headings."""
    chapters: list[Chapter] = []
    title, current = "", []
    for raw in text.split("\n\n"):
        paragraph = raw.strip()
        if not paragraph:
            continue
        if paragraph.startswith(("# ", "## ")) and "\n" not in paragraph:
            if current:
                chapters.append((title, current))
            title, current = paragraph.lstrip("# "), []
            current.append(title)
        else:
            current.append(paragraph.replace("\n", " "))
    if current:
        chapters.append((title, current))
    return chapters


def load_chapters(path: Path) -> list[Chapter]:
    """Read a text/ebook file and return its chapters."""
    suffix = path.suffix.lower()
    if suffix in (".mobi", ".azw", ".azw3"):
        chapters = mobi_to_chapters(path)
    elif suffix == ".epub":
        chapters = epub_to_chapters(path)
    else:
        chapters = text_to_chapters(path.read_text(encoding="utf-8"))
    return [
        (title or f"Chapter {i}", paragraphs)
        for i, (title, paragraphs) in enumerate(chapters, start=1)
    ]


def resolve_voice(pipeline, spec: str):
    """Resolve a voice spec, supporting weighted blends like 'af_heart:2,af_bella:1'.

    Plain names and unweighted blends ('af_heart,af_bella') are passed through
    to Kokoro, which averages them; weights need the tensor mix done here.
    """
    if ":" not in spec:
        return spec

    import torch

    names, weights = [], []
    for part in spec.split(","):
        name, _, weight = part.partition(":")
        names.append(name.strip())
        weights.append(float(weight or 1))
    packs = torch.stack([pipeline.load_single_voice(n) for n in names])
    w = torch.tensor(weights, dtype=packs.dtype) / sum(weights)
    return torch.sum(packs * w.reshape(-1, *[1] * (packs.dim() - 1)), dim=0)


def trim_edges(audio, threshold: float = 2e-3):
    """Trim near-silence off both ends of a segment, keeping a little padding.

    Kokoro segments carry variable amounts of edge silence; trimming it and
    inserting explicit pauses gives the narration a uniform rhythm.
    """
    import numpy as np

    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    keep = int(SAMPLE_RATE * EDGE_KEEP_SECONDS)
    return audio[max(0, loud[0] - keep) : min(len(audio), loud[-1] + keep)]


def narrate(
    chapters: list[Chapter], voice: str, speed: float, device: str | None, wav_path: Path
) -> list[tuple[str, float, float]]:
    """Synthesize all chapters into wav_path; return (title, start_s, end_s) marks."""
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline
    from tqdm import tqdm

    lang_code = voice[0]  # Kokoro convention: voice prefix selects the language
    pipeline = KPipeline(
        lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=device
    )
    voice = resolve_voice(pipeline, voice)

    def silence(seconds: float):
        return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)

    marks = []
    written = 0
    total = sum(len(paragraphs) for _, paragraphs in chapters)
    with (
        sf.SoundFile(
            wav_path, mode="w", samplerate=SAMPLE_RATE, channels=1, format="WAV"
        ) as f,
        tqdm(total=total, unit="para", desc="Narrating") as progress,
    ):
        def emit(chunk) -> None:
            nonlocal written
            f.write(chunk)
            written += len(chunk)

        for title, paragraphs in chapters:
            start = written
            for i, paragraph in enumerate(paragraphs):
                first = True
                for _, _, audio in pipeline(paragraph, voice=voice, speed=speed):
                    if not first:
                        emit(silence(SENTENCE_PAUSE))
                    emit(trim_edges(audio.numpy()))
                    first = False
                # A chapter's first paragraph is usually its heading; give it
                # the longer, audiobook-style pause.
                is_heading = i == 0 and paragraph == title
                emit(silence(HEADING_PAUSE if is_heading else PARAGRAPH_PAUSE))
                progress.update(1)
            marks.append((title, start / SAMPLE_RATE, written / SAMPLE_RATE))
    return marks


def ffmetadata(marks: list[tuple[str, float, float]]) -> str:
    """Render chapter marks as an ffmpeg metadata file."""
    lines = [";FFMETADATA1"]
    for title, start, end in marks:
        escaped = "".join("\\" + c if c in "=;#\\\n" else c for c in title)
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={int(start * 1000)}",
            f"END={int(end * 1000)}",
            f"title={escaped}",
        ]
    return "\n".join(lines) + "\n"


def convert(wav_path: Path, output: Path, marks: list[tuple[str, float, float]]) -> None:
    """Convert the intermediate wav to the requested container via ffmpeg."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path)]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete_on_close=False) as meta:
        if output.suffix.lower() in CHAPTER_FORMATS:
            meta.write(ffmetadata(marks))
            meta.close()
            cmd += ["-f", "ffmetadata", "-i", meta.name, "-map_metadata", "1"]
            cmd += ["-map_chapters", "1", "-c:a", "aac", "-b:a", "96k", "-f", "ipod"]
        cmd.append(str(output))
        subprocess.run(cmd, check=True)


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
    chapters = load_chapters(args.input)
    if not chapters:
        sys.exit("error: input file contains no text")

    paragraph_count = sum(len(p) for _, p in chapters)
    print(
        f"Device: {device} | Voice: {args.voice} | "
        f"Chapters: {len(chapters)} | Paragraphs: {paragraph_count}"
    )

    if output.suffix.lower() == ".wav":
        narrate(chapters, args.voice, args.speed, device, output)
    else:
        wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            marks = narrate(chapters, args.voice, args.speed, device, wav_path)
            convert(wav_path, output, marks)
        finally:
            wav_path.unlink(missing_ok=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
