"""Command-line interface: text file in, narrated audio out."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from cathy.engines import (
    ENGINES,
    KOKORO_ALIASES,
    KOKORO_VOICES,
    QWEN_SPEAKERS,
    engine_available,
)
from cathy.normalize import drop_front_back_matter, normalize_chapters

# Where delegated engine environments are installed from. Overridable so a
# development clone can point at itself: CATHY_SOURCE=/path/to/clone
CATHY_SOURCE = "git+https://github.com/cleanunicorn/cathy"

# Graded pauses, audiobook-style: short between sentences (engine chunk
# boundaries fall on sentence ends), longer between paragraphs, longest after
# a chapter heading.
SENTENCE_PAUSE = 0.15
PARAGRAPH_PAUSE = 0.55
HEADING_PAUSE = 1.0
EDGE_KEEP_SECONDS = 0.05  # padding kept when trimming silence off segment edges
CHAPTER_FORMATS = (".m4b", ".m4a")  # containers that get chapter markers
EBOOK_FORMATS = (".mobi", ".azw", ".azw3", ".epub")

# A chapter is (title, paragraphs). Titles become m4b chapter markers.
Chapter = tuple[str, list[str]]


EPILOG = """\
subcommands:
  convert IN OUT [-s SPEED]   convert existing audio without re-narrating
                              (keeps chapter markers via IN.chapters.txt)
  setup [ENGINE ...]          pre-download engine environments (qwen,
                              chatterbox, fish); also refreshes them after
                              an upgrade
  voices                      list all voices (same as --list-voices)
  help [SUBCOMMAND]           show this help, or a subcommand's

examples:
  cathy book.mobi -o book.m4b            audiobook with chapter markers
  cathy notes.txt -v female -s 1.2       pick a voice, 20% faster
  cathy book.epub -e fish -v sample.wav  clone the voice in sample.wav
  cathy convert book.wav book.m4b        re-container without re-narrating
"""


def cathy_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("cathy")
    except PackageNotFoundError:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cathy",
        description="Turn a text file or ebook into narrated audio, fully "
        "locally (GPU-accelerated).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="input .txt/.md/.mobi/.azw3/.epub file"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"cathy {cathy_version()}"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output audio file (.wav, .mp3, .m4b, ...); default: input name + .wav. "
        ".m4b/.m4a outputs include chapter markers",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=ENGINES,
        default="kokoro",
        help="TTS engine; kokoro (default) is ~50x faster than the others and "
        "the right choice for whole books. qwen/chatterbox/fish run in their "
        "own environments, downloaded on first use (or via: cathy setup)",
    )
    parser.add_argument(
        "-v",
        "--voice",
        help="'male' (default) or 'female' works on every engine. Kokoro also "
        "takes preset names and blends like 'af_heart:2,af_bella:1'; qwen "
        "takes speaker names; chatterbox/fish take a reference .wav to clone "
        "(see --list-voices)",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=1.0,
        help="speech speed; kokoro re-times natively, other engines are "
        "re-timed with ffmpeg's pitch-preserving tempo filter; default: 1.0",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="narrate everything; by default front/back matter that makes no "
        "sense read aloud (title page, contents, copyright, index, ...) is "
        "skipped",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="force CPU even if a GPU is available"
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="list available voices and exit"
    )
    args = parser.parse_args(argv)
    if not args.list_voices and args.input is None:
        parser.error(
            "input file is required — e.g. `cathy book.mobi -o book.m4b`; "
            "see `cathy help` for options and subcommands"
        )
    return args


def list_voices() -> None:
    print("Every engine accepts -v male (default) or -v female.\n")
    print("kokoro: aliases expand to presets/blends; presets are mixable")
    for name, target in KOKORO_ALIASES.items():
        print(f"  {name:16} = {target}")
    for name, description in KOKORO_VOICES.items():
        print(f"  {name:16} {description}")
    print("\nqwen: preset speakers")
    for name, description in QWEN_SPEAKERS.items():
        print(f"  {name:16} {description}")
    print("\nchatterbox, fish: pass a reference .wav to clone that voice;")
    print("male/female use a generated reference clip.")


HEADING_TAGS = ["h1", "h2", "h3", "h4"]
BLOCK_TAGS = ["p", "li", "blockquote", *HEADING_TAGS]


def html_blocks(html: str):
    """Return top-level readable block elements from an HTML document."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Superscript numbers are footnote markers, not prose — don't read them.
    for tag in soup("sup"):
        if re.fullmatch(r"[\d*†‡]{1,4}", tag.get_text(strip=True)):
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
    """Read a text/ebook file and return its chapters (titles as found;
    untitled chapters are numbered later, after front-matter filtering)."""
    suffix = path.suffix.lower()
    if suffix in (".mobi", ".azw", ".azw3"):
        return mobi_to_chapters(path)
    if suffix == ".epub":
        return epub_to_chapters(path)
    return text_to_chapters(path.read_text(encoding="utf-8"))


def trim_edges(audio, sr: int, threshold: float = 2e-3):
    """Trim near-silence off both ends of a segment, keeping a little padding.

    Engines emit variable amounts of edge silence; trimming it and inserting
    explicit pauses gives the narration a uniform rhythm.
    """
    import numpy as np

    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    keep = int(sr * EDGE_KEEP_SECONDS)
    return audio[max(0, loud[0] - keep) : min(len(audio), loud[-1] + 1 + keep)]


def chapter_hash(fingerprint: str, title: str, paragraphs: list[str]) -> str:
    import hashlib

    text = "\n\n".join([fingerprint, title, *paragraphs])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def narrate_chapters(
    engine, chapters: list[Chapter], workdir: Path, fingerprint: str
) -> list[Path]:
    """Synthesize each chapter into its own wav inside workdir; return paths.

    Chapter files are named by a hash of the settings and text, and finished
    files are renamed into place atomically — so an interrupted run resumes
    where it left off, and a changed voice/engine/text never reuses stale
    audio.
    """
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm

    sr = engine.sr

    def silence(seconds: float):
        return np.zeros(int(sr * seconds), dtype=np.float32)

    workdir.mkdir(parents=True, exist_ok=True)
    paths = [
        workdir / f"chapter-{i:04d}-{chapter_hash(fingerprint, title, paragraphs)}.wav"
        for i, (title, paragraphs) in enumerate(chapters, start=1)
    ]
    for stale in set(workdir.glob("*.wav")) - set(paths):
        stale.unlink()
    done = sum(1 for p in paths if p.exists())
    if done:
        print(f"Resuming: {done} of {len(chapters)} chapters already narrated.")

    total = sum(len(paragraphs) for _, paragraphs in chapters)
    with tqdm(total=total, unit="para", desc="Narrating") as progress:
        for path, (title, paragraphs) in zip(paths, chapters):
            if path.exists():
                progress.update(len(paragraphs))
                continue
            tmp = path.with_suffix(".tmp.wav")
            with sf.SoundFile(
                tmp, mode="w", samplerate=sr, channels=1, format="WAV"
            ) as f:
                for i, paragraph in enumerate(paragraphs):
                    first = True
                    for audio in engine.paragraph(paragraph):
                        if not first:
                            f.write(silence(SENTENCE_PAUSE))
                        f.write(trim_edges(audio, sr))
                        first = False
                    # A chapter's first paragraph is usually its heading; give
                    # it the longer, audiobook-style pause.
                    is_heading = i == 0 and paragraph == title
                    f.write(silence(HEADING_PAUSE if is_heading else PARAGRAPH_PAUSE))
                    progress.update(1)
            tmp.replace(path)
    return paths


def concat_chapters(
    paths: list[Path], titles: list[str], wav_path: Path, sr: int
) -> list[tuple[str, float, float]]:
    """Concatenate chapter wavs into wav_path; return (title, start_s, end_s)."""
    import soundfile as sf

    marks = []
    written = 0
    with sf.SoundFile(
        wav_path, mode="w", samplerate=sr, channels=1, format="WAV"
    ) as out:
        for title, path in zip(titles, paths):
            start = written
            with sf.SoundFile(path) as f:
                for block in f.blocks(blocksize=1 << 16, dtype="float32"):
                    out.write(block)
                    written += len(block)
            marks.append((title, start / sr, written / sr))
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


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def atempo_chain(speed: float) -> str:
    """Build an ffmpeg atempo filter chain (each stage handles 0.5-2.0x)."""
    factors = []
    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    factors.append(speed)
    return ",".join(f"atempo={f:g}" for f in factors)


def scale_metadata(metadata: str | None, speed: float) -> str | None:
    """Rescale chapter timestamps in ffmetadata text for tempo-changed audio."""
    import re

    if metadata is None or speed == 1.0:
        return metadata
    return re.sub(
        r"^(START=|END=)(\d+)$",
        lambda m: f"{m.group(1)}{int(int(m.group(2)) / speed)}",
        metadata,
        flags=re.MULTILINE,
    )


def convert(
    wav_path: Path,
    output: Path,
    metadata: str | None,
    duration: float,
    speed: float = 1.0,
) -> None:
    """Convert audio to the requested container via ffmpeg.

    metadata is ffmetadata text with chapter marks; it is embedded when the
    output container supports chapters (.m4b/.m4a). speed != 1 re-times the
    audio with the atempo filter (pitch-preserving), chapters included.
    """
    from tqdm import tqdm

    metadata = scale_metadata(metadata, speed)
    duration /= speed
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
    cmd += ["-i", str(wav_path)]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete_on_close=False) as meta:
        if output.suffix.lower() in CHAPTER_FORMATS:
            if metadata:
                meta.write(metadata)
                meta.close()
                cmd += ["-f", "ffmetadata", "-i", meta.name, "-map_metadata", "1"]
                cmd += ["-map_chapters", "1"]
        # Output options must come after every -i input
        if speed != 1.0:
            cmd += ["-filter:a", atempo_chain(speed)]
        if output.suffix.lower() in CHAPTER_FORMATS:
            cmd += ["-c:a", "aac", "-b:a", "96k", "-f", "ipod"]
        cmd.append(str(output))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        with tqdm(
            total=round(duration), unit="s", desc=f"Encoding {output.name}"
        ) as progress:
            for line in proc.stdout:
                # ffmpeg's -progress emits key=value lines; out_time_us is
                # microseconds encoded so far ("N/A" until the first frame).
                key, _, value = line.strip().partition("=")
                if key in ("out_time_us", "out_time_ms") and value.isdigit():
                    progress.n = min(round(duration), round(int(value) / 1_000_000))
                    progress.refresh()
            progress.n = progress.total
            progress.refresh()
        if proc.wait() != 0:
            sys.exit(f"error: ffmpeg failed converting to {output}")


def engine_spec(engine: str) -> str:
    import os

    return f"cathy[{engine}] @ {os.environ.get('CATHY_SOURCE', CATHY_SOURCE)}"


def delegate(engine: str, argv: list[str]) -> None:
    """Re-run this command via uv in an environment that has the engine.

    uv caches the environment per spec, so only the first run downloads.
    """
    import os
    import shutil

    if os.environ.get("CATHY_DELEGATED"):
        sys.exit(f"error: engine {engine!r} still unavailable after delegation")
    if shutil.which("uv") is None:
        sys.exit(
            f"error: engine {engine!r} is not installed and uv is not on PATH; "
            f"install with: uv tool install --reinstall '{engine_spec(engine)}'"
        )
    print(f"Engine {engine!r} runs in its own environment (first use downloads it)...")
    result = subprocess.run(
        delegate_cmd(engine) + ["cathy", *argv],
        env={**os.environ, "CATHY_DELEGATED": "1", **delegate_env()},
    )
    sys.exit(result.returncode)


def delegate_env() -> dict[str, str]:
    """Extra environment for delegated uv invocations.

    Mirrors the --index-strategy flag as an env var: some uv setups (an
    ambient UV_INDEX, older versions) otherwise resolve the pytorch indexes
    first-match and can't see packages that only exist on PyPI.
    """
    if sys.platform == "win32":
        return {"UV_INDEX_STRATEGY": "unsafe-best-match"}
    return {}


def delegate_cmd(engine: str) -> list[str]:
    """uv invocation prefix for running cathy with the given engine's extra."""
    import os

    source = os.environ.get("CATHY_SOURCE", CATHY_SOURCE)
    if source.startswith("git+"):
        # python 3.12: the engines' pinned dependency sets are validated there
        cmd = ["uv", "tool", "run", "--python", "3.12"]
        if sys.platform == "win32":
            # PyPI torch is CPU-only on Windows; CUDA wheels live on the
            # pytorch.org indexes (cu126 covers chatterbox's torch 2.6 pin)
            cmd += ["--index", "https://download.pytorch.org/whl/cu128"]
            cmd += ["--index", "https://download.pytorch.org/whl/cu126"]
            cmd += ["--index-strategy", "unsafe-best-match"]
        return cmd + ["--from", engine_spec(engine)]
    # A local CATHY_SOURCE is a development tree: run it as a project so the
    # environment always tracks the current source (uv's URL-spec caching
    # would serve stale builds)
    return ["uv", "run", "--project", source, "--extra", engine]


def setup(argv: list[str]) -> None:
    """Pre-download engine environments so first narration doesn't wait."""
    optional = [e for e in ENGINES if e != "kokoro"]
    parser = argparse.ArgumentParser(
        prog="cathy setup",
        description="Pre-download engine environments so the first narration "
        "doesn't wait; also refreshes them after `uv tool upgrade cathy`. "
        "With no arguments, sets up all optional engines.",
    )
    parser.add_argument(
        "engines",
        nargs="*",
        metavar="ENGINE",
        help=f"engines to set up: {', '.join(optional)} (default: all)",
    )
    engines = parser.parse_args(argv).engines or optional
    for engine in engines:
        if engine not in optional:
            parser.error(
                f"unknown engine {engine!r} (choose from {', '.join(optional)})"
            )
    for engine in engines:
        print(f"Setting up {engine}...")
        cmd = delegate_cmd(engine)
        if cmd[:3] == ["uv", "tool", "run"]:
            # --refresh so `cathy setup` after an upgrade also updates the
            # cached engine environments to the latest cathy
            cmd.insert(3, "--refresh")
        import os

        result = subprocess.run(
            cmd + ["cathy", "--list-voices"],
            stdout=subprocess.DEVNULL,
            env={**os.environ, **delegate_env()},
        )
        if result.returncode != 0:
            sys.exit(f"error: setup of {engine!r} failed")
        print(f"Engine {engine} ready.")


def convert_command(argv: list[str]) -> None:
    """cathy convert <input audio> <output audio> [-s SPEED]

    Converts without re-narrating. If <input>.chapters.txt exists (written
    when narrating to .wav), its chapter marks are embedded in m4b/m4a output.
    """
    parser = argparse.ArgumentParser(
        prog="cathy convert",
        description="Convert existing audio to another format without "
        "re-narrating. If IN.chapters.txt exists (cathy writes it when "
        "narrating to .wav), chapter markers are embedded in .m4b/.m4a output.",
    )
    parser.add_argument("source", metavar="IN", type=Path, help="input audio file")
    parser.add_argument(
        "output",
        metavar="OUT",
        type=Path,
        help="output audio file (.m4b/.m4a get chapters; any ffmpeg format works)",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=1.0,
        help="re-time the audio (pitch-preserving); default: 1.0",
    )
    args = parser.parse_args(argv)
    if not args.source.exists():
        sys.exit(f"error: {args.source} not found")

    metadata = None
    sidecar = args.source.with_suffix(".chapters.txt")
    if args.output.suffix.lower() in CHAPTER_FORMATS:
        if sidecar.exists():
            metadata = sidecar.read_text(encoding="utf-8")
            print(f"Using chapters from {sidecar}")
        else:
            print(f"note: {sidecar} not found; {args.output.name} will have no chapters")
    convert(args.source, args.output, metadata, probe_duration(args.source), args.speed)
    print(f"Wrote {args.output}")


def help_command(argv: list[str]) -> None:
    topic = argv[0] if argv else None
    if topic == "convert":
        convert_command(["--help"])
    elif topic == "setup":
        setup(["--help"])
    elif topic in ("voices", "--list-voices"):
        list_voices()
    else:
        parse_args(["--help"])


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    subcommands = {
        "setup": setup,
        "convert": convert_command,
        "help": help_command,
        "voices": lambda _: list_voices(),
    }
    if argv and argv[0] in subcommands:
        subcommands[argv[0]](argv[1:])
        return

    args = parse_args(argv)

    if args.list_voices:
        list_voices()
        return

    if not args.input.exists():
        sys.exit(f"error: {args.input} not found")

    if not engine_available(args.engine):
        delegate(args.engine, argv)

    import torch

    from cathy.engines import build_engine

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and not args.cpu:
        import shutil

        if shutil.which("nvidia-smi"):
            # An NVIDIA driver is installed but torch can't reach it: this
            # environment has CPU-only torch (the PyPI default on Windows)
            print(
                "note: this machine has an NVIDIA GPU, but this environment "
                "has CPU-only torch, so narration runs on CPU (much slower).\n"
                "To fix (see the README's 'Windows and GPU' section):\n"
                "  1. reinstall cathy with the CUDA wheel index:\n"
                "     uv tool install --reinstall --python 3.12 "
                "--index https://download.pytorch.org/whl/cu128 "
                "--index-strategy unsafe-best-match "
                "git+https://github.com/cleanunicorn/cathy\n"
                "  2. rebuild cached engine environments: cathy setup"
            )
        else:
            print(
                "note: no CUDA GPU detected, narrating on CPU (much slower). "
                "Kokoro is the only engine with tolerable CPU speed."
            )
    output = args.output or args.input.with_suffix(".wav")
    if args.output is None and args.input.suffix.lower() in EBOOK_FORMATS:
        print(
            f"tip: writing plain {output.name}; use `-o {output.stem}.m4b` "
            "for an audiobook with chapter markers"
        )
    chapters = load_chapters(args.input)
    if not args.all:
        chapters, skipped = drop_front_back_matter(chapters)
        if skipped:
            print(f"Skipping (use --all to keep): {', '.join(skipped)}")
    chapters = normalize_chapters(chapters)
    if not chapters:
        sys.exit("error: input file contains no text")
    chapters = [
        (title or f"Chapter {i}", paragraphs)
        for i, (title, paragraphs) in enumerate(chapters, start=1)
    ]

    # Kokoro re-times speech natively (best quality); the other engines have
    # no speed input, so their audio is re-timed afterwards with ffmpeg.
    native_speed = args.engine == "kokoro"
    post_speed = 1.0 if native_speed else args.speed

    paragraph_count = sum(len(p) for _, p in chapters)
    print(
        f"Engine: {args.engine} | Device: {device} | Voice: {args.voice or 'male'} | "
        f"Chapters: {len(chapters)} | Paragraphs: {paragraph_count}"
    )
    engine = build_engine(
        args.engine, args.voice, args.speed if native_speed else 1.0, device
    )

    # Per-chapter checkpoints: an interrupted run resumes from here. The
    # fingerprint covers everything that changes the audio itself.
    workdir = output.parent / f"{output.name}.partial"
    fingerprint = "|".join(
        [args.engine, args.voice or "male",
         str(args.speed if native_speed else 1.0), str(engine.sr)]
    )
    try:
        paths = narrate_chapters(engine, chapters, workdir, fingerprint)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted — finished chapters are saved in {workdir}/; "
            "rerun the same command to resume."
        )
        sys.exit(130)

    titles = [title for title, _ in chapters]
    if output.suffix.lower() == ".wav" and post_speed == 1.0:
        marks = concat_chapters(paths, titles, output, engine.sr)
        metadata = ffmetadata(marks)
    else:
        wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            marks = concat_chapters(paths, titles, wav_path, engine.sr)
            duration = marks[-1][2] if marks else 0.0
            convert(wav_path, output, ffmetadata(marks), duration, post_speed)
        finally:
            wav_path.unlink(missing_ok=True)
        metadata = scale_metadata(ffmetadata(marks), post_speed)
    import shutil

    shutil.rmtree(workdir, ignore_errors=True)
    if output.suffix.lower() == ".wav":
        # Sidecar lets `cathy convert book.wav book.m4b` keep the chapters.
        output.with_suffix(".chapters.txt").write_text(metadata, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
