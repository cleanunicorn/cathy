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
    parser.add_argument("input", nargs="?", type=Path, help="input .txt/.md file")
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


def load_text(path: Path) -> list[str]:
    """Read the input file and return non-empty paragraphs."""
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
