"""TTS engine backends. Each engine synthesizes one paragraph at a time and
exposes a sample rate; everything else (chunking pauses, chapters, output
formats) is shared in the CLI."""

import re
import sys
from pathlib import Path

ENGINES = ("kokoro", "qwen", "chatterbox", "fish")

# Top-level module each engine extra provides, to detect availability.
ENGINE_IMPORTS = {"qwen": "qwen_tts", "chatterbox": "chatterbox", "fish": "fish_speech"}


def engine_available(name: str) -> bool:
    """True if the engine can run in the current environment."""
    if name == "kokoro":
        return True
    import importlib.util

    return importlib.util.find_spec(ENGINE_IMPORTS[name]) is not None

# Curated subset of Kokoro's built-in voices. Prefix encodes language/gender:
# a=American, b=British English; f=female, m=male.
KOKORO_VOICES = {
    "af_heart": "American female (highest rated)",
    "af_bella": "American female, warm",
    "af_nicole": "American female, soft-spoken",
    "am_michael": "American male",
    "am_fenrir": "American male, deep",
    "bf_emma": "British female",
    "bm_george": "British male",
    "bm_fable": "British male, narrator",
}

KOKORO_ALIASES = {
    "male": "am_michael:2,am_fenrir:1",
    "female": "af_heart",
    "british-male": "bm_george:2,bm_fable:1",
    "british-female": "bf_emma",
}

QWEN_SPEAKERS = {
    "Ryan": "male, dynamic",
    "Aiden": "male, casual American",
    "Eric": "male, lively Sichuan accent",
    "Dylan": "male, Beijing accent",
    "Uncle_Fu": "male, seasoned",
    "Vivian": "female, bright",
    "Serena": "female, gentle",
    "Ono_Anna": "female, Japanese",
    "Sohee": "female, Korean",
}

QWEN_ALIASES = {"male": "Ryan", "female": "Vivian"}

# Read by Kokoro to build a reference clip when a cloning engine (chatterbox,
# fish) is asked for a generic 'male'/'female' voice instead of a .wav sample.
REFERENCE_TEXT = (
    "The old lighthouse keeper climbed the spiral stairs slowly, "
    "carrying a lantern in one hand and a worn leather book in the other. "
    "Outside, the waves rolled in from the dark Atlantic, "
    "steady and patient as they had always been."
)


def split_sentences(text: str, limit: int = 300) -> list[str]:
    """Split text into sentence groups of at most ~limit characters."""
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    groups: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > limit:
            groups.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        groups.append(current)
    return groups


def _quiet() -> None:
    """Silence third-party warnings and logging we cannot act on."""
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        from loguru import logger

        for noisy in ("kokoro", "misaki", "fish_speech"):
            logger.disable(noisy)
    except ImportError:
        pass

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except ImportError:
        pass


def _hushed(fn, *args, **kwargs):
    """Run fn with stdout swallowed; engines print progress chatter we already
    surface through our own progress bar (which writes to stderr)."""
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class KokoroEngine:
    """Kokoro-82M: fastest by far, preset voices, best fit for whole books."""

    sr = 24_000

    def __init__(self, voice: str, speed: float, device: str):
        _quiet()
        from kokoro import KPipeline

        voice = KOKORO_ALIASES.get(voice, voice)
        lang_code = voice[0]  # Kokoro convention: voice prefix picks the language
        self.pipeline = KPipeline(
            lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=device
        )
        self.voice = self._resolve(voice)
        self.speed = speed

    def _resolve(self, spec: str):
        """Support weighted blends like 'af_heart:2,af_bella:1'."""
        if ":" not in spec:
            return spec

        import torch

        names, weights = [], []
        for part in spec.split(","):
            name, _, weight = part.partition(":")
            names.append(name.strip())
            weights.append(float(weight or 1))
        packs = torch.stack([self.pipeline.load_single_voice(n) for n in names])
        w = torch.tensor(weights, dtype=packs.dtype) / sum(weights)
        return torch.sum(packs * w.reshape(-1, *[1] * (packs.dim() - 1)), dim=0)

    def paragraph(self, text: str):
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.speed):
            yield audio.numpy()


class QwenEngine:
    """Qwen3-TTS-0.6B-CustomVoice: expressive preset speakers, ~real-time."""

    def __init__(self, voice: str, speed: float, device: str):
        _quiet()
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            sys.exit("error: qwen engine not installed; run: uv run --extra qwen cathy ...")

        import torch

        voice = QWEN_ALIASES.get(voice, voice)
        if voice not in QWEN_SPEAKERS:
            sys.exit(f"error: unknown qwen speaker {voice!r}; options: {', '.join(QWEN_SPEAKERS)}")
        self.voice = voice
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map=device,
            dtype=torch.bfloat16,
        )
        wavs, self.sr = self._generate("Hello.")  # warm up and learn sample rate

    def _generate(self, text: str):
        return _hushed(
            self.model.generate_custom_voice,
            text=text,
            language="English",
            speaker=self.voice,
        )

    def paragraph(self, text: str):
        for group in split_sentences(text):
            wavs, _ = self._generate(group)
            yield wavs[0]


class ClonedVoiceMixin:
    """Resolve 'male'/'female'/path specs into a reference clip for cloning."""

    def reference_clip(self, voice: str, device: str) -> Path:
        if voice not in ("male", "female", "british-male", "british-female"):
            path = Path(voice)
            if path.suffix.lower() != ".wav" or not path.exists():
                sys.exit(
                    f"error: {type(self).__name__} needs a voice of 'male', 'female', "
                    f"or a path to an existing reference .wav, got {voice!r}"
                )
            return path

        import soundfile as sf

        cache = Path.home() / ".cache" / "cathy"
        cache.mkdir(parents=True, exist_ok=True)
        clip = cache / f"reference-{voice}.wav"
        if not clip.exists():
            kokoro = KokoroEngine(voice, 1.0, device)
            import numpy as np

            audio = np.concatenate(list(kokoro.paragraph(REFERENCE_TEXT)))
            sf.write(clip, audio, kokoro.sr)
        return clip


class ChatterboxEngine(ClonedVoiceMixin):
    """Chatterbox-Nano: 110M cloning model, expressive, MIT-licensed."""

    def __init__(self, voice: str, speed: float, device: str):
        _quiet()
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except ImportError:
            sys.exit(
                "error: chatterbox engine not installed; run: uv run --extra chatterbox cathy ..."
            )

        self.clip = str(self.reference_clip(voice, device))
        self.model = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
        self.sr = self.model.sr

    def paragraph(self, text: str):
        for group in split_sentences(text):
            wav = _hushed(self.model.generate, group, audio_prompt_path=self.clip)
            yield wav.squeeze(0).cpu().numpy()


class FishEngine(ClonedVoiceMixin):
    """Fish Audio OpenAudio S1-mini: 0.5B cloning model, research license."""

    def __init__(self, voice: str, speed: float, device: str):
        _quiet()
        try:
            import fish_speech
        except ImportError:
            sys.exit("error: fish engine not installed; run: uv run --extra fish cathy ...")

        # fish-speech modules locate their repo root via a .project-root marker
        # at import time; recreate it next to the installed package.
        marker = Path(next(iter(fish_speech.__path__))).parent / ".project-root"
        if not marker.exists():
            marker.touch()

        from fish_speech.inference_engine import TTSInferenceEngine

        import torch
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.utils.schema import ServeReferenceAudio
        from huggingface_hub import snapshot_download

        clip = self.reference_clip(voice, device)
        self.reference = ServeReferenceAudio(
            audio=clip.read_bytes(), text=REFERENCE_TEXT
        )

        checkpoints = Path(snapshot_download("fishaudio/s1-mini"))
        precision = torch.bfloat16
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=checkpoints, device=device, precision=precision, compile=False
        )
        decoder = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=checkpoints / "codec.pth",
            device=device,
        )
        self.engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder,
            precision=precision,
            compile=False,
        )
        if hasattr(decoder, "spec_transform"):
            self.sr = int(decoder.spec_transform.sample_rate)
        else:
            self.sr = int(decoder.sample_rate)

    def paragraph(self, text: str):
        from fish_speech.utils.schema import ServeTTSRequest

        request = ServeTTSRequest(
            text=text,
            references=[self.reference],
            max_new_tokens=2048,
            seed=42,  # keep the timbre stable across paragraphs
            format="wav",
        )
        segments = []
        for result in _hushed(lambda: list(self.engine.inference(request))):
            if result.code == "error":
                raise result.error
            if result.code == "segment" and result.audio is not None:
                segments.append(result.audio[1])
            elif result.code == "final" and result.audio is not None and not segments:
                segments.append(result.audio[1])
        yield from segments


def build_engine(name: str, voice: str | None, speed: float, device: str):
    classes = {
        "kokoro": KokoroEngine,
        "qwen": QwenEngine,
        "chatterbox": ChatterboxEngine,
        "fish": FishEngine,
    }
    return classes[name](voice or "male", speed, device)
