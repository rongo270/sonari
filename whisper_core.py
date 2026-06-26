"""
whisper_core.py
===============
Shared transcription engine built on faster-whisper (CTranslate2).

Used by:
  - tools/transcribe.py  (speech -> text)
  - lyrics.py            (music -> lyrics, after vocal separation)

Design goals:
  - Highest practical accuracy (defaults to the large-v3 model).
  - Runs well on a CPU-only machine (int8) but auto-uses an NVIDIA GPU if present.
  - Writes clean output in several formats: txt, srt, vtt, json.

Nothing here is specific to your machine; the same file works unchanged on a
cloud GPU later.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

# Keep downloaded model weights inside the project (./models) so everything
# lives in one place and is easy to find / back up / delete.
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Device / precision selection
# --------------------------------------------------------------------------- #
def pick_device_and_compute(device: str = "auto", compute_type: str = "auto"):
    """Choose the best device and numeric precision.

    CPU  -> int8   (fast + low memory; best choice for a laptop)
    CUDA -> float16 (best choice for an NVIDIA GPU)
    """
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def load_model(
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "auto",
    cpu_threads: int = 0,
) -> tuple[WhisperModel, str, str]:
    """Load a Whisper model. Downloads it on first use (cached in ./models)."""
    device, compute_type = pick_device_and_compute(device, compute_type)
    print(f"[model] loading '{model_size}' on {device} ({compute_type}) ...")
    t0 = time.time()
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,  # 0 = let the library decide
        download_root=str(MODELS_DIR),
    )
    print(f"[model] ready in {time.time() - t0:.1f}s")
    return model, device, compute_type


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
@dataclass
class TranscriptResult:
    text: str
    segments: list
    language: str
    language_probability: float
    duration: float
    elapsed: float


def transcribe_stream(
    model: WhisperModel,
    audio_path,
    *,
    language: str | None = None,
    task: str = "transcribe",          # "transcribe" or "translate" (to English)
    beam_size: int = 5,
    vad: bool = True,
    word_timestamps: bool = True,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool = True,
    temperature=None,
):
    """Streaming/generator variant of transcribe_file.

    Yields ("info", info) exactly once first, then ("segment", segment) for each
    segment as Whisper decodes it. A UI (or transcribe_file below) consumes this
    to show results as they arrive instead of waiting for the whole file.

    `temperature` defaults to a fallback ladder: if decoding looks unreliable
    Whisper retries with higher temperature. This greatly reduces failures on
    hard audio.
    """
    if temperature is None:
        temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    segments_gen, info = model.transcribe(
        str(audio_path),
        language=language,
        task=task,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters=({"min_silence_duration_ms": 500} if vad else None),
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        temperature=temperature,
    )

    yield "info", info
    for seg in segments_gen:  # streamed: decoded one chunk at a time
        yield "segment", seg


def _build_result(info, segments, elapsed: float) -> TranscriptResult:
    """Assemble a TranscriptResult from the streamed info + segments."""
    return TranscriptResult(
        text="".join(seg.text for seg in segments).strip(),
        segments=segments,
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        elapsed=elapsed,
    )


def transcribe_file(
    model: WhisperModel,
    audio_path,
    *,
    language: str | None = None,
    task: str = "transcribe",          # "transcribe" or "translate" (to English)
    beam_size: int = 5,
    vad: bool = True,
    word_timestamps: bool = True,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool = True,
    temperature=None,
    verbose: bool = True,
) -> TranscriptResult:
    """Transcribe one audio file and return a TranscriptResult.

    Thin wrapper over transcribe_stream that collects everything and (when
    verbose) prints progress to the console, exactly as before.
    """
    t0 = time.time()
    info = None
    segments = []
    for kind, payload in transcribe_stream(
        model,
        audio_path,
        language=language,
        task=task,
        beam_size=beam_size,
        vad=vad,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        temperature=temperature,
    ):
        if kind == "info":
            info = payload
            if verbose:
                print(
                    f"[lang] detected '{info.language}' "
                    f"(confidence {info.language_probability:.0%}), "
                    f"audio length {info.duration:.1f}s"
                )
        else:
            seg = payload
            segments.append(seg)
            if verbose:
                print(f"  [{_ts(seg.start)} -> {_ts(seg.end)}] {seg.text.strip()}")

    elapsed = time.time() - t0
    if verbose:
        rtf = elapsed / info.duration if info and info.duration else 0
        print(f"[done] {elapsed:.1f}s  (={rtf:.2f}x audio length)")

    return _build_result(info, segments, elapsed)


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def _ts(seconds: float, sep: str = ",") -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:  # rounding edge case
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_txt(result: TranscriptResult, path) -> None:
    Path(path).write_text(result.text + "\n", encoding="utf-8")


def write_srt(result: TranscriptResult, path) -> None:
    lines = []
    for i, seg in enumerate(result.segments, 1):
        lines += [str(i), f"{_ts(seg.start)} --> {_ts(seg.end)}", seg.text.strip(), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_vtt(result: TranscriptResult, path) -> None:
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines += [f"{_ts(seg.start, '.')} --> {_ts(seg.end, '.')}", seg.text.strip(), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_json(result: TranscriptResult, path) -> None:
    data = {
        "text": result.text,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.duration,
        "segments": [
            {
                "id": i,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": (
                    [
                        {
                            "start": w.start,
                            "end": w.end,
                            "word": w.word,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ]
                    if seg.words
                    else []
                ),
            }
            for i, seg in enumerate(result.segments)
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


_WRITERS = {"txt": write_txt, "srt": write_srt, "vtt": write_vtt, "json": write_json}


def write_all(result: TranscriptResult, out_dir, stem: str, formats=("txt", "srt", "vtt", "json")):
    """Write every requested format. Returns the list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        _WRITERS[fmt](result, path)
        written.append(path)
    return written
