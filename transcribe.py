#!/usr/bin/env python
"""
transcribe.py  -  Speech -> text (highest-accuracy)
===================================================

Turns spoken audio (talking, interviews, podcasts, voice notes, video, ...)
into accurate text using faster-whisper with the large-v3 model.

EXAMPLES
--------
  # one file (uses the best model, auto language detection)
  python transcribe.py input/jfk.flac

  # a file anywhere on disk
  python transcribe.py "C:/Users/rongo/Desktop/meeting.mp3"

  # force English, faster model (good when the laptop is slow)
  python transcribe.py input/talk.m4a --model large-v3-turbo --language en

  # transcribe every audio file in a folder
  python transcribe.py input/

  # translate any language INTO English text
  python transcribe.py input/spanish.mp3 --task translate

MODELS (accuracy vs speed on CPU)
---------------------------------
  large-v3        <- best accuracy (default).      Slowest on CPU.
  large-v3-turbo  <- ~almost as good, much faster. Recommended on this laptop.
  medium / small  <- faster, lower accuracy.
  base / tiny     <- fastest, lowest accuracy (quick tests).
"""
import argparse
import sys
import time
from pathlib import Path

import whisper_core as core

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_DIR / "output"

AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v",  # video: audio is extracted
}


def gather_inputs(path_str: str) -> list[Path]:
    p = Path(path_str)
    if not p.exists():
        sys.exit(f"ERROR: path does not exist: {p}")
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        if not files:
            sys.exit(f"ERROR: no audio/video files found in folder: {p}")
        return files
    return [p]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Speech -> text with faster-whisper (large-v3 by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", nargs="?", help="audio/video file, or a folder of them")
    ap.add_argument("--url", default=None,
                    help="download audio from a YouTube (or other) link first, then transcribe")
    ap.add_argument("--model", default="large-v3-turbo",
                    help="large-v3-turbo (default, best for this laptop) | large-v3 (max accuracy, slow) "
                         "| medium | small | base | tiny")
    ap.add_argument("--language", default=None,
                    help="language code e.g. en, he, es, fr. Default: auto-detect")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                    help="'translate' outputs English text from any language")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder")
    ap.add_argument("--formats", default="txt,srt,vtt,json",
                    help="comma list of: txt,srt,vtt,json")
    ap.add_argument("--beam-size", type=int, default=5, help="higher = more accurate, slower")
    ap.add_argument("--no-vad", action="store_true",
                    help="disable voice-activity detection (silence skipping)")
    ap.add_argument("--no-word-timestamps", action="store_true",
                    help="skip per-word timing (a little faster)")
    ap.add_argument("--prompt", default=None,
                    help="hint words/spellings, e.g. names or jargon, to improve accuracy")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default="auto",
                    help="auto | int8 | int8_float32 | float16 | float32")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads (0 = auto). Try 4 on this laptop if it feels slow.")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    if args.url:
        from media import download_audio
        print(f"[youtube] downloading audio from: {args.url}")
        audio, title = download_audio(args.url, PROJECT_DIR / "input")
        print(f"[youtube] got: {audio.name}")
        inputs = [audio]
    elif args.input:
        inputs = gather_inputs(args.input)
    else:
        sys.exit("ERROR: provide an input file/folder, or --url <link>")

    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())

    model, device, compute_type = core.load_model(
        args.model, args.device, args.compute_type, args.threads
    )

    out_dir = Path(args.out)
    overall = time.time()
    print(f"\n{'='*70}\nTranscribing {len(inputs)} file(s) -> {out_dir}\n{'='*70}")

    for idx, audio in enumerate(inputs, 1):
        print(f"\n[{idx}/{len(inputs)}] {audio.name}")
        try:
            result = core.transcribe_file(
                model,
                audio,
                language=args.language,
                task=args.task,
                beam_size=args.beam_size,
                vad=not args.no_vad,
                word_timestamps=not args.no_word_timestamps,
                initial_prompt=args.prompt,
            )
        except Exception as exc:  # keep going on the rest of the batch
            print(f"  !! FAILED: {exc}")
            continue

        written = core.write_all(result, out_dir, audio.stem, formats)
        print("  saved:")
        for w in written:
            print(f"    {w}")

    print(f"\n{'='*70}\nAll done in {time.time() - overall:.1f}s. Output in: {out_dir}\n{'='*70}")


if __name__ == "__main__":
    main()
