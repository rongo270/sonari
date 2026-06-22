#!/usr/bin/env python
"""
lyrics.py  -  Music -> lyrics
=============================

Extracts the lyrics from a song in two stages:

  1. SEPARATE:  Demucs (Meta's state-of-the-art free model) splits the song
                into "vocals" and "everything else", so the instruments are
                removed and only the singing remains.
  2. TRANSCRIBE: faster-whisper turns that isolated vocal track into text.

This two-stage approach is how you get usable lyrics: running Whisper directly
on a full song fails because the music drowns out the words.

EXAMPLES
--------
  python lyrics.py "input/song.mp3"
  python lyrics.py "input/song.mp3" --model large-v3-turbo --language en
  python lyrics.py "input/song.mp3" --keep-vocals     # also keep vocals.wav

NOTE ON SPEED
-------------
Demucs is heavy. On this laptop (CPU only) a 3-4 minute song can take roughly
5-20 minutes to separate. That is normal. A GPU would do it in seconds.
Lyrics transcription is inherently harder than speech (singing, rhyme, overlap),
so expect good-but-not-perfect results.
"""
import argparse
import sys
import time
from pathlib import Path

import whisper_core as core

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_DIR / "output"
SEP_DIR = PROJECT_DIR / "separated"  # where Demucs writes stems


def _load_audio_tensor(path: Path, target_sr: int, target_channels: int):
    """Load any audio file to a (channels, samples) float32 torch tensor.

    Uses soundfile (libsndfile, which reads mp3/flac/wav/ogg) so we don't need
    ffmpeg or torchcodec. Falls back to faster-whisper's PyAV decoder for exotic
    formats. Resamples / re-channels to match the Demucs model.
    """
    import numpy as np
    import torch

    try:
        import soundfile as sf

        data, in_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, ch)
        wav = torch.from_numpy(data.T.copy())  # (ch, frames)
    except Exception as exc:
        print(f"[separate] soundfile couldn't read it ({exc}); using PyAV decoder...")
        from faster_whisper.audio import decode_audio

        mono = decode_audio(str(path), sampling_rate=target_sr)  # mono float32
        wav = torch.from_numpy(np.asarray(mono)).unsqueeze(0)  # (1, frames)
        in_sr = target_sr

    # match channel count expected by the model
    if wav.shape[0] == 1 and target_channels == 2:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] == 2 and target_channels == 1:
        wav = wav.mean(0, keepdim=True)

    # resample if needed
    if in_sr != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, in_sr, target_sr)
    return wav


def separate_vocals(audio: Path, model_name: str = "htdemucs", max_seconds: float = 0) -> Path:
    """Isolate the vocal stem using the Demucs Python API. Returns vocals.wav.

    We call apply_model() on an audio tensor we load ourselves, which avoids the
    Demucs CLI's dependency on torchcodec/ffmpeg.
    """
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    out_dir = SEP_DIR / model_name / audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = out_dir / "vocals.wav"

    print(f"[separate] loading Demucs model '{model_name}' ...")
    model = get_model(model_name)
    model.cpu().eval()

    print(f"[separate] reading audio: {audio.name}")
    wav = _load_audio_tensor(audio, model.samplerate, model.audio_channels)
    if max_seconds and max_seconds > 0:
        keep = int(max_seconds * model.samplerate)
        wav = wav[:, :keep]
        print(f"[separate] (testing first {max_seconds:.0f}s only)")

    # standard Demucs normalization
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    print("[separate] separating vocals - SLOW on CPU, please be patient ...")
    t0 = time.time()
    with torch.no_grad():
        sources = apply_model(
            model, wav[None], device="cpu", split=True, overlap=0.25, progress=True
        )[0]
    sources = sources * std + mean

    vocals = sources[model.sources.index("vocals")]  # (channels, samples)
    sf.write(str(vocals_path), vocals.t().numpy(), model.samplerate)
    print(f"[separate] done in {time.time() - t0:.1f}s -> {vocals_path}")
    return vocals_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Music -> lyrics (Demucs vocal separation + faster-whisper).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", nargs="?", help="a song file (mp3, wav, flac, m4a, ...)")
    ap.add_argument("--url", default=None,
                    help="download audio from a YouTube (or other) link first, then make lyrics")
    ap.add_argument("--model", default="large-v3-turbo",
                    help="large-v3-turbo (default) | large-v3 (max accuracy, slow) | medium | small")
    ap.add_argument("--language", default=None, help="language code, e.g. en. Default: auto")
    ap.add_argument("--demucs-model", default="htdemucs",
                    help="htdemucs (default) | htdemucs_ft (slower, better) | mdx_extra")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder")
    ap.add_argument("--formats", default="txt,srt,json", help="comma list: txt,srt,vtt,json")
    ap.add_argument("--keep-vocals", action="store_true",
                    help="copy the isolated vocals.wav next to the lyrics")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="only process the first N seconds (great for a quick test)")
    ap.add_argument("--vad", action="store_true",
                    help="enable silence skipping (VAD). OFF by default for songs: VAD tends "
                         "to drop repeated chorus lines and mangle singing")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = auto)")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    if args.url:
        from media import download_audio
        print(f"[youtube] downloading audio from: {args.url}")
        audio, title = download_audio(args.url, PROJECT_DIR / "input")
        print(f"[youtube] got: {audio.name}")
    elif args.input:
        audio = Path(args.input)
        if not audio.exists():
            sys.exit(f"ERROR: file not found: {audio}")
    else:
        sys.exit("ERROR: provide a song file, or --url <link>")

    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    overall = time.time()

    # 1) isolate the vocals
    vocals = separate_vocals(audio, args.demucs_model, args.max_seconds)

    # 2) transcribe the vocals
    model, *_ = core.load_model(args.model, "auto", "auto", args.threads)
    print("\n[lyrics] transcribing isolated vocals ...")
    result = core.transcribe_file(
        model,
        vocals,
        language=args.language,
        beam_size=5,
        # VAD is OFF by default for music (it drops repeated lines); --vad re-enables it.
        vad=args.vad,
        word_timestamps=True,
        # Singing confuses the "use previous text as context" feature and makes
        # Whisper repeat lines, so we turn it off for music.
        condition_on_previous_text=False,
    )

    out_dir = Path(args.out)
    written = core.write_all(result, out_dir, f"{audio.stem}.lyrics", formats)

    if args.keep_vocals:
        import shutil
        dest = out_dir / f"{audio.stem}.vocals.wav"
        shutil.copy2(vocals, dest)
        written.append(dest)

    print("\n  saved:")
    for w in written:
        print(f"    {w}")
    print(f"\nAll done in {time.time() - overall:.1f}s.")


if __name__ == "__main__":
    main()
