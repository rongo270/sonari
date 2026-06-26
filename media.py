"""
media.py  -  Download audio from a link (YouTube, etc.)
=======================================================

Thin wrapper around yt-dlp so the rest of Sonari can accept an online link in
addition to local files. Used by:
  - app.py            (the "YouTube -> lyrics" tab)
  - tools/transcribe.py (--url)
  - lyrics.py         (--url)

It downloads the best available audio track. If ffmpeg is installed it converts
to a clean .mp3; if not, it keeps the native track (.m4a/.webm/.opus) which the
rest of the pipeline can still read via faster-whisper's bundled PyAV decoder.
So this works even without system ffmpeg.

Note: only download audio you have the right to use (your own content, public
domain, or where the platform's terms allow it).
"""
from __future__ import annotations

import shutil
from pathlib import Path


def have_ffmpeg() -> bool:
    """True if a system ffmpeg is on PATH (lets us convert to mp3)."""
    return shutil.which("ffmpeg") is not None


def _enabled_js_runtimes() -> dict:
    """JavaScript runtimes yt-dlp may use to unlock YouTube downloads.

    YouTube now requires *running its player JavaScript* to sign the media URLs;
    with no JS runtime available, yt-dlp gets a plain "HTTP Error 403: Forbidden"
    on the actual audio download (extraction still succeeds, which is what makes
    the error so confusing). yt-dlp only auto-enables Deno, so here we also enable
    Node / Bun when they're on PATH — most machines already have Node installed.

    Returns a dict like {"node": {}} for yt-dlp's `js_runtimes` option, or {} if
    none are found (the caller turns that into a friendly install message).
    """
    return {name: {} for name in ("deno", "node", "bun") if shutil.which(name)}


def download_audio(url: str, out_dir, prefer_mp3: bool = True) -> tuple[Path, str]:
    """Download the best audio from `url` into `out_dir`.

    Returns (path_to_audio_file, video_title).
    """
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runtimes = _enabled_js_runtimes()
    if not runtimes:
        raise RuntimeError(
            "YouTube needs a JavaScript runtime to unlock downloads (otherwise it "
            "returns 403 Forbidden). Please install Node.js from https://nodejs.org "
            "(the easiest option on Windows), then restart the app and try again."
        )

    convert = prefer_mp3 and have_ffmpeg()
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,          # a link to a video in a playlist -> just that video
        "restrictfilenames": True,   # ASCII-safe filenames (avoids Windows/unicode issues)
        "js_runtimes": runtimes,     # let yt-dlp use Node/Deno to sign URLs (avoids 403)
        "quiet": True,
        "no_warnings": True,
    }
    if convert:
        ydl_opts["postprocessors"] = [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "audio")
        path = Path(ydl.prepare_filename(info))
        if convert:
            path = path.with_suffix(".mp3")  # ffmpeg postprocessor changed the extension

    if not path.exists():
        # Fallback: yt-dlp's reported name didn't match (rare) -> grab the newest file.
        candidates = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        if candidates:
            path = candidates[-1]
        else:
            raise FileNotFoundError("Download finished but no audio file was found.")

    return path, title
