"""
media.py  -  Download audio from a link (YouTube, etc.)
=======================================================

Thin wrapper around yt-dlp so the rest of Sonari can accept an online link in
addition to local files. Used by:
  - app.py        (the "YouTube -> lyrics" tab)
  - transcribe.py (--url)
  - lyrics.py     (--url)

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


def download_audio(url: str, out_dir, prefer_mp3: bool = True) -> tuple[Path, str]:
    """Download the best audio from `url` into `out_dir`.

    Returns (path_to_audio_file, video_title).
    """
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convert = prefer_mp3 and have_ffmpeg()
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,          # a link to a video in a playlist -> just that video
        "restrictfilenames": True,   # ASCII-safe filenames (avoids Windows/unicode issues)
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
