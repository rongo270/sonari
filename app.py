#!/usr/bin/env python
"""
app.py  -  Sonari visual app (web UI)
=====================================

A friendly browser interface for everything the command-line tools do, plus an
"online" tab that takes a YouTube (or other) link. It opens in your browser and
runs 100% on your own computer.

Three tabs, and every tab works the same way — like a little 2-step wizard:

    SCREEN 1 (setup)    pick your file / link and the options, press the button
            │
            ▼
    SCREEN 2 (work)     ONE clean progress bar fills up while it works …
                        … and only when it's finished do the lyrics + the
                        karaoke player appear.

The three tabs:
  1. Speech -> text     (upload a file or record from your mic)
  2. Song  -> lyrics    (Demucs separates the vocals, then Whisper reads them)
  3. From a link        (paste a YouTube URL -> audio -> lyrics or transcript)

Highlights:
  - A single, smooth progress bar (no more flickering / "page refreshing").
  - A built-in KARAOKE player: press play and the words light up in time with
    the audio, click any line to jump, and go fullscreen. For songs it plays the
    INSTRUMENTAL (a true sing-along) with a 🎤/🎹 toggle to peek at the vocals.
  - A GUITAR-CHORD sheet: the REAL chords are heard from the music (offline) and
    laid above the words — hard songs stay hard, easy songs stay easy. A Transpose
    slider moves the key up/down, an "Easy" button switches to the open-chord
    shapes you'd finger with a capo, and "Original" switches back.
  - VAD (silence skipping) defaults OFF for songs, because it tends to drop
    repeated chorus lines and mangle singing.

The command line still works exactly as before (transcribe / lyrics); this UI
just calls the same engine (whisper_core) under the hood.

Run it:
    Windows:        app.bat          (or: venv\\Scripts\\python app.py)
    macOS / Linux:  ./app.sh         (or: venv/bin/python app.py)
"""
from __future__ import annotations

import html
import json
import re
import threading
import time
from pathlib import Path

import gradio as gr

import chords
import whisper_core as core
from lyrics import separate_stems
from media import download_audio

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
# Small audio files the karaoke player streams from (see build_karaoke_html).
KARAOKE_DIR = OUTPUT_DIR / "karaoke"
# Where Demucs writes isolated vocals — used to rebuild a song's karaoke later.
SEPARATED_DIR = PROJECT_DIR / "separated"
# One tiny JSON "manifest" per finished job, so you can reopen it from the library.
LIBRARY_DIR = OUTPUT_DIR / "library"
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
KARAOKE_DIR.mkdir(exist_ok=True)
LIBRARY_DIR.mkdir(exist_ok=True)

# Choices shown in the dropdowns ------------------------------------------------
WHISPER_MODELS = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]
DEMUCS_MODELS = ["htdemucs", "htdemucs_ft", "mdx_extra"]
LANGUAGES = [
    "auto", "en", "he", "es", "fr", "de", "it", "pt", "nl", "ru", "uk",
    "ar", "fa", "tr", "hi", "ur", "zh", "ja", "ko", "id", "vi", "pl", "sv", "el",
]
SEGMENT_HEADERS = ["Start", "End", "Text"]

# Loaded models are cached so clicking the button twice doesn't reload from disk.
_MODEL_CACHE: dict = {}


def _get_model(model_name: str, threads):
    """Load (and cache) a Whisper model. Returns (model, device, compute_type)."""
    key = (model_name, int(threads))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = core.load_model(model_name, "auto", "auto", int(threads))
    return _MODEL_CACHE[key]


def _is_model_downloaded(model_name: str) -> bool:
    """True if this model's weights are already on disk (so loading is quick).

    Used only to pick a 'Downloading…' vs 'Loading…' message. faster-whisper
    stores models as models--<org>--faster-whisper-<name>; the org differs per
    model, so we match on the trailing name and confirm the weights (model.bin)
    actually finished downloading.
    """
    try:
        for d in core.MODELS_DIR.glob(f"models--*--faster-whisper-{model_name}"):
            if any(d.glob("snapshots/*/model.bin")):
                return True
    except Exception:
        pass
    return False


def _fmt_size(num) -> str:
    """Human-readable byte size, e.g. 1.24 GB."""
    n = float(num or 0)
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def _download_model_with_progress(model_name, state) -> None:
    """Download a Whisper model from Hugging Face with VISIBLE progress.

    faster-whisper deliberately hides the download bar (it passes a disabled
    tqdm), which is exactly why a first-time download looked silent and "stuck".
    Here we pre-fetch the model ourselves into the SAME cache folder faster-whisper
    uses, with:
      - a real tqdm bar  -> shows the download in the TERMINAL, and
      - a tiny hook       -> reports bytes done/total into `state` for the WEB bar.
    Because the files land in the same cache, the later WhisperModel(...) call
    finds them and does NOT download again.
    """
    import huggingface_hub
    from faster_whisper.utils import _MODELS
    from tqdm.auto import tqdm as _tqdm

    # Same repo + file list faster-whisper itself would use.
    repo_id = model_name if "/" in model_name else _MODELS.get(model_name, model_name)
    allow = ["config.json", "preprocessor_config.json", "model.bin",
             "tokenizer.json", "vocabulary.*"]

    bars: list = []

    class _ReportingTqdm(_tqdm):
        """A normal tqdm (prints to the terminal) that also sums byte progress
        across every file being downloaded and stores it in `state`."""
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            bars.append(self)

        def update(self, n=1):
            r = super().update(n)
            byte_bars = [b for b in bars if getattr(b, "unit", "") == "B"]
            state["downloaded"] = sum(int(b.n or 0) for b in byte_bars)
            state["total"] = sum(int(b.total or 0) for b in byte_bars)
            return r

    print(f"[model] '{model_name}' is not on disk yet — downloading from Hugging Face "
          f"(one-time). Progress below:")
    huggingface_hub.snapshot_download(
        repo_id, cache_dir=str(core.MODELS_DIR), allow_patterns=allow,
        tqdm_class=_ReportingTqdm,
    )
    print(f"[model] download of '{model_name}' complete.")


def _load_model_streaming(model_name, threads, *, pct):
    """Load a model on a side-thread, yielding live progress so the bar never
    *looks* frozen. If the model isn't on disk yet it is downloaded first, and
    the bar shows the real byte progress (e.g. "1.24 GB / 3.09 GB · 40%").

    Returns (model, device, compute_type) — capture it with `yield from`.
    """
    cached = _is_model_downloaded(model_name)
    # `phase` flips from "download" to "load" inside the worker thread.
    state = {"phase": "load" if cached else "download", "downloaded": 0, "total": 0}
    holder: dict = {}

    def _work():
        try:
            if not cached:
                _download_model_with_progress(model_name, state)
            state["phase"] = "load"
            holder["m"] = _get_model(model_name, threads)
        except Exception as exc:
            holder["e"] = exc

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    t0 = time.time()
    while th.is_alive():
        el = int(time.time() - t0)
        if state["phase"] == "download":
            done, total = state["downloaded"], state["total"]
            label = f"⬇️ Downloading the “{model_name}” model…"
            if total > 0:
                sub = (f"{_fmt_size(done)} / {_fmt_size(total)} · {int(100 * done / total)}% "
                       f"· {el}s · one-time download, please keep this open")
            else:
                sub = f"{el}s · contacting Hugging Face… (first-time download)"
        else:
            label = f"Loading the “{model_name}” model into memory…"
            sub = f"{el}s · almost ready…"
        yield {"percent": pct, "label": label, "sub": sub}
        th.join(timeout=0.4)

    if "e" in holder:
        raise gr.Error(f"Could not load the model “{model_name}”: {holder['e']}")
    return holder["m"]


def _norm_lang(language):
    """Turn the dropdown value into what the engine expects (None = auto-detect)."""
    if not language or str(language).lower() in ("auto", "auto-detect"):
        return None
    return str(language).strip()


def _row(seg):
    return [core._ts(seg.start), core._ts(seg.end), seg.text.strip()]


# --------------------------------------------------------------------------- #
# The single progress bar
# --------------------------------------------------------------------------- #
# Instead of Gradio's built-in progress meter (which flickered and looked like
# the page was constantly refreshing), we draw our OWN little bar as plain HTML
# and update just this one thing while work is happening. `percent` is 0..1.
def progress_html(percent: float, label: str, sub: str = "") -> str:
    """Return the HTML for one progress bar: a title, a % and a filling track."""
    pct = max(0, min(100, int(round(float(percent) * 100))))
    return (
        '<div class="pbar-card">'
        '<div class="pbar-head">'
        f'<span class="pbar-label">{html.escape(str(label))}</span>'
        f'<span class="pbar-pct">{pct}%</span>'
        '</div>'
        f'<div class="pbar-track"><div class="pbar-fill" style="width:{pct}%"></div></div>'
        f'<div class="pbar-sub">{html.escape(str(sub))}</div>'
        '</div>'
    )


# --------------------------------------------------------------------------- #
# Karaoke player (self-contained HTML + a tiny bit of JS, wired up on load)
# --------------------------------------------------------------------------- #
def _read_mono(src):
    """Load any audio file to (mono float32 samples, sample_rate).

    Tries soundfile first (wav/flac/ogg); falls back to faster-whisper's bundled
    PyAV decoder for mp3/m4a/webm/opus, so the karaoke player also works for
    those — including YouTube downloads made without a system ffmpeg.
    """
    import numpy as np

    try:
        import soundfile as sf

        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        return np.clip(data.mean(axis=1), -1.0, 1.0), int(sr)
    except Exception:
        from faster_whisper.audio import decode_audio

        sr = 16000
        mono = np.asarray(decode_audio(str(src), sampling_rate=sr), dtype="float32")
        return np.clip(mono, -1.0, 1.0), sr


def _karaoke_audio_file(src):
    """Encode the karaoke audio to a small mono OGG ON DISK and return its path.

    Earlier versions embedded the whole track as a base64 `data:` URI right in
    the results HTML. For a full song that string is ~2.5 MB, and cramming it
    into the single "done" update made that message too big for the browser to
    render — so the results screen never appeared and you were left looking at
    just the page footer. Writing a small file and letting Gradio serve it (see
    build_karaoke_html) keeps that final update tiny and reliable.

    The filename is a hash of the source path + last-modified time, so re-running
    the same song reuses the file (and the browser cache), while a changed source
    re-encodes. Returns None if the audio can't be decoded (player is skipped).
    Writes in chunks — handing libsndfile's Vorbis encoder one big buffer crashes
    it on Windows.
    """
    try:
        import hashlib

        import soundfile as sf

        src = Path(src)
        key = hashlib.md5(
            f"{src.resolve()}:{src.stat().st_mtime_ns}".encode()
        ).hexdigest()[:16]
        out = KARAOKE_DIR / f"{key}.ogg"
        if not out.exists():
            mono, sr = _read_mono(src)
            with sf.SoundFile(str(out), mode="w", samplerate=int(sr), channels=1,
                              format="OGG", subtype="VORBIS") as f:
                step = 32768
                for i in range(0, len(mono), step):
                    f.write(mono[i:i + step])
        return out
    except Exception as exc:
        print(f"[karaoke] could not prepare audio ({exc}); skipping the player.")
        return None


def _gradio_file_url(path) -> str:
    """A URL the browser can stream a local file from, served by Gradio itself.

    Gradio serves files at /gradio_api/file=<path> (with HTTP Range support, so
    seeking / click-to-jump works) as long as the file lives inside one of the
    allowed_paths folders — we pass OUTPUT_DIR in main(). We give it the absolute
    path with forward slashes and percent-encode it so the Windows drive ':' and
    any spaces survive the trip through the URL.
    """
    from urllib.parse import quote

    return "/gradio_api/file=" + quote(Path(path).resolve().as_posix(), safe="/")


def build_karaoke_html(seg_dicts, audio_source, *, alt_source=None,
                       primary_label="🎵 Audio", alt_label="🎤 Vocals") -> str:
    """Build the karaoke widget: an <audio> player + timestamped, clickable lyrics.

    `seg_dicts` is a list of plain dicts: {start, end, text, words:[{start,end,word}]}
    — the same shape whether the song was just made or is being reopened from the
    library. `audio_source` is the track that plays by default (encoded once to a
    small ogg the browser can stream). For songs we pass the INSTRUMENTAL here, so
    it's a true karaoke — you supply the voice. If `alt_source` (the isolated
    vocals) is given, a 🎤/🎹 toggle appears that flips between the two WITHOUT
    losing your place, so you can peek at the melody and switch back. Word-level
    <span>s carry their own start/end so the JS can light up each word as it's
    sung. Returns "" if there's no audio to embed.
    """
    if not audio_source:
        return ""
    audio_file = _karaoke_audio_file(audio_source)
    if not audio_file:
        return ""
    src = _gradio_file_url(audio_file)

    # The optional second track (e.g. the vocals guide), encoded the same way.
    alt_url = ""
    if alt_source:
        alt_file = _karaoke_audio_file(alt_source)
        if alt_file:
            alt_url = _gradio_file_url(alt_file)

    uid = str(int(time.time() * 1000))
    lines = []
    for seg in seg_dicts:
        s = max(float(seg.get("start") or 0.0), 0.0)
        e = max(float(seg.get("end") or s), s)
        words = seg.get("words") or []
        if words:
            inner = "".join(
                '<span class="kara-w" data-s="%.2f" data-e="%.2f">%s</span>' % (
                    max(float(w["start"]) if w.get("start") is not None else s, 0.0),
                    max(float(w["end"]) if w.get("end") is not None else (w.get("start") or s), 0.0),
                    html.escape(w.get("word", "")),
                )
                for w in words
            )
        else:
            inner = html.escape((seg.get("text") or "").strip())
        lines.append('<div class="kara-line" data-s="%.2f" data-e="%.2f">%s</div>' % (s, e, inner))

    body = "\n".join(lines) or '<div class="kara-line">(no lyrics detected)</div>'

    # Both track URLs (and their button labels) ride along as data-* attributes so
    # the JS toggle can swap audio.src instantly. The toggle button only appears
    # when there's a second track to switch to.
    data_attrs = 'data-primary="%s"' % src
    toggle_btn = ""
    if alt_url:
        data_attrs += ' data-alt="%s" data-primary-label="%s" data-alt-label="%s"' % (
            alt_url,
            html.escape(primary_label, quote=True),
            html.escape(alt_label, quote=True),
        )
        # Starting on the primary track, the button offers the OTHER one.
        toggle_btn = '<button class="kara-toggle" type="button">%s</button>' % html.escape(alt_label)

    return (
        '<div class="kara-root" id="kara-%s" data-wired="0" %s>'
        '<div class="kara-bar">'
        '<span class="kara-title">🎤 Press play — words light up · click the waveform or a line to jump</span>'
        '<span class="kara-bar-btns">%s<button class="kara-fs" type="button">⛶ Fullscreen</button></span>'
        '</div>'
        # The <audio> is now just the hidden "engine"; the custom player row below
        # (play button · clickable Audio Waveform · time readout) drives it via JS.
        '<audio class="kara-audio" preload="metadata" src="%s"></audio>'
        '<div class="kara-player">'
        '<button class="kara-play" type="button" aria-label="Play / pause">▶</button>'
        '<canvas class="kara-wave"></canvas>'
        '<span class="kara-time">0:00 / 0:00</span>'
        '</div>'
        '<div class="kara-lyrics">%s</div>'
        '</div>'
    ) % (uid, data_attrs, toggle_btn, src, body)


# Shown above a chord sheet when the detector found nothing usable.
_CHORDS_EMPTY = (
    '<div class="cs-empty">🎸 No chords detected for this track — the chord '
    'finder works best on songs with clear guitar/piano backing.</div>'
)

# A tiny placeholder shown in the chord box for the split-second between the
# results screen appearing and the follow-up .then(_render_sheet) painting the
# real (possibly large) sheet. Keeping the reveal update small is what stops the
# "results never appear / blank screen" failure (see _karaoke_audio_file).
_CHORDS_LOADING = '<div class="cs-empty">🎸 Laying out the chords…</div>'


def build_chord_sheet_html(sheet_model, transpose: int = 0, easy: bool = False) -> str:
    """Render the lyrics-with-chords sheet, applying transpose / easy on the fly.

    `sheet_model` is what chords.align_chords_to_lines produced, plus the key:
        {"key": "Am", "lines": [ {"type":"lyric"/"chords", ...}, ... ]}
    We never re-detect here — this is pure, instant text work — so the Transpose
    slider and the Easy / Original buttons just re-call this function.

    `transpose` shifts every chord (and the key) by N semitones. By default
    (`easy=False`) we show the REAL chords as heard — a hard song stays hard.
    With `easy=True` we find ONE capo position that turns the song into familiar
    open shapes and print the SHAPES YOU FINGER (e.g. "Capo 3" + C G Am F), which
    is what actually makes a tricky song playable.
    """
    if not sheet_model or not sheet_model.get("lines"):
        return _CHORDS_EMPTY

    key = sheet_model.get("key") or ""
    # When you transpose, the key moves too — and that decides sharp vs flat
    # spelling for everything on the page.
    new_key = chords.transpose_chord(key, transpose) if key else ""
    prefer_flats = chords.key_prefers_flats(new_key) if new_key else False

    def sounding(name: str) -> str:
        """The chord as it really sounds at the current transpose (triad if easy)."""
        n = chords.simplify_chord(name) if easy else name
        return chords.transpose_chord(n, transpose, prefer_flats)

    # In easy mode, pick one capo for the whole song and remember the open shape
    # each sounding chord maps to, so the body shows shapes (not concert names).
    capo, shapes = 0, {}
    if easy:
        uniq_sounding = list(dict.fromkeys(
            sounding(c) for c in chords.unique_chords(sheet_model["lines"])))
        capo, shapes = chords.suggest_capo(uniq_sounding)

    def disp(name: str) -> str:
        """One chord exactly as it should print (the capo'd open shape in easy mode)."""
        s = sounding(name)
        return shapes.get(s, s)

    rows = []
    for ln in sheet_model["lines"]:
        if ln["type"] == "chords":
            label = ln.get("label") or ""
            chips = "".join(
                '<span class="cs-chord cs-inline">%s</span> ' % html.escape(disp(c))
                for c in ln["chords"]
            )
            sect = '<span class="cs-section">%s</span> ' % html.escape(label) if label else ""
            rows.append('<div class="cs-line cs-instr">%s%s</div>' % (sect, chips))
        else:
            units = ""
            for chord, word in ln["cells"]:
                w = html.escape((word or "").strip())
                if not w:
                    continue
                c = html.escape(disp(chord)) if chord else ""
                units += ('<span class="cs-unit"><span class="cs-chord">%s</span>'
                          '<span class="cs-word">%s</span></span> ') % (c, w)
            rows.append('<div class="cs-line">%s</div>' % (units or "&nbsp;"))

    # Header: key, the transpose amount, and (in easy mode) the capo to put on.
    head_bits = []
    if new_key:
        head_bits.append('<span class="cs-key">Key: %s</span>' % html.escape(chords.key_long(new_key)))
    if transpose:
        head_bits.append('<span class="cs-tag">Transpose %+d</span>' % transpose)
    if easy:
        if capo > 0:
            head_bits.append('<span class="cs-tag cs-capo">🎸 Easy — put a capo on fret %d, '
                             'then play the shapes below</span>' % capo)
        else:
            head_bits.append('<span class="cs-tag cs-capo">🎸 Easy — already open chords, no capo</span>')
    header = '<div class="cs-head">%s</div>' % "".join(head_bits) if head_bits else ""

    return '<div class="cs-root">%s<div class="cs-body">%s</div></div>' % (header, "\n".join(rows))


# Shown in the results area if the audio couldn't be embedded for the player.
_KARA_EMPTY = (
    '<div class="kara-empty">🎤 The karaoke player isn’t available for this file '
    '(its audio couldn’t be embedded), but your transcript and downloads are right '
    'below.</div>'
)


# --------------------------------------------------------------------------- #
# "My library": remember finished jobs so you can reopen them later
# --------------------------------------------------------------------------- #
# Every time a song or recording finishes we drop a tiny JSON "manifest" into
# output/library/. It records the words + timestamps and WHERE the audio lives —
# everything the karaoke player needs — so reopening a past song is instant (no
# Demucs, no Whisper, no internet). The audio itself isn't copied; we just point
# at it and let build_karaoke_html re-use the small cached .ogg.
def _seg_dicts(result) -> list:
    """Turn a TranscriptResult's segments into plain dicts (JSON-friendly)."""
    out = []
    for seg in result.segments:
        words = [
            {"start": w.start, "end": w.end, "word": w.word}
            for w in (getattr(seg, "words", None) or [])
        ]
        out.append({"start": seg.start, "end": seg.end,
                    "text": seg.text, "words": words})
    return out


def _safe_slug(text: str) -> str:
    """A short, filename-safe version of a title (keeps letters/numbers/_- )."""
    slug = re.sub(r"[^\w\- ]+", "", str(text)).strip().replace(" ", "_")
    return (slug or "item")[:60]


def _save_to_library(kind, title, out_stem, audio_source, seg_dicts, text,
                     files, language="", instrumental="", sheet=None) -> str | None:
    """Write one manifest describing a finished job. Returns its id (or None).

    `instrumental` (the karaoke backing track) and `sheet` (the detected chord
    sheet) are saved too, so reopening a song from the library brings back its
    instrumental karaoke + chords instantly — no Demucs, no re-detection.
    """
    try:
        # If we'd previously auto-imported this same source as a "legacy" entry,
        # drop it now so the library never shows the same song twice.
        for p in LIBRARY_DIR.glob("legacy_*.json"):
            try:
                old = json.loads(p.read_text(encoding="utf-8"))
                if old.get("kind") == kind and old.get("out_stem") == out_stem:
                    p.unlink()
            except Exception:
                pass

        ts = time.time()
        item_id = f"{int(ts)}_{_safe_slug(title)}"
        manifest = {
            "id": item_id,
            "kind": kind,                       # "song" or "speech"
            "title": str(title),
            "out_stem": out_stem,               # used to avoid duplicate entries
            "created": ts,
            "created_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)),
            "language": language or "",
            "audio_source": str(audio_source) if audio_source else "",
            "instrumental": str(instrumental) if instrumental else "",
            "text": text or "",
            "files": [str(f) for f in (files or [])],
            "segments": seg_dicts,
            "sheet": sheet,
        }
        (LIBRARY_DIR / f"{item_id}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return item_id
    except Exception as exc:
        print(f"[library] could not save this job ({exc}).")
        return None


def _load_manifest(item_id: str) -> dict:
    """Read one saved manifest back from disk."""
    return json.loads((LIBRARY_DIR / f"{item_id}.json").read_text(encoding="utf-8"))


def _find_legacy_audio(stem: str, is_song: bool):
    """Best-effort: locate the audio for a result made BEFORE the library existed.

    Songs  -> the isolated vocals Demucs already wrote into separated/.
    Speech -> the matching file still sitting in input/.
    Returns a path or None (None just means 'no karaoke for this old item').
    """
    if is_song:
        for vp in SEPARATED_DIR.glob(f"*/{stem}/vocals.wav"):
            if vp.exists():
                return vp
    else:
        for ip in INPUT_DIR.glob(f"{stem}.*"):
            if ip.suffix.lower() != ".json":
                return ip
    return None


def _import_legacy_outputs() -> None:
    """Create manifests for results made before this feature, so they show up too.

    Scans output/*.json (what the engine has always written) and, for anything
    not already in the library, records a manifest — trying to point it at the
    original vocals/input so its karaoke still works. Idempotent and defensive:
    it never raises, so a stray file can't stop the app from starting.
    """
    try:
        present = set()
        for p in LIBRARY_DIR.glob("*.json"):
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
                present.add((m.get("kind"), m.get("out_stem")))
            except Exception:
                pass

        for jp in sorted(OUTPUT_DIR.glob("*.json")):
            name = jp.stem                         # "song.lyrics" or "recording"
            is_song = name.endswith(".lyrics")
            stem = name[:-len(".lyrics")] if is_song else name
            kind = "song" if is_song else "speech"
            if (kind, stem) in present:
                continue
            marker_id = "legacy_" + _safe_slug(name)
            if (LIBRARY_DIR / f"{marker_id}.json").exists():
                continue
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue

            seg_dicts = [
                {"start": s.get("start") or 0.0, "end": s.get("end") or 0.0,
                 "text": s.get("text") or "", "words": s.get("words") or []}
                for s in (data.get("segments") or [])
            ]
            audio = _find_legacy_audio(stem, is_song)
            files = sorted(
                str(p) for p in OUTPUT_DIR.glob(f"{name}.*")
                if p.suffix.lower() != ".ogg"
            )
            mtime = jp.stat().st_mtime
            manifest = {
                "id": marker_id, "kind": kind, "title": stem, "out_stem": stem,
                "created": mtime,
                "created_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)),
                "language": data.get("language", ""),
                "audio_source": str(audio) if audio else "",
                "text": data.get("text", ""),
                "files": files, "segments": seg_dicts,
            }
            (LIBRARY_DIR / f"{marker_id}.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            present.add((kind, stem))
    except Exception as exc:
        print(f"[library] legacy import skipped ({exc}).")


def _scan_library() -> list:
    """Return [(label, id), ...] for the library dropdown, newest first."""
    _import_legacy_outputs()  # cheap + idempotent: surface old results too
    rows = []
    for p in LIBRARY_DIR.glob("*.json"):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        icon = "🎵" if m.get("kind") == "song" else "🎙️"
        label = f"{icon}  {m.get('title', '(untitled)')}    ·    {m.get('created_str', '')}"
        rows.append((m.get("created", 0), label, m.get("id", p.stem)))
    rows.sort(key=lambda r: r[0], reverse=True)
    return [(label, item_id) for _, label, item_id in rows]


# --------------------------------------------------------------------------- #
# All our custom CSS — ONE real stylesheet, handed to gr.Blocks(css=...)
# --------------------------------------------------------------------------- #
# IMPORTANT: this used to be injected from JavaScript, which was unreliable — when
# it didn't run you got the dreaded "black text on a black box" karaoke. Giving
# the CSS to Gradio directly guarantees it always loads. The karaoke colours are
# deliberately high-contrast and self-contained (the player owns its own bright
# yellow background) so it looks the same and stays readable in any page theme.
CUSTOM_CSS = """
/* ---- center the page a little ---------------------------------------- */
.gradio-container{max-width:1040px!important;margin:0 auto!important}

/* ---- the single progress bar ----------------------------------------- */
.pbar-card{border:1px solid rgba(130,130,170,.28);border-radius:16px;padding:22px 24px;background:rgba(130,130,170,.06)}
.pbar-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:14px}
.pbar-label{font-size:17px;font-weight:650}
.pbar-pct{font-size:14px;font-weight:700;opacity:.7;font-variant-numeric:tabular-nums}
.pbar-track{height:16px;border-radius:999px;background:rgba(130,130,170,.22);overflow:hidden;position:relative}
.pbar-track::before{content:'';position:absolute;inset:0;background:repeating-linear-gradient(45deg,transparent 0 10px,rgba(130,130,170,.16) 10px 20px);background-size:28px 28px;animation:pbar-stripe .9s linear infinite}
@keyframes pbar-stripe{to{background-position:28px 0}}
.pbar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#6d6df0,#9b6df0);width:0;transition:width .35s ease;position:relative;z-index:1;overflow:hidden}
.pbar-fill::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.45),transparent);transform:translateX(-100%);animation:pbar-sheen 1.15s linear infinite}
@keyframes pbar-sheen{to{transform:translateX(100%)}}
.pbar-sub{margin-top:13px;font-size:13px;opacity:.7;min-height:1.1em}

/* ---- karaoke player (self-contained sunny YELLOW theme, high contrast) - */
/* Light gold background + DARK text so the lyrics are easy to read.
   WANT A GREEN SCREEN INSTEAD? Swap the two `background:` gradients below for
   green ones, e.g.
     .kara-root           -> radial-gradient(120% 140% at 50% 0%,#eaffd1 0%,#b6f06a 55%,#8fe03d 100%)
     .kara-root:fullscreen-> radial-gradient(120% 120% at 50% 30%,#dcffb0,#8fe03d)
   The dark text already reads fine on green too, so nothing else needs to change. */
.kara-root{--kara-up:#241a00;--kara-sung:#9b8b48;border:1px solid #e6c34a;border-radius:18px;padding:18px 18px 20px;background:radial-gradient(120% 140% at 50% 0%,#fff7c4 0%,#ffe874 55%,#ffd23f 100%);color:var(--kara-up);font-family:system-ui,-apple-system,'Segoe UI',sans-serif;box-shadow:0 16px 44px rgba(150,115,15,.35)}
.kara-bar{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}
.kara-title{font-size:13px;color:#7a6a2a;font-weight:600}
.kara-fs{cursor:pointer;border:0;border-radius:10px;padding:8px 14px;background:#8a5a00;color:#fff;font-size:13px;font-weight:600;transition:background .15s}
.kara-fs:hover{background:#a86d00}
/* the <audio> is now just the hidden engine; the waveform player drives it */
.kara-audio{display:none}
/* custom player row: play button · clickable waveform · time-on-the-side */
.kara-player{display:flex;align-items:center;gap:12px;margin-bottom:14px}
.kara-play{flex:0 0 auto;width:46px;height:46px;border:0;border-radius:50%;background:#8a5a00;color:#fff;font-size:17px;cursor:pointer;transition:background .15s,transform .08s;box-shadow:0 3px 10px rgba(150,100,0,.3)}
.kara-play:hover{background:#a86d00}
.kara-play:active{transform:scale(.93)}
.kara-wave{flex:1 1 auto;min-width:0;height:54px;cursor:pointer;display:block}
.kara-time{flex:0 0 auto;min-width:88px;text-align:right;font-size:13px;font-weight:700;color:#7a6a2a;font-variant-numeric:tabular-nums}
/* the scrolling lyrics; soft fade at top & bottom so it glides */
.kara-lyrics{max-height:430px;overflow:auto;scroll-behavior:smooth;padding:10px 6px;-webkit-mask-image:linear-gradient(180deg,transparent,#000 9%,#000 91%,transparent);mask-image:linear-gradient(180deg,transparent,#000 9%,#000 91%,transparent)}
/* a line has three states: UPCOMING (default, clearly readable) ·
   NOW SINGING (.on, big + bright + highlighted) · ALREADY SUNG (.sung, dim) */
.kara-line{padding:7px 14px;margin:3px 0;border-radius:12px;font-size:20px;line-height:1.5;font-weight:600;color:var(--kara-up)!important;opacity:.9;cursor:pointer;transition:color .2s,background .2s,transform .2s,opacity .2s}
.kara-line:hover{background:rgba(120,80,0,.12);opacity:1}
.kara-line.sung{color:var(--kara-sung)!important;opacity:.6}
.kara-line.on{color:#1a1300!important;font-size:23px;font-weight:750;opacity:1;background:linear-gradient(90deg,rgba(255,255,255,.6),rgba(255,255,255,.1));box-shadow:inset 3px 0 0 #c47d00,0 2px 10px rgba(150,100,0,.2);transform:scale(1.012)}
/* a single word: upcoming-in-current-line stays darkest so you can read
   ahead; the word being sung gets a hot orange highlight; sung words dim */
.kara-w{border-radius:7px;padding:0 2px;transition:color .12s,background .12s}
.kara-line.on .kara-w:not(.on):not(.sung){color:#1a1300!important}
.kara-w.sung{color:#a89860!important}
.kara-w.on{background:linear-gradient(180deg,#ff8a1e,#e23b00);color:#fff!important;font-weight:800;padding:2px 5px;box-shadow:0 2px 8px rgba(200,60,0,.45)}
.kara-empty{padding:18px;border:1px dashed rgba(150,110,20,.5);border-radius:14px;opacity:.9;font-size:14px;color:#3a2f00}
/* fullscreen = a real karaoke screen */
.kara-root:fullscreen{padding:6vh 8vw;display:flex;flex-direction:column;justify-content:center;background:radial-gradient(120% 120% at 50% 30%,#fff2a8,#ffce3a)}
.kara-root:fullscreen .kara-lyrics{max-height:none;flex:1;text-align:center}
.kara-root:fullscreen .kara-line{font-size:30px}
.kara-root:fullscreen .kara-line.on{font-size:42px}
.kara-root:fullscreen .kara-wave{height:80px}

/* the right-hand buttons in the karaoke title bar (vocals toggle + fullscreen) */
.kara-bar-btns{display:flex;gap:8px;align-items:center;flex:0 0 auto}
.kara-toggle{cursor:pointer;border:0;border-radius:10px;padding:8px 14px;background:#5a3a00;color:#fff;font-size:13px;font-weight:600;transition:background .15s}
.kara-toggle:hover{background:#7a5210}

/* ---- guitar chord sheet (lyrics with chords above the words) ------------- */
.cs-root{border:1px solid rgba(130,130,170,.28);border-radius:16px;padding:16px 18px;background:rgba(130,130,170,.05)}
.cs-head{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid rgba(130,130,170,.2)}
.cs-key{font-size:15px;font-weight:700}
.cs-tag{font-size:13px;font-weight:600;padding:3px 10px;border-radius:999px;background:rgba(130,130,170,.16)}
.cs-capo{background:rgba(179,0,90,.12);color:#b3005a}
/* big line spacing leaves room for a chord sitting above each word */
.cs-body{font-size:17px;line-height:2.6}
.cs-line{margin:2px 0}
.cs-instr{opacity:.85;margin:9px 0}
.cs-section{font-style:italic;opacity:.7;font-size:13px;margin-right:4px}
/* each word + its (optional) chord is one bottom-aligned stack, so the lyrics
   line up no matter which words carry a chord */
.cs-unit{display:inline-block;vertical-align:bottom}
.cs-chord{display:block;height:1.25em;line-height:1.25;font-weight:800;color:#b3005a;font-size:.8em;white-space:pre}
.cs-word{white-space:pre}
.cs-chord.cs-inline{display:inline-block;height:auto;margin-right:6px;font-size:.95em}
.cs-empty{padding:16px;border:1px dashed rgba(130,130,170,.5);border-radius:14px;opacity:.85;font-size:14px}
"""


# --------------------------------------------------------------------------- #
# Karaoke behaviour (the moving highlight + click-to-jump + fullscreen).
# The CSS lives in CUSTOM_CSS above; this only WIRES players up, because
# Gradio's HTML component won't run <script>. We attach behaviour on page load
# and a MutationObserver catches any players that appear later.
# --------------------------------------------------------------------------- #
INIT_JS = r"""
() => {
  if (window.__sonariInit) return;
  window.__sonariInit = true;

  function wire(root) {
    if (root.dataset.wired === '1') return;
    root.dataset.wired = '1';
    const audio = root.querySelector('.kara-audio');
    const box = root.querySelector('.kara-lyrics');
    const lines = Array.prototype.slice.call(root.querySelectorAll('.kara-line'));
    const words = Array.prototype.slice.call(root.querySelectorAll('.kara-w'));
    if (audio) {
      audio.addEventListener('timeupdate', function () {
        const t = audio.currentTime;
        let cur = null;
        for (const l of lines) {
          const s = +l.dataset.s, e = +l.dataset.e;
          const on = (t >= s && t < e);
          l.classList.toggle('on', on);
          l.classList.toggle('sung', t >= e);   // already finished -> dim
          if (on) cur = l;
        }
        for (const w of words) {
          const s = +w.dataset.s, e = +w.dataset.e;
          w.classList.toggle('on', t >= s && t < e);   // singing now -> highlight
          w.classList.toggle('sung', t >= e);          // already sung -> dim
        }
        if (cur && box) {
          const top = cur.offsetTop - box.clientHeight / 2 + cur.clientHeight / 2;
          box.scrollTo({ top: top, behavior: 'smooth' });
        }
      });
    }
    for (const l of lines) {
      l.addEventListener('click', function () {
        if (audio) { audio.currentTime = (+l.dataset.s) + 0.01; audio.play(); }
      });
    }
    const fb = root.querySelector('.kara-fs');
    if (fb) {
      fb.addEventListener('click', function () {
        if (!document.fullscreenElement) {
          (root.requestFullscreen || root.webkitRequestFullscreen).call(root);
        } else {
          document.exitFullscreen();
        }
      });
    }

    // ---- vocals / instrumental toggle ---------------------------------------
    // Swap the <audio> source between the two tracks (e.g. instrumental <-> the
    // vocals guide) WITHOUT losing your place: remember the time + whether it was
    // playing, switch src, then restore. The waveform is rebuilt for the new one.
    const toggleBtn = root.querySelector('.kara-toggle');
    if (toggleBtn && audio && root.dataset.alt) {
      let onPrimary = true;
      toggleBtn.addEventListener('click', function () {
        const t = audio.currentTime, playing = !audio.paused;
        onPrimary = !onPrimary;
        audio.src = onPrimary ? root.dataset.primary : root.dataset.alt;
        audio.addEventListener('loadedmetadata', function once() {
          audio.removeEventListener('loadedmetadata', once);
          try { audio.currentTime = t; } catch (e) {}
          if (playing) audio.play();
          buildWave();
        });
        audio.load();
        // the button always advertises the OTHER track you can switch to
        toggleBtn.textContent = onPrimary ? root.dataset.altLabel : root.dataset.primaryLabel;
      });
    }

    // ---- custom Audio Waveform player (offline: decoded & drawn in-browser) --
    // Replaces the old native time bar. The <audio> above is the hidden engine;
    // here we draw a clickable waveform on a <canvas>, fill it as the song plays
    // and show the time (current / total) on the side. No library, no internet.
    const playBtn = root.querySelector('.kara-play');
    const canvas  = root.querySelector('.kara-wave');
    const timeEl  = root.querySelector('.kara-time');
    // We oversample the audio into a high-res envelope, then draw it as ONE
    // smooth curved shape (not separate bars) so it looks clean. Playback fills
    // it with a moving orange front, redrawn every animation frame for a glide.
    const SAMPLES = 600;
    let peaks = null, lastW = 0, lastH = 0, wavePath = null, raf = 0;

    function fmt(t) {
      if (!isFinite(t) || t < 0) t = 0;
      const m = Math.floor(t / 60), s = Math.floor(t % 60);
      return m + ':' + (s < 10 ? '0' : '') + s;
    }
    // Build the smooth waveform silhouette (one closed Path2D) for this size.
    // The top edge follows the envelope using quadratic curves THROUGH the
    // points (rounded, not jagged); the bottom edge mirrors it, so the whole
    // thing is a single smooth blob we can clip the fill into.
    function buildPath(pw, ph) {
      const path = new Path2D();
      const mid = ph / 2, amp = ph * 0.46;
      const n = peaks ? peaks.length : 2;
      const top = [];
      for (let i = 0; i < n; i++) {
        const x = n < 2 ? 0 : (i / (n - 1)) * pw;
        const p = peaks ? peaks[i] : 0.06;
        top.push({ x: x, y: mid - Math.max(p * amp, 1) });
      }
      const curve = function (pts) {
        for (let i = 0; i < pts.length - 1; i++) {
          const xc = (pts[i].x + pts[i + 1].x) / 2, yc = (pts[i].y + pts[i + 1].y) / 2;
          path.quadraticCurveTo(pts[i].x, pts[i].y, xc, yc);
        }
        const last = pts[pts.length - 1];
        path.lineTo(last.x, last.y);
      };
      const bottom = top.map(function (p) { return { x: p.x, y: mid + (mid - p.y) }; }).reverse();
      path.moveTo(top[0].x, top[0].y);
      curve(top);
      curve(bottom);
      path.closePath();
      return path;
    }
    function render() {
      if (!canvas) return;
      const cx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const pw = Math.max(1, Math.floor((canvas.clientWidth  || 1) * dpr));
      const ph = Math.max(1, Math.floor((canvas.clientHeight || 1) * dpr));
      if (pw !== lastW || ph !== lastH) { canvas.width = pw; canvas.height = ph; lastW = pw; lastH = ph; wavePath = null; }
      if (!wavePath) wavePath = buildPath(pw, ph);
      cx.clearRect(0, 0, pw, ph);
      const prog = (audio && audio.duration) ? (audio.currentTime / audio.duration) : 0;
      const px = Math.max(0, Math.min(pw, prog * pw));
      cx.save();
      cx.clip(wavePath);                       // keep every fill inside the curve
      cx.fillStyle = 'rgba(122,106,42,.40)';   // not-yet-played part (muted gold)
      cx.fillRect(0, 0, pw, ph);
      if (px > 0) {                            // already-played part (orange)
        const g = cx.createLinearGradient(0, 0, 0, ph);
        g.addColorStop(0, '#ff8a1e'); g.addColorStop(1, '#e23b00');
        cx.fillStyle = g;
        cx.fillRect(0, 0, px, ph);
      }
      cx.restore();
      if (timeEl && audio) timeEl.textContent = fmt(audio.currentTime) + ' / ' + fmt(audio.duration || 0);
    }
    // Animate only while playing (requestAnimationFrame = smooth ~60fps glide);
    // when paused we just render once on demand, so it costs nothing idle.
    function loop() { render(); raf = (audio && !audio.paused && !audio.ended) ? requestAnimationFrame(loop) : 0; }
    function kick() { if (!raf) raf = requestAnimationFrame(loop); }

    async function buildWave() {
      try {
        const buf = await (await fetch(audio.src)).arrayBuffer();
        const AC  = window.AudioContext || window.webkitAudioContext;
        const ac  = new AC();
        const dec = await ac.decodeAudioData(buf);
        if (ac.close) ac.close();
        const data = dec.getChannelData(0);
        const block = Math.max(1, Math.floor(data.length / SAMPLES));
        const raw = new Array(SAMPLES); let max = 1e-4;
        for (let i = 0; i < SAMPLES; i++) {
          let peak = 0; const start = i * block;
          for (let j = 0; j < block; j++) { const v = Math.abs(data[start + j] || 0); if (v > peak) peak = v; }
          raw[i] = peak; if (peak > max) max = peak;
        }
        // normalise + lightly smooth (5-wide moving average) to kill the jaggies
        const sm = new Array(SAMPLES);
        for (let i = 0; i < SAMPLES; i++) {
          let acc = 0, cnt = 0;
          for (let k = -2; k <= 2; k++) { const j = i + k; if (j >= 0 && j < SAMPLES) { acc += raw[j]; cnt++; } }
          sm[i] = (acc / cnt) / max;
        }
        peaks = sm;
      } catch (e) { peaks = null; }   // decode failed -> flat line; playback still fine
      wavePath = null;
      render();
    }

    if (playBtn && audio) {
      playBtn.addEventListener('click', function () { audio.paused ? audio.play() : audio.pause(); });
      audio.addEventListener('play',  function () { playBtn.textContent = '⏸'; kick(); });
      audio.addEventListener('pause', function () { playBtn.textContent = '▶'; render(); });
      audio.addEventListener('ended', function () { playBtn.textContent = '▶'; render(); });
    }
    if (canvas && audio) {
      canvas.addEventListener('click', function (ev) {
        const rect = canvas.getBoundingClientRect();
        const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
        if (audio.duration) { audio.currentTime = frac * audio.duration; render(); }
      });
    }
    if (audio) {
      audio.addEventListener('seeked', render);
      audio.addEventListener('timeupdate', render);   // keeps time text live when paused
      audio.addEventListener('loadedmetadata', render);
    }
    window.addEventListener('resize', function () { wavePath = null; render(); });
    render();
    buildWave();
  }
  function scan() { for (const r of document.querySelectorAll('.kara-root')) wire(r); }
  scan();
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
}
"""


# --------------------------------------------------------------------------- #
# The "engine" generators
# --------------------------------------------------------------------------- #
# These do the real work and report progress by yielding tiny dictionaries
# ("events"). They know NOTHING about Gradio or which boxes things go in — that
# keeps them simple. The UI layer (_pump, below) decides where each value lands.
#
#   while working:   {"percent": 0..1, "label": "...", "sub": "..."}
#   right at the end:{"done": True, "text", "rows", "files", "kara", "summary"}
#                    (songs also add "vocals": <path>)
# --------------------------------------------------------------------------- #
def _process_speech(audio_path, model_name, language, task, beam_size,
                    vad, word_ts, prompt, threads, title=None):
    """Engine generator for plain speech."""
    # Loading the model can download a few GB the first time — stream a live
    # elapsed counter through it so the bar never looks stuck.
    model, device, _ = yield from _load_model_streaming(model_name, threads, pct=0.02)

    segments, info = [], None
    t0 = time.time()
    yield {"percent": 0.05, "label": "Listening to the audio…", "sub": ""}

    for kind, payload in core.transcribe_stream(
        model, audio_path,
        language=_norm_lang(language), task=task, beam_size=int(beam_size),
        vad=vad, word_timestamps=word_ts, initial_prompt=(prompt or None),
    ):
        if kind == "info":
            info = payload
            yield {"percent": 0.08, "label": "Transcribing…",
                   "sub": f"{info.language} · {info.duration:.0f}s of audio · running on {device}"}
        else:
            seg = payload
            segments.append(seg)
            if info and info.duration:
                frac = min(seg.end / info.duration, 1.0)
                sub = f"{core._ts(seg.end)} / {core._ts(info.duration)}"
            else:
                frac, sub = 0.5, ""
            # transcription occupies 8% -> 98% of the bar
            yield {"percent": 0.08 + 0.90 * frac, "label": "Transcribing…", "sub": sub}

    result = core._build_result(info, segments, time.time() - t0)
    written = core.write_all(result, OUTPUT_DIR, Path(audio_path).stem,
                             ("txt", "srt", "vtt", "json"))
    seg_dicts = _seg_dicts(result)
    kara = build_karaoke_html(seg_dicts, audio_path)
    _save_to_library("speech", title or Path(audio_path).stem, Path(audio_path).stem,
                     audio_path, seg_dicts, result.text, written,
                     info.language if info else "")
    yield {"done": True,
           "text": result.text,
           "rows": [_row(s) for s in segments],
           "files": [str(p) for p in written],
           "kara": kara,
           "summary": f"### ✅ Done\n{info.language} · {len(segments)} segments · {result.elapsed:.0f}s"}


def _process_lyrics(audio_path, whisper_model, demucs_model, language,
                    max_seconds, vad, threads, title=None):
    """Engine generator for songs: separate the vocals, then read the lyrics."""
    # Estimate how long separation will take so the bar can keep moving during it.
    try:
        import soundfile as sf
        total = sf.info(str(audio_path)).duration
    except Exception:
        total = 0.0
    if max_seconds and float(max_seconds) > 0 and total:
        total = min(total, float(max_seconds))
    est = max(15.0, (total or 30.0) * 4.0)

    # Demucs is one long blocking call with no progress callback, so we run it on
    # a side thread and show a live elapsed-seconds counter while it works.
    holder: dict = {}

    def _work():
        try:
            holder["s"] = separate_stems(Path(audio_path), demucs_model, float(max_seconds or 0))
        except Exception as exc:
            holder["e"] = exc

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    t0 = time.time()
    yield {"percent": 0.02, "label": "Separating the vocals from the music…",
           "sub": "This is the slow step on a CPU — please wait."}
    while th.is_alive():
        el = time.time() - t0
        # separation occupies 2% -> ~52% of the bar (time-estimate based)
        yield {"percent": 0.02 + 0.50 * min(el / est, 0.95),
               "label": "Separating the vocals from the music…",
               "sub": f"{int(el)}s elapsed — the slow step on a CPU, please wait."}
        th.join(timeout=0.8)
    if "e" in holder:
        raise gr.Error(f"Vocal separation failed: {holder['e']}")
    stems = holder["s"]
    vocals = stems.vocals

    # Same here: the lyrics model can be a big first-time download, so stream a
    # live counter while it loads instead of freezing the bar at one spot.
    model, _, _ = yield from _load_model_streaming(whisper_model, threads, pct=0.54)

    segments, info = [], None
    t1 = time.time()
    for kind, payload in core.transcribe_stream(
        model, vocals,
        language=_norm_lang(language), beam_size=5, vad=vad, word_timestamps=True,
        # Singing confuses "use previous text as context" -> repeats. Off for music.
        condition_on_previous_text=False,
    ):
        if kind == "info":
            info = payload
            yield {"percent": 0.56, "label": "Reading the lyrics…",
                   "sub": f"{info.language} · vocals {info.duration:.0f}s"}
        else:
            seg = payload
            segments.append(seg)
            if info and info.duration:
                frac = min(seg.end / info.duration, 1.0)
                sub = f"{core._ts(seg.end)} / {core._ts(info.duration)}"
            else:
                frac, sub = 0.5, ""
            # reading lyrics occupies 56% -> ~98% of the bar
            yield {"percent": 0.56 + 0.42 * frac, "label": "Reading the lyrics…", "sub": sub}

    result = core._build_result(info, segments, time.time() - t1)
    written = core.write_all(result, OUTPUT_DIR, f"{Path(audio_path).stem}.lyrics",
                             ("txt", "srt", "json"))
    seg_dicts = _seg_dicts(result)

    # Work out the guitar chords from the INSTRUMENTAL (quick next to Demucs) and
    # lay them above the words. A chord hiccup must never sink the whole result.
    yield {"percent": 0.98, "label": "Working out the chords…", "sub": "almost done"}
    try:
        chord_data = chords.detect_chords(stems.instrumental, float(max_seconds or 0))
    except Exception as exc:
        print(f"[chords] detection failed ({exc}); skipping the chord sheet.")
        chord_data = {"key": "", "spans": []}
    sheet_model = {"key": chord_data["key"],
                   "lines": chords.align_chords_to_lines(seg_dicts, chord_data["spans"])}
    chord_html = build_chord_sheet_html(sheet_model)

    # Karaoke plays the INSTRUMENTAL (true karaoke); the toggle reveals the vocals.
    kara = build_karaoke_html(seg_dicts, stems.instrumental, alt_source=vocals,
                              primary_label="🎹 Instrumental", alt_label="🎤 Vocals guide")
    _save_to_library("song", title or Path(audio_path).stem, Path(audio_path).stem,
                     vocals, seg_dicts, result.text, written,
                     info.language if info else "",
                     instrumental=stems.instrumental, sheet=sheet_model)
    yield {"done": True,
           "text": result.text,
           "rows": [_row(s) for s in segments],
           "files": [str(p) for p in written],
           "kara": kara,
           "chords": chord_html,
           "sheet": sheet_model,
           "vocals": str(vocals),
           "summary": f"### ✅ Lyrics ready\n{len(segments)} lines · {result.elapsed:.0f}s (＋ separation time)"}


# --------------------------------------------------------------------------- #
# UI plumbing: drive an engine generator and route its events to components
# --------------------------------------------------------------------------- #
def _pump(gen, *, progress, results, pbar, summary, text, segs, files, kara,
          vocals=None, chord_box=None, sheet=None, tr=None, easy=None):
    """Run `gen` and send each event to the right Gradio component.

    While it works, ONLY the progress bar (`pbar`) changes — that's why nothing
    flickers any more. When the engine signals "done", we hide the progress
    screen, reveal the results screen, and fill everything in at once.

    `chord_box`/`sheet`/`tr`/`easy` are the optional chord-sheet bits (songs only):
    the box to paint into, the State holding the chord data, the Transpose slider,
    and the easy-vs-original State — all reset for each fresh result.

    IMPORTANT: we deliberately do NOT pour the (often large) chord-sheet HTML into
    this "done" update. A too-big reveal message stops the results screen from ever
    appearing — the classic "song finishes but the screen stays blank" bug (the
    same trap documented in _karaoke_audio_file). We only stash the small `sheet`
    data here; a follow-up .then(_render_sheet) paints the chords a beat later.
    """
    for ev in gen:
        if ev.get("done"):
            out = {
                progress: gr.update(visible=False),
                results: gr.update(visible=True),
                summary: ev.get("summary", "### ✅ Done"),
                text: ev["text"],
                segs: ev["rows"],
                files: ev["files"],
                kara: ev["kara"] or _KARA_EMPTY,
            }
            if vocals is not None:
                out[vocals] = ev.get("vocals")
            if chord_box is not None:
                out[chord_box] = _CHORDS_LOADING if ev.get("sheet") else _CHORDS_EMPTY
            if sheet is not None:
                out[sheet] = ev.get("sheet")
            if tr is not None:
                out[tr] = gr.update(value=0)
            if easy is not None:
                out[easy] = False  # back to "Original" for a fresh song
            yield out
        else:
            yield {pbar: progress_html(ev["percent"], ev["label"], ev.get("sub", ""))}


def _go_setup():
    """Send the user back to the setup screen (used by the 'start over' buttons)."""
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


def _render_sheet(sheet_model, transpose, easy):
    """Redraw the chord sheet for the current transpose / easy settings.

    Wired to the Transpose slider, the Easy/Original buttons, and the post-reveal
    .then() that paints the sheet. If there's no sheet (e.g. a speech item, or
    chords weren't detected) we leave whatever's shown untouched rather than
    blanking it.
    """
    if not sheet_model:
        return gr.update()
    return build_chord_sheet_html(sheet_model, int(transpose), bool(easy))


def _set_easy(sheet_model, transpose):
    """The 'Easy chords' button: switch to the capo/open-shape view and remember it."""
    return True, _render_sheet(sheet_model, transpose, True)


def _set_original(sheet_model, transpose):
    """The 'Original' button: switch back to the real chords as heard."""
    return False, _render_sheet(sheet_model, transpose, False)


# --------------------------------------------------------------------------- #
# UI layout
# --------------------------------------------------------------------------- #
def _model_dropdown(label="Whisper model", value="large-v3-turbo"):
    return gr.Dropdown(WHISPER_MODELS, value=value, label=label,
                       info="turbo = fast + great (recommended on a CPU). large-v3 = top "
                            "accuracy but a ~3 GB download and much slower on CPU. "
                            "A new model downloads once (needs internet).")


def _lang_dropdown():
    return gr.Dropdown(LANGUAGES, value="auto", label="Language",
                       allow_custom_value=True,
                       info="auto-detect, or type a code like en / he / es")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Sonari") as demo:
        gr.Markdown(
            "# 🎙️🎵 Sonari\n"
            "Turn audio into text — **speech** or **song lyrics** — running fully "
            "on your own computer. Pick a file (or a YouTube link) and your options, "
            "press the button, watch the progress bar, then follow along in the "
            "**karaoke player**."
        )

        # ================= TAB 1: Speech → text ========================= #
        with gr.Tab("🎙️ Speech → text"):
            # ---- SCREEN 1: setup ----
            with gr.Column() as sp_setup:
                sp_audio = gr.Audio(sources=["upload", "microphone"],
                                    type="filepath", label="Audio / video file or microphone recording")
                sp_model = _model_dropdown()
                with gr.Row():
                    sp_lang = _lang_dropdown()
                    sp_task = gr.Radio(["transcribe", "translate"], value="transcribe",
                                       label="Task", info="translate = output English")
                with gr.Accordion("Advanced options", open=False):
                    sp_beam = gr.Slider(1, 10, value=5, step=1, label="Beam size (higher = more accurate, slower)")
                    sp_vad = gr.Checkbox(value=True, label="Skip silence (voice-activity detection)")
                    sp_words = gr.Checkbox(value=True, label="Word-level timestamps (needed for karaoke)")
                    sp_prompt = gr.Textbox(label="Hint words / spellings (names, jargon)", placeholder="optional")
                    sp_threads = gr.Slider(0, 16, value=0, step=1, label="CPU threads (0 = auto)")
                sp_btn = gr.Button("Transcribe  →", variant="primary", size="lg")

            # ---- SCREEN 2a: progress ----
            with gr.Column(visible=False) as sp_progress:
                sp_pbar = gr.HTML(progress_html(0.0, "Getting ready…"))
                gr.Markdown("*Working on your own computer — you can leave this tab open.*")

            # ---- SCREEN 2b: results (revealed when finished) ----
            with gr.Column(visible=False) as sp_results:
                with gr.Row():
                    sp_summary = gr.Markdown()
                    sp_back = gr.Button("⬅ New transcription", size="sm")
                with gr.Accordion("🎤 Karaoke player", open=True):
                    sp_kara = gr.HTML()
                sp_text = gr.Textbox(label="Transcript", lines=10)
                with gr.Accordion("Segments & downloads", open=False):
                    sp_segs = gr.Dataframe(headers=SEGMENT_HEADERS, wrap=True,
                                           label="Segments (with timestamps)")
                    sp_files = gr.File(label="Download (txt / srt / vtt / json)", file_count="multiple")

            def ui_speech(audio, model_name, language, task, beam, vad, words, prompt, threads):
                if not audio:
                    raise gr.Error("Please choose a file or record from your microphone first.")
                # flip to the progress screen straight away …
                yield {sp_setup: gr.update(visible=False),
                       sp_progress: gr.update(visible=True),
                       sp_results: gr.update(visible=False),
                       sp_pbar: progress_html(0.0, "Getting ready…", "Warming up.")}
                # … then let the engine run, routing its events to the components.
                gen = _process_speech(audio, model_name, language, task, beam, vad, words, prompt, threads)
                yield from _pump(gen, progress=sp_progress, results=sp_results, pbar=sp_pbar,
                                 summary=sp_summary, text=sp_text, segs=sp_segs, files=sp_files, kara=sp_kara)

            sp_btn.click(
                ui_speech,
                inputs=[sp_audio, sp_model, sp_lang, sp_task, sp_beam, sp_vad, sp_words, sp_prompt, sp_threads],
                outputs=[sp_setup, sp_progress, sp_results, sp_pbar, sp_summary, sp_text, sp_segs, sp_files, sp_kara],
                show_progress="hidden",  # we draw our own bar; hide Gradio's flickery one
            )
            sp_back.click(_go_setup, outputs=[sp_setup, sp_progress, sp_results])

        # ================= TAB 2: Song → lyrics ========================= #
        with gr.Tab("🎵 Song → lyrics"):
            # ---- SCREEN 1: setup ----
            with gr.Column() as ly_setup:
                ly_audio = gr.Audio(sources=["upload"], type="filepath", label="Song file")
                ly_model = _model_dropdown()
                with gr.Row():
                    ly_demucs = gr.Dropdown(DEMUCS_MODELS, value="htdemucs", label="Vocal separator",
                                            info="htdemucs = default. _ft = slower but cleaner.")
                    ly_lang = _lang_dropdown()
                ly_max = gr.Slider(0, 120, value=0, step=5,
                                   label="Test only first N seconds (0 = whole song)")
                ly_vad = gr.Checkbox(value=False, label="Skip silence (VAD)",
                                     info="Leave OFF for songs — ON can drop repeated lines and mangle singing.")
                with gr.Accordion("Advanced options", open=False):
                    ly_threads = gr.Slider(0, 16, value=0, step=1, label="CPU threads (0 = auto)")
                ly_btn = gr.Button("Extract lyrics  →", variant="primary", size="lg")
                gr.Markdown("_Separating vocals is the slow part on a CPU — a 3–4 min song "
                            "can take several minutes. Use the slider to test the first 30s first._")

            # ---- SCREEN 2a: progress ----
            with gr.Column(visible=False) as ly_progress:
                ly_pbar = gr.HTML(progress_html(0.0, "Getting ready…"))
                gr.Markdown("*The vocal-separation step is the slow one on a CPU — "
                            "the bar keeps moving while it works.*")

            # ---- SCREEN 2b: results ----
            with gr.Column(visible=False) as ly_results:
                with gr.Row():
                    ly_summary = gr.Markdown()
                    ly_back = gr.Button("⬅ New song", size="sm")
                with gr.Accordion("🎤 Karaoke player", open=True):
                    ly_kara = gr.HTML()
                with gr.Accordion("🎸 Guitar chords", open=True):
                    with gr.Row():
                        ly_transpose = gr.Slider(-6, 6, value=0, step=1,
                                                 label="Transpose — move the chords up / down (semitones)")
                        ly_easy_btn = gr.Button("🎸 Easy (capo)", size="sm")
                        ly_orig_btn = gr.Button("🎵 Original", size="sm")
                    ly_chords = gr.HTML()
                    ly_sheet = gr.State()            # the chord data the buttons re-render
                    ly_easy_state = gr.State(False)  # remembers Easy vs Original
                ly_text = gr.Textbox(label="Lyrics", lines=10)
                with gr.Accordion("Lines, isolated vocals & downloads", open=False):
                    ly_segs = gr.Dataframe(headers=SEGMENT_HEADERS, wrap=True,
                                           label="Lines (with timestamps)")
                    ly_vocals = gr.Audio(label="Isolated vocals", type="filepath", interactive=False)
                    ly_files = gr.File(label="Download (txt / srt / json)", file_count="multiple")

            def ui_lyrics(audio, whisper_model, demucs_model, language, max_seconds, vad, threads):
                if not audio:
                    raise gr.Error("Please upload a song first.")
                yield {ly_setup: gr.update(visible=False),
                       ly_progress: gr.update(visible=True),
                       ly_results: gr.update(visible=False),
                       ly_pbar: progress_html(0.0, "Getting ready…")}
                gen = _process_lyrics(audio, whisper_model, demucs_model, language, max_seconds, vad, threads)
                yield from _pump(gen, progress=ly_progress, results=ly_results, pbar=ly_pbar,
                                 summary=ly_summary, text=ly_text, segs=ly_segs, files=ly_files,
                                 kara=ly_kara, vocals=ly_vocals,
                                 chord_box=ly_chords, sheet=ly_sheet, tr=ly_transpose, easy=ly_easy_state)

            ly_btn.click(
                ui_lyrics,
                inputs=[ly_audio, ly_model, ly_demucs, ly_lang, ly_max, ly_vad, ly_threads],
                outputs=[ly_setup, ly_progress, ly_results, ly_pbar, ly_summary, ly_text,
                         ly_segs, ly_files, ly_kara, ly_vocals,
                         ly_chords, ly_sheet, ly_transpose, ly_easy_state],
                show_progress="hidden",
            ).then(  # paint the (large) chord sheet AFTER the screen is shown — keeps the reveal small
                _render_sheet, inputs=[ly_sheet, ly_transpose, ly_easy_state], outputs=[ly_chords]
            )
            ly_back.click(_go_setup, outputs=[ly_setup, ly_progress, ly_results])
            # Transpose / Easy / Original just re-draw the (already detected) chords — instant.
            ly_transpose.change(_render_sheet, inputs=[ly_sheet, ly_transpose, ly_easy_state], outputs=[ly_chords])
            ly_easy_btn.click(_set_easy, inputs=[ly_sheet, ly_transpose], outputs=[ly_easy_state, ly_chords])
            ly_orig_btn.click(_set_original, inputs=[ly_sheet, ly_transpose], outputs=[ly_easy_state, ly_chords])

        # ================= TAB 3: From a link (YouTube) ================= #
        with gr.Tab("▶️ From a link (YouTube)"):
            # ---- SCREEN 1: setup ----
            with gr.Column() as yt_setup:
                yt_url = gr.Textbox(label="Link", placeholder="https://www.youtube.com/watch?v=…")
                yt_mode = gr.Radio(["Song → lyrics", "Speech → transcript"],
                                   value="Song → lyrics", label="What is it?")
                yt_model = _model_dropdown()
                with gr.Row():
                    yt_demucs = gr.Dropdown(DEMUCS_MODELS, value="htdemucs", label="Vocal separator (songs)")
                    yt_lang = _lang_dropdown()
                yt_task = gr.Radio(["transcribe", "translate"], value="transcribe",
                                   label="Task (speech mode)", info="translate = output English")
                yt_max = gr.Slider(0, 120, value=0, step=5,
                                   label="Test only first N seconds (0 = whole thing)")
                yt_vad = gr.Checkbox(value=False, label="Skip silence (VAD)",
                                     info="Leave OFF for songs.")
                with gr.Accordion("Advanced options", open=False):
                    yt_threads = gr.Slider(0, 16, value=0, step=1, label="CPU threads (0 = auto)")
                yt_btn = gr.Button("Download & process  →", variant="primary", size="lg")
                gr.Markdown("_Downloading needs internet. Only download audio you're allowed to use._")

            # ---- SCREEN 2a: progress ----
            with gr.Column(visible=False) as yt_progress:
                yt_pbar = gr.HTML(progress_html(0.0, "Getting ready…"))
                gr.Markdown("*Downloads first, then the same steps as the other tabs.*")

            # ---- SCREEN 2b: results ----
            with gr.Column(visible=False) as yt_results:
                yt_title = gr.Markdown()
                with gr.Row():
                    yt_summary = gr.Markdown()
                    yt_back = gr.Button("⬅ New link", size="sm")
                with gr.Accordion("🎤 Karaoke player", open=True):
                    yt_kara = gr.HTML()
                with gr.Accordion("🎸 Guitar chords (songs)", open=True):
                    with gr.Row():
                        yt_transpose = gr.Slider(-6, 6, value=0, step=1,
                                                 label="Transpose — move the chords up / down (semitones)")
                        yt_easy_btn = gr.Button("🎸 Easy (capo)", size="sm")
                        yt_orig_btn = gr.Button("🎵 Original", size="sm")
                    yt_chords = gr.HTML()
                    yt_sheet = gr.State()
                    yt_easy_state = gr.State(False)
                yt_text = gr.Textbox(label="Result", lines=10)
                with gr.Accordion("Segments, isolated vocals & downloads", open=False):
                    yt_segs = gr.Dataframe(headers=SEGMENT_HEADERS, wrap=True,
                                           label="Segments (with timestamps)")
                    yt_vocals = gr.Audio(label="Isolated vocals (songs)", type="filepath", interactive=False)
                    yt_files = gr.File(label="Download", file_count="multiple")

            def ui_youtube(url, mode, model_name, demucs_model, language, task, max_seconds, vad, threads):
                if not (url or "").strip():
                    raise gr.Error("Paste a link (for example a YouTube URL) first.")
                yield {yt_setup: gr.update(visible=False),
                       yt_progress: gr.update(visible=True),
                       yt_results: gr.update(visible=False),
                       yt_title: "",
                       yt_vocals: None,
                       yt_chords: "",     # clear any chords from a previous run
                       yt_sheet: None,    # and forget the previous song's chord data
                       yt_pbar: progress_html(0.0, "Downloading audio from the link…", "Needs internet.")}
                try:
                    audio, title = download_audio(url.strip(), INPUT_DIR)
                except Exception as exc:  # surface a clean message in the UI
                    raise gr.Error(f"Could not download that link: {exc}")

                yield {yt_title: f"### 🎬 {title}",
                       yt_pbar: progress_html(0.03, "Got the audio — starting…", title)}

                if mode.startswith("Song"):
                    gen = _process_lyrics(str(audio), model_name, demucs_model, language,
                                          max_seconds, vad, threads, title=title)
                    yield from _pump(gen, progress=yt_progress, results=yt_results, pbar=yt_pbar,
                                     summary=yt_summary, text=yt_text, segs=yt_segs, files=yt_files,
                                     kara=yt_kara, vocals=yt_vocals,
                                     chord_box=yt_chords, sheet=yt_sheet, tr=yt_transpose, easy=yt_easy_state)
                else:
                    gen = _process_speech(str(audio), model_name, language, task, 5, vad, True, None,
                                          threads, title=title)
                    yield from _pump(gen, progress=yt_progress, results=yt_results, pbar=yt_pbar,
                                     summary=yt_summary, text=yt_text, segs=yt_segs, files=yt_files, kara=yt_kara)

            yt_btn.click(
                ui_youtube,
                inputs=[yt_url, yt_mode, yt_model, yt_demucs, yt_lang, yt_task, yt_max, yt_vad, yt_threads],
                outputs=[yt_setup, yt_progress, yt_results, yt_pbar, yt_title, yt_summary, yt_text,
                         yt_segs, yt_files, yt_kara, yt_vocals,
                         yt_chords, yt_sheet, yt_transpose, yt_easy_state],
                show_progress="hidden",
            ).then(  # paint the chord sheet AFTER the screen is shown — keeps the reveal small
                _render_sheet, inputs=[yt_sheet, yt_transpose, yt_easy_state], outputs=[yt_chords]
            )
            yt_back.click(_go_setup, outputs=[yt_setup, yt_progress, yt_results])
            yt_transpose.change(_render_sheet, inputs=[yt_sheet, yt_transpose, yt_easy_state], outputs=[yt_chords])
            yt_easy_btn.click(_set_easy, inputs=[yt_sheet, yt_transpose], outputs=[yt_easy_state, yt_chords])
            yt_orig_btn.click(_set_original, inputs=[yt_sheet, yt_transpose], outputs=[yt_easy_state, yt_chords])

        # ================= TAB 4: My library =========================== #
        with gr.Tab("📂 My library") as lib_tab:
            gr.Markdown(
                "### 📂 Open something you already made\n"
                "Every song and recording you finish is saved here automatically. "
                "Pick one and press **Open** to bring its **karaoke player** back "
                "instantly — no waiting, no internet."
            )
            with gr.Row():
                lib_pick = gr.Dropdown(choices=_scan_library(), value=None, scale=5,
                                       label="Your saved items (newest first)")
                lib_refresh = gr.Button("⟳ Refresh", scale=1)
            lib_open = gr.Button("Open  ▶", variant="primary", size="lg")
            gr.Markdown("_Empty? Make a song or a transcript in the other tabs first — it "
                        "shows up here the moment it's done (press Refresh if you don't see it)._")

            # ---- the reopened result (hidden until you press Open) ----
            with gr.Column(visible=False) as lib_results:
                lib_title = gr.Markdown()
                with gr.Accordion("🎤 Karaoke player", open=True):
                    lib_kara = gr.HTML()
                with gr.Accordion("🎸 Guitar chords", open=True):
                    with gr.Row():
                        lib_transpose = gr.Slider(-6, 6, value=0, step=1,
                                                  label="Transpose — move the chords up / down (semitones)")
                        lib_easy_btn = gr.Button("🎸 Easy (capo)", size="sm")
                        lib_orig_btn = gr.Button("🎵 Original", size="sm")
                    lib_chords = gr.HTML()
                    lib_sheet = gr.State()
                    lib_easy_state = gr.State(False)
                lib_text = gr.Textbox(label="Text", lines=10)
                with gr.Accordion("Lines & downloads", open=False):
                    lib_segs = gr.Dataframe(headers=SEGMENT_HEADERS, wrap=True,
                                            label="Lines (with timestamps)")
                    lib_files = gr.File(label="Download", file_count="multiple")

            def _refresh_library():
                """Re-scan the library folder and repopulate the dropdown."""
                return gr.update(choices=_scan_library())

            def _open_library(item_id):
                """Rebuild a saved job's karaoke + chords + transcript from its manifest."""
                if not item_id:
                    raise gr.Error("Pick a saved item from the list first.")
                m = _load_manifest(item_id)
                seg_dicts = m.get("segments") or []
                # Songs play the instrumental with a vocals-guide toggle; older saves
                # (and speech) only have one track, so fall back to that.
                instrumental = m.get("instrumental") or ""
                vocals = m.get("audio_source") or ""
                if m.get("kind") == "song" and instrumental:
                    kara = build_karaoke_html(seg_dicts, instrumental, alt_source=vocals,
                                              primary_label="🎹 Instrumental",
                                              alt_label="🎤 Vocals guide") or _KARA_EMPTY
                else:
                    kara = build_karaoke_html(seg_dicts, vocals) or _KARA_EMPTY
                # Chords come straight from the saved sheet (older saves have none).
                # We stash the sheet here but let the follow-up .then(_render_sheet)
                # paint it, so this reveal update stays small (no blank screen).
                sheet_model = m.get("sheet")
                rows = [[core._ts(s.get("start") or 0.0), core._ts(s.get("end") or 0.0),
                         (s.get("text") or "").strip()] for s in seg_dicts]
                files = [f for f in (m.get("files") or []) if Path(f).exists()]
                icon = "🎵" if m.get("kind") == "song" else "🎙️"
                title_md = f"### {icon} {m.get('title', '(untitled)')}\n*saved {m.get('created_str', '')}*"
                return {lib_results: gr.update(visible=True),
                        lib_title: title_md, lib_kara: kara, lib_text: m.get("text", ""),
                        lib_segs: rows, lib_files: files,
                        lib_chords: _CHORDS_LOADING if sheet_model else _CHORDS_EMPTY,
                        lib_sheet: sheet_model,
                        lib_transpose: gr.update(value=0), lib_easy_state: False}

            lib_refresh.click(_refresh_library, outputs=[lib_pick])
            lib_tab.select(_refresh_library, outputs=[lib_pick])  # auto-refresh on open
            lib_open.click(_open_library, inputs=[lib_pick],
                           outputs=[lib_results, lib_title, lib_kara, lib_text, lib_segs, lib_files,
                                    lib_chords, lib_sheet, lib_transpose, lib_easy_state]
                           ).then(  # paint the chord sheet after the screen is shown
                               _render_sheet, inputs=[lib_sheet, lib_transpose, lib_easy_state],
                               outputs=[lib_chords])
            lib_transpose.change(_render_sheet, inputs=[lib_sheet, lib_transpose, lib_easy_state], outputs=[lib_chords])
            lib_easy_btn.click(_set_easy, inputs=[lib_sheet, lib_transpose], outputs=[lib_easy_state, lib_chords])
            lib_orig_btn.click(_set_original, inputs=[lib_sheet, lib_transpose], outputs=[lib_easy_state, lib_chords])

        gr.Markdown("---\nResults are also saved to the **output/** folder. "
                    "Everything runs locally; the only step that needs internet is "
                    "downloading a link or a model you haven't used before.")

        demo.load(js=INIT_JS)  # install all custom styles + karaoke behaviour once per page
    return demo


def main() -> None:
    demo = build_ui()
    demo.queue()  # required for live streaming + progress bars
    # NOTE: in Gradio 6 both `theme` and `css` live on launch(), NOT on Blocks().
    # Passing our stylesheet here is the reliable fix for the old "black on black"
    # karaoke (the previous JS-injected CSS didn't always run).
    # allowed_paths lets the browser stream the karaoke audio files we write to
    # output/ (see _gradio_file_url); without it Gradio blocks serving them.
    demo.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS, inbrowser=True, show_error=True,
                allowed_paths=[str(OUTPUT_DIR)])


if __name__ == "__main__":
    main()
