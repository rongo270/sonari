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
    the audio, click any line to jump, and go fullscreen.
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
import threading
import time
from pathlib import Path

import gradio as gr

import whisper_core as core
from lyrics import separate_vocals
from media import download_audio

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

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
def _audio_ogg_data_uri(path):
    """Encode an audio file to a small OGG/Vorbis data URI for the <audio> tag.

    Downmixes to mono and writes in chunks — writing a big buffer in one go
    crashes libsndfile's Vorbis encoder on Windows. Returns None on failure
    (e.g. a video/m4a that the audio library can't read), so the player is just
    skipped rather than breaking the run.
    """
    try:
        import base64
        import io

        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = np.clip(data.mean(axis=1), -1.0, 1.0)
        buf = io.BytesIO()
        with sf.SoundFile(buf, mode="w", samplerate=int(sr), channels=1,
                          format="OGG", subtype="VORBIS") as f:
            step = 32768
            for i in range(0, len(mono), step):
                f.write(mono[i:i + step])
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:audio/ogg;base64,{b64}"
    except Exception as exc:
        print(f"[karaoke] could not encode audio ({exc}); skipping the player.")
        return None


def build_karaoke_html(result, audio_path) -> str:
    """Build the karaoke widget: an <audio> player + timestamped, clickable lyrics.

    Word-level <span>s carry their own start/end so the JS can light up each word
    as it's sung. Returns "" if the audio couldn't be embedded.
    """
    uri = _audio_ogg_data_uri(audio_path)
    if not uri:
        return ""

    uid = str(int(time.time() * 1000))
    lines = []
    for seg in result.segments:
        s = max(seg.start or 0.0, 0.0)
        e = max(seg.end or s, s)
        words = getattr(seg, "words", None)
        if words:
            inner = "".join(
                '<span class="kara-w" data-s="%.2f" data-e="%.2f">%s</span>' % (
                    max(w.start if w.start is not None else s, 0.0),
                    max(w.end if w.end is not None else (w.start or s), 0.0),
                    html.escape(w.word),
                )
                for w in words
            )
        else:
            inner = html.escape(seg.text.strip())
        lines.append('<div class="kara-line" data-s="%.2f" data-e="%.2f">%s</div>' % (s, e, inner))

    body = "\n".join(lines) or '<div class="kara-line">(no lyrics detected)</div>'
    return (
        '<div class="kara-root" id="kara-%s" data-wired="0">'
        '<div class="kara-bar">'
        '<span class="kara-title">🎤 Press play — the current word lights up · click any line to jump</span>'
        '<button class="kara-fs" type="button">⛶ Fullscreen</button>'
        '</div>'
        '<audio class="kara-audio" controls preload="metadata" src="%s"></audio>'
        '<div class="kara-lyrics">%s</div>'
        '</div>'
    ) % (uid, uri, body)


# Shown in the results area if the audio couldn't be embedded for the player.
_KARA_EMPTY = (
    '<div class="kara-empty">🎤 The karaoke player isn’t available for this file '
    '(its audio couldn’t be embedded), but your transcript and downloads are right '
    'below.</div>'
)


# Installed once on page load. Adds ALL our custom styles (the progress bar and
# the karaoke player) and wires up every karaoke widget — now and any that appear
# later — because Gradio's HTML component doesn't run <script> tags, so we attach
# behaviour from here via a MutationObserver.
INIT_JS = r"""
() => {
  if (window.__sonariInit) return;
  window.__sonariInit = true;

  const css = `
/* ---- center the page a little ---------------------------------------- */
.gradio-container{max-width:1040px!important;margin:0 auto!important}

/* ---- the single progress bar ----------------------------------------- */
.pbar-card{border:1px solid rgba(130,130,170,.28);border-radius:16px;padding:22px 24px;background:rgba(130,130,170,.06)}
.pbar-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:14px}
.pbar-label{font-size:17px;font-weight:650}
.pbar-pct{font-size:14px;font-weight:700;opacity:.7;font-variant-numeric:tabular-nums}
.pbar-track{height:16px;border-radius:999px;background:rgba(130,130,170,.22);overflow:hidden;position:relative}
/* gentle moving stripes on the empty track, so even at 0% (e.g. while a model
   loads) it still looks busy rather than frozen */
.pbar-track::before{content:'';position:absolute;inset:0;background:repeating-linear-gradient(45deg,transparent 0 10px,rgba(130,130,170,.16) 10px 20px);background-size:28px 28px;animation:pbar-stripe .9s linear infinite}
@keyframes pbar-stripe{to{background-position:28px 0}}
.pbar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#6d6df0,#9b6df0);width:0;transition:width .35s ease;position:relative;z-index:1;overflow:hidden}
/* a soft sheen sweeping across the filled part */
.pbar-fill::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.45),transparent);transform:translateX(-100%);animation:pbar-sheen 1.15s linear infinite}
@keyframes pbar-sheen{to{transform:translateX(100%)}}
.pbar-sub{margin-top:13px;font-size:13px;opacity:.7;min-height:1.1em}

/* ---- karaoke player -------------------------------------------------- */
.kara-root{border:1px solid #34344c;border-radius:16px;padding:16px 18px;background:linear-gradient(180deg,#15151f,#0e0e16);color:#edecf6;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;box-shadow:0 10px 30px rgba(10,10,30,.25)}
.kara-bar{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:12px}
.kara-title{font-size:13px;color:#a7a7c6}
.kara-fs{cursor:pointer;border:0;border-radius:9px;padding:7px 13px;background:#5b5bd6;color:#fff;font-size:13px;font-weight:600}
.kara-fs:hover{background:#6e6ef2}
.kara-audio{width:100%;margin-bottom:14px}
.kara-lyrics{max-height:380px;overflow:auto;line-height:2.15;font-size:20px;scroll-behavior:smooth;padding:6px 4px}
/* three states for a line: upcoming (default) · now singing (.on) · already sung (.sung) */
.kara-line{padding:6px 12px;margin:2px 0;border-radius:11px;color:#b9b9d0;transition:color .2s,background .2s;cursor:pointer}
.kara-line:hover{background:rgba(124,124,240,.12);color:#e2e1f2}
.kara-line.on{background:rgba(124,124,240,.18);color:#ffffff}
.kara-line.sung{color:#7a7a93}
/* three states for a single word */
.kara-w{border-radius:6px;padding:0 1px;transition:color .12s,background .12s}
.kara-w.sung{color:#6f6f8c}
.kara-w.on{background:linear-gradient(180deg,#ffdf72,#ffc83d);color:#1a1a26;font-weight:700;padding:1px 4px;box-shadow:0 1px 2px rgba(0,0,0,.25)}
/* in the current line, words not yet reached stay bright white so you can read ahead */
.kara-line.on .kara-w:not(.on):not(.sung){color:#ffffff}
.kara-empty{padding:18px;border:1px dashed rgba(130,130,170,.4);border-radius:14px;opacity:.85;font-size:14px}
.kara-root:fullscreen{padding:7vh 10vw;display:flex;flex-direction:column;justify-content:center;background:#08080f}
.kara-root:fullscreen .kara-lyrics{max-height:none;flex:1;font-size:34px;text-align:center;line-height:2.3}
`;
  const st = document.createElement('style');
  st.textContent = css;
  document.head.appendChild(st);

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
                    vad, word_ts, prompt, threads):
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
    kara = build_karaoke_html(result, audio_path)
    yield {"done": True,
           "text": result.text,
           "rows": [_row(s) for s in segments],
           "files": [str(p) for p in written],
           "kara": kara,
           "summary": f"### ✅ Done\n{info.language} · {len(segments)} segments · {result.elapsed:.0f}s"}


def _process_lyrics(audio_path, whisper_model, demucs_model, language,
                    max_seconds, vad, threads):
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
            holder["v"] = separate_vocals(Path(audio_path), demucs_model, float(max_seconds or 0))
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
    vocals = holder["v"]

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
    kara = build_karaoke_html(result, vocals)
    yield {"done": True,
           "text": result.text,
           "rows": [_row(s) for s in segments],
           "files": [str(p) for p in written],
           "kara": kara,
           "vocals": str(vocals),
           "summary": f"### ✅ Lyrics ready\n{len(segments)} lines · {result.elapsed:.0f}s (＋ separation time)"}


# --------------------------------------------------------------------------- #
# UI plumbing: drive an engine generator and route its events to components
# --------------------------------------------------------------------------- #
def _pump(gen, *, progress, results, pbar, summary, text, segs, files, kara, vocals=None):
    """Run `gen` and send each event to the right Gradio component.

    While it works, ONLY the progress bar (`pbar`) changes — that's why nothing
    flickers any more. When the engine signals "done", we hide the progress
    screen, reveal the results screen, and fill everything in at once.
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
            yield out
        else:
            yield {pbar: progress_html(ev["percent"], ev["label"], ev.get("sub", ""))}


def _go_setup():
    """Send the user back to the setup screen (used by the 'start over' buttons)."""
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


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
                                 kara=ly_kara, vocals=ly_vocals)

            ly_btn.click(
                ui_lyrics,
                inputs=[ly_audio, ly_model, ly_demucs, ly_lang, ly_max, ly_vad, ly_threads],
                outputs=[ly_setup, ly_progress, ly_results, ly_pbar, ly_summary, ly_text,
                         ly_segs, ly_files, ly_kara, ly_vocals],
                show_progress="hidden",
            )
            ly_back.click(_go_setup, outputs=[ly_setup, ly_progress, ly_results])

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
                       yt_pbar: progress_html(0.0, "Downloading audio from the link…", "Needs internet.")}
                try:
                    audio, title = download_audio(url.strip(), INPUT_DIR)
                except Exception as exc:  # surface a clean message in the UI
                    raise gr.Error(f"Could not download that link: {exc}")

                yield {yt_title: f"### 🎬 {title}",
                       yt_pbar: progress_html(0.03, "Got the audio — starting…", title)}

                if mode.startswith("Song"):
                    gen = _process_lyrics(str(audio), model_name, demucs_model, language,
                                          max_seconds, vad, threads)
                    yield from _pump(gen, progress=yt_progress, results=yt_results, pbar=yt_pbar,
                                     summary=yt_summary, text=yt_text, segs=yt_segs, files=yt_files,
                                     kara=yt_kara, vocals=yt_vocals)
                else:
                    gen = _process_speech(str(audio), model_name, language, task, 5, vad, True, None, threads)
                    yield from _pump(gen, progress=yt_progress, results=yt_results, pbar=yt_pbar,
                                     summary=yt_summary, text=yt_text, segs=yt_segs, files=yt_files, kara=yt_kara)

            yt_btn.click(
                ui_youtube,
                inputs=[yt_url, yt_mode, yt_model, yt_demucs, yt_lang, yt_task, yt_max, yt_vad, yt_threads],
                outputs=[yt_setup, yt_progress, yt_results, yt_pbar, yt_title, yt_summary, yt_text,
                         yt_segs, yt_files, yt_kara, yt_vocals],
                show_progress="hidden",
            )
            yt_back.click(_go_setup, outputs=[yt_setup, yt_progress, yt_results])

        gr.Markdown("---\nResults are also saved to the **output/** folder. "
                    "Everything runs locally; the only step that needs internet is "
                    "downloading a link or a model you haven't used before.")

        demo.load(js=INIT_JS)  # install all custom styles + karaoke behaviour once per page
    return demo


def main() -> None:
    demo = build_ui()
    demo.queue()  # required for live streaming + progress bars
    demo.launch(theme=gr.themes.Soft(), inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
