#!/usr/bin/env python
"""
app.py  -  Sonari visual app (web UI)
=====================================

A friendly browser interface for everything the command-line tools do, plus an
"online" tab that takes a YouTube (or other) link. It opens in your browser and
runs 100% on your own computer.

Three tabs:
  1. Speech -> text     (upload a file or record from your mic)
  2. Song  -> lyrics    (Demucs separates the vocals, then Whisper reads them)
  3. From a link        (paste a YouTube URL -> audio -> lyrics or transcript)

Highlights:
  - Live progress on screen (including the slow vocal-separation step).
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


def _norm_lang(language):
    """Turn the dropdown value into what the engine expects (None = auto-detect)."""
    if not language or str(language).lower() in ("auto", "auto-detect"):
        return None
    return str(language).strip()


def _row(seg):
    return [core._ts(seg.start), core._ts(seg.end), seg.text.strip()]


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
        '<span class="kara-title">🎤 Press play — words light up in time · click a line to jump</span>'
        '<button class="kara-fs" type="button">⛶ Fullscreen</button>'
        '</div>'
        '<audio class="kara-audio" controls preload="metadata" src="%s"></audio>'
        '<div class="kara-lyrics">%s</div>'
        '</div>'
    ) % (uid, uri, body)


# Installed once on page load. Adds the styles and wires up every karaoke widget
# (now and any that appear later) — Gradio's HTML component doesn't run <script>
# tags, so we attach behaviour from here via a MutationObserver.
KARA_JS = r"""
() => {
  if (window.__karaInit) return;
  window.__karaInit = true;
  const css = `
.kara-root{border:1px solid #2a2a3a;border-radius:12px;padding:14px;background:#0c0c12;color:#ececf5;font-family:system-ui,-apple-system,'Segoe UI',sans-serif}
.kara-bar{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px}
.kara-title{font-size:13px;opacity:.75}
.kara-fs{cursor:pointer;border:0;border-radius:8px;padding:6px 12px;background:#5b5bd6;color:#fff;font-size:13px;font-weight:600}
.kara-fs:hover{background:#6d6df0}
.kara-audio{width:100%;margin-bottom:12px}
.kara-lyrics{max-height:340px;overflow:auto;line-height:2;font-size:18px;scroll-behavior:smooth;padding:4px}
.kara-line{padding:5px 8px;border-radius:8px;opacity:.5;transition:opacity .15s,background .15s;cursor:pointer}
.kara-line:hover{opacity:.85}
.kara-line.on{opacity:1;background:rgba(91,91,214,.22)}
.kara-w.on{background:#ffd64a;color:#10101a;border-radius:5px}
.kara-root:fullscreen{padding:6vh 10vw;display:flex;flex-direction:column;justify-content:center;background:#08080d}
.kara-root:fullscreen .kara-lyrics{max-height:none;flex:1;font-size:32px;text-align:center;line-height:2.2}
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
          if (t >= s && t < e) { l.classList.add('on'); cur = l; } else { l.classList.remove('on'); }
        }
        for (const w of words) {
          const s = +w.dataset.s, e = +w.dataset.e;
          if (t >= s && t < e) { w.classList.add('on'); } else { w.classList.remove('on'); }
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
# Core processing (generators -> live updates in the UI)
# --------------------------------------------------------------------------- #
def _process_speech(audio_path, model_name, language, task, beam_size,
                    vad, word_ts, prompt, threads, progress):
    """Yields (status, text, rows, files, karaoke_html) as the transcript streams."""
    progress(0.0, desc=f"Loading model '{model_name}'…")
    model, device, _ = _get_model(model_name, threads)

    status, text, rows, segments, info = "Starting…", "", [], [], None
    t0 = time.time()
    progress(0.05, desc="Listening…")
    for kind, payload in core.transcribe_stream(
        model, audio_path,
        language=_norm_lang(language), task=task, beam_size=int(beam_size),
        vad=vad, word_timestamps=word_ts, initial_prompt=(prompt or None),
    ):
        if kind == "info":
            info = payload
            status = (f"🌍 {info.language} ({info.language_probability:.0%}) · "
                      f"{info.duration:.0f}s audio · running on {device}")
            yield status, text, rows, None, ""
        else:
            seg = payload
            segments.append(seg)
            text += seg.text
            rows = rows + [_row(seg)]
            if info and info.duration:
                progress(min(seg.end / info.duration, 0.99), desc="Transcribing…")
            yield status, text.strip(), rows, None, ""

    result = core._build_result(info, segments, time.time() - t0)
    written = core.write_all(result, OUTPUT_DIR, Path(audio_path).stem,
                             ("txt", "srt", "vtt", "json"))
    progress(1.0, desc="Building player…")
    kara = build_karaoke_html(result, audio_path)
    yield (f"✅ Done · {info.language} · {len(segments)} segments · {result.elapsed:.0f}s",
           result.text, rows, [str(p) for p in written], kara)


def _process_lyrics(audio_path, whisper_model, demucs_model, language,
                    max_seconds, vad, threads, progress):
    """Yields (status, text, rows, files, vocals_path, karaoke_html) for music."""
    # Estimate the separation time so the progress bar can move while it runs.
    try:
        import soundfile as sf
        total = sf.info(str(audio_path)).duration
    except Exception:
        total = 0.0
    if max_seconds and float(max_seconds) > 0 and total:
        total = min(total, float(max_seconds))
    est = max(15.0, (total or 30.0) * 4.0)

    # Demucs is one long blocking call with no yield points, so run it in a
    # thread and stream an elapsed-time counter to the screen meanwhile.
    holder: dict = {}

    def _work():
        try:
            holder["v"] = separate_vocals(Path(audio_path), demucs_model, float(max_seconds or 0))
        except Exception as exc:
            holder["e"] = exc

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    t0 = time.time()
    while th.is_alive():
        el = time.time() - t0
        progress(0.02 + 0.5 * min(el / est, 0.95), desc=f"Separating vocals… {int(el)}s")
        yield (f"🎛️ Separating vocals from the music… **{int(el)}s** elapsed "
               f"(this is the slow part on CPU — please wait)", "", [], None, None, "")
        th.join(timeout=0.8)
    if "e" in holder:
        raise gr.Error(f"Vocal separation failed: {holder['e']}")
    vocals = holder["v"]

    progress(0.55, desc=f"Loading model '{whisper_model}'…")
    yield "🎙️ Vocals isolated · loading model…", "", [], None, str(vocals), ""
    model, _, _ = _get_model(whisper_model, threads)

    status, text, rows, segments, info = "Reading lyrics…", "", [], [], None
    t1 = time.time()
    for kind, payload in core.transcribe_stream(
        model, vocals,
        language=_norm_lang(language), beam_size=5, vad=vad, word_timestamps=True,
        # Singing confuses "use previous text as context" -> repeats. Off for music.
        condition_on_previous_text=False,
    ):
        if kind == "info":
            info = payload
            status = f"🎤 {info.language} ({info.language_probability:.0%}) · vocals {info.duration:.0f}s"
            yield status, text, rows, None, str(vocals), ""
        else:
            seg = payload
            segments.append(seg)
            text += seg.text
            rows = rows + [_row(seg)]
            if info and info.duration:
                progress(0.55 + 0.4 * min(seg.end / info.duration, 1.0), desc="Transcribing lyrics…")
            yield status, text.strip(), rows, None, str(vocals), ""

    result = core._build_result(info, segments, time.time() - t1)
    written = core.write_all(result, OUTPUT_DIR, f"{Path(audio_path).stem}.lyrics",
                             ("txt", "srt", "json"))
    progress(0.98, desc="Building karaoke player…")
    kara = build_karaoke_html(result, vocals)
    progress(1.0, desc="Done")
    yield (f"✅ Lyrics done · {len(segments)} lines · {result.elapsed:.0f}s (+ separation)",
           result.text, rows, [str(p) for p in written], str(vocals), kara)


# --------------------------------------------------------------------------- #
# Button handlers (validate, then delegate to the generators above)
# --------------------------------------------------------------------------- #
def ui_speech(audio, model_name, language, task, beam_size, vad, word_ts,
              prompt, threads, progress=gr.Progress()):
    if not audio:
        raise gr.Error("Please upload a file or record from your microphone first.")
    yield from _process_speech(audio, model_name, language, task, beam_size,
                               vad, word_ts, prompt, threads, progress)


def ui_lyrics(audio, whisper_model, demucs_model, language, max_seconds,
              vad, threads, progress=gr.Progress()):
    if not audio:
        raise gr.Error("Please upload a song first.")
    yield from _process_lyrics(audio, whisper_model, demucs_model, language,
                               max_seconds, vad, threads, progress)


def ui_youtube(url, mode, whisper_model, demucs_model, language, task,
               max_seconds, vad, threads, progress=gr.Progress()):
    if not (url or "").strip():
        raise gr.Error("Paste a link (e.g. a YouTube URL) first.")
    progress(0.0, desc="Downloading audio from the link…")
    yield "### ⏬ Downloading audio…", "Fetching audio from the link…", "", [], None, None, ""
    try:
        audio, title = download_audio(url.strip(), INPUT_DIR)
    except Exception as exc:  # surface a clean message in the UI
        raise gr.Error(f"Could not download that link: {exc}")

    head = f"### 🎬 {title}"
    if mode.startswith("Song"):
        for status, text, rows, files, vocals, kara in _process_lyrics(
            str(audio), whisper_model, demucs_model, language, max_seconds, vad, threads, progress
        ):
            yield head, status, text, rows, files, vocals, kara
    else:
        for status, text, rows, files, kara in _process_speech(
            str(audio), whisper_model, language, task, 5, vad, True, None, threads, progress
        ):
            yield head, status, text, rows, files, None, kara


# --------------------------------------------------------------------------- #
# UI layout
# --------------------------------------------------------------------------- #
def _model_dropdown(label="Whisper model", value="large-v3-turbo"):
    return gr.Dropdown(WHISPER_MODELS, value=value, label=label,
                       info="turbo = fast + great. large-v3 = max accuracy. "
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
            "on your own computer. Pick a file (or a YouTube link), choose your "
            "options, press the button, and use the **karaoke player** to follow along."
        )

        # ---- Tab 1: Speech -> text ---------------------------------------- #
        with gr.Tab("🎙️ Speech → text"):
            with gr.Row():
                with gr.Column(scale=1):
                    sp_audio = gr.Audio(sources=["upload", "microphone"],
                                        type="filepath", label="Audio / video file or recording")
                    sp_model = _model_dropdown()
                    with gr.Row():
                        sp_lang = _lang_dropdown()
                        sp_task = gr.Radio(["transcribe", "translate"], value="transcribe",
                                           label="Task", info="translate = output English")
                    with gr.Accordion("Advanced options", open=False):
                        sp_beam = gr.Slider(1, 10, value=5, step=1, label="Beam size (higher = more accurate, slower)")
                        sp_vad = gr.Checkbox(value=True, label="Skip silence (voice-activity detection)")
                        sp_words = gr.Checkbox(value=True, label="Word-level timestamps")
                        sp_prompt = gr.Textbox(label="Hint words / spellings (names, jargon)", placeholder="optional")
                        sp_threads = gr.Slider(0, 16, value=0, step=1, label="CPU threads (0 = auto)")
                    sp_btn = gr.Button("Transcribe", variant="primary")
                with gr.Column(scale=1):
                    sp_status = gr.Markdown()
                    sp_text = gr.Textbox(label="Transcript", lines=12)
                    sp_segs = gr.Dataframe(headers=SEGMENT_HEADERS, wrap=True,
                                           label="Segments (with timestamps)")
                    sp_files = gr.File(label="Download (txt / srt / vtt / json)", file_count="multiple")
                    with gr.Accordion("🎤 Karaoke player", open=False):
                        sp_kara = gr.HTML()
            sp_btn.click(
                ui_speech,
                inputs=[sp_audio, sp_model, sp_lang, sp_task, sp_beam, sp_vad, sp_words, sp_prompt, sp_threads],
                outputs=[sp_status, sp_text, sp_segs, sp_files, sp_kara],
            )

        # ---- Tab 2: Song -> lyrics ---------------------------------------- #
        with gr.Tab("🎵 Song → lyrics"):
            with gr.Row():
                with gr.Column(scale=1):
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
                    ly_btn = gr.Button("Extract lyrics", variant="primary")
                    gr.Markdown("_Separating vocals is the slow part on a CPU — a 3–4 min song "
                                "can take several minutes. Use the slider to test the first 30s first._")
                with gr.Column(scale=1):
                    ly_status = gr.Markdown()
                    ly_text = gr.Textbox(label="Lyrics", lines=10)
                    with gr.Accordion("🎤 Karaoke player", open=True):
                        ly_kara = gr.HTML()
                    ly_segs = gr.Dataframe(headers=SEGMENT_HEADERS, wrap=True,
                                           label="Lines (with timestamps)")
                    ly_vocals = gr.Audio(label="Isolated vocals", type="filepath", interactive=False)
                    ly_files = gr.File(label="Download (txt / srt / json)", file_count="multiple")
            ly_btn.click(
                ui_lyrics,
                inputs=[ly_audio, ly_model, ly_demucs, ly_lang, ly_max, ly_vad, ly_threads],
                outputs=[ly_status, ly_text, ly_segs, ly_files, ly_vocals, ly_kara],
            )

        # ---- Tab 3: From a link (YouTube) --------------------------------- #
        with gr.Tab("▶️ From a link (YouTube)"):
            with gr.Row():
                with gr.Column(scale=1):
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
                    yt_btn = gr.Button("Download & process", variant="primary")
                    gr.Markdown("_Downloading needs internet. Only download audio you're allowed to use._")
                with gr.Column(scale=1):
                    yt_title = gr.Markdown()
                    yt_status = gr.Markdown()
                    yt_text = gr.Textbox(label="Result", lines=10)
                    with gr.Accordion("🎤 Karaoke player", open=True):
                        yt_kara = gr.HTML()
                    yt_segs = gr.Dataframe(headers=SEGMENT_HEADERS, wrap=True,
                                           label="Segments (with timestamps)")
                    yt_vocals = gr.Audio(label="Isolated vocals (songs)", type="filepath", interactive=False)
                    yt_files = gr.File(label="Download", file_count="multiple")
            yt_btn.click(
                ui_youtube,
                inputs=[yt_url, yt_mode, yt_model, yt_demucs, yt_lang, yt_task, yt_max, yt_vad, yt_threads],
                outputs=[yt_title, yt_status, yt_text, yt_segs, yt_files, yt_vocals, yt_kara],
            )

        gr.Markdown("---\nResults are also saved to the **output/** folder. "
                    "Everything runs locally; the only step that needs internet is "
                    "downloading a link or a model you haven't used before.")

        demo.load(js=KARA_JS)  # install karaoke styles + behaviour once per page
    return demo


def main() -> None:
    demo = build_ui()
    demo.queue()  # required for live streaming + progress bars
    demo.launch(theme=gr.themes.Soft(), inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
