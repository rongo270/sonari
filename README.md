# Sonari 🎙️🎵

**Turn any audio into text — speech *or* music.** Sonari transcribes talking
(meetings, interviews, voice notes, video) and extracts **lyrics from songs** by
removing the instruments — all using the best free open-source AI models, running
**fully offline** on your own computer. No GPU required, and after setup it needs
no internet.

---

## ✨ Features

- 🎙️ **Speech → text** — powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with OpenAI's `large-v3-turbo`
- 🎵 **Music → lyrics** — [Demucs](https://github.com/facebookresearch/demucs) isolates the vocals, then Whisper transcribes them
- 📝 **Multiple outputs** — plain text, **SRT/VTT subtitles** (with timing), and JSON (word-level timestamps + confidence)
- 🌍 **99+ languages** — auto-detect, or translate any language into English
- 💻 **Offline & private** — runs locally on CPU; your audio never leaves your machine
- ⚡ **One-click setup** — installer scripts for Windows and macOS/Linux fetch everything for you
- 🖥️ **Visual app** — a simple browser UI (`app.py`) with model/option pickers; the command line still works too
- ▶️ **From a link** — paste a YouTube URL and Sonari fetches the audio, then transcribes it
- 🎤 **Karaoke & chords** — sing along to the **instrumental** (vocals removed) with the words lighting up in time, plus an auto-detected **guitar-chord sheet** you can transpose up/down or switch to **easy** chords with a suggested capo — all offline
- 📖 **Built-in guide** — a friendly in-app **📖 Guide** tab explains every feature, step by step
- 🖥️ **One-click icon** — setup drops a **Sonari** shortcut (with a custom app icon) on your Desktop

---

## 📦 Requirements

- **Python 3.10–3.12** (3.11 recommended) — [get it here](https://www.python.org/downloads/)
- ~3 GB free disk (for models) and ~4 GB RAM
- Windows, macOS, or Linux
- *(Optional)* an NVIDIA GPU makes it many times faster — but is **not** required

---

## 🚀 Quick start

### 1. Get the code
```bash
git clone https://github.com/<your-username>/sonari.git
cd sonari
```
*(or download the ZIP via the green **Code** button and extract it)*

### 2. Install everything — one step
| Your OS | Do this |
|---------|---------|
| **Windows** | double-click **`setup.bat`** |
| **macOS / Linux** | run **`bash setup.sh`** |

It creates a virtual environment, installs dependencies, and downloads all AI
models (~1.8 GB, one time).

### 3. Use it
**Speech → text:**
```bash
tools\transcribe input\jfk.flac        # Windows
./tools/transcribe.sh input/jfk.flac   # macOS / Linux
```
**Song → lyrics:**
```bash
tools\lyrics "input\song.mp3"          # Windows
./tools/lyrics.sh "input/song.mp3"     # macOS / Linux
```

**Prefer buttons? Launch the visual app:**
```bash
app             # Windows — or just double-click the Sonari icon (or app.bat)
./app.sh        # macOS / Linux
```
It opens in your browser with tabs for **speech**, **lyrics**, **YouTube links**, your
**library**, and a **📖 Guide** that walks you through everything.

**Straight from a link** (works in the app *or* the terminal):
```bash
tools\lyrics --url "https://www.youtube.com/watch?v=..."        # song → lyrics
tools\transcribe --url "https://www.youtube.com/watch?v=..."    # talk → text
```

Your results appear in the **`output/`** folder.

---

## 🎛️ Common options

```
--model large-v3-turbo   default — near-best quality, good speed
--model large-v3         maximum accuracy (slower)
--model base | small     faster, lower accuracy (quick drafts)
--language en            force a language (skips auto-detect)
--task translate         output English from any spoken language
--max-seconds 30         (lyrics) process only the first 30s — handy for a quick test
--demucs-model htdemucs_ft   (lyrics) slower but cleaner vocal separation
```
Full help: `tools\transcribe --help` / `tools\lyrics --help`.

---

## ⚙️ How it works

```
SPEECH:  audio ─► faster-whisper (large-v3-turbo) ─► text + subtitles
                  · voice-activity detection   · word-level timestamps

MUSIC:   song ─► Demucs (isolate vocals) ─► vocals ─► Whisper ─► lyrics
```

## 🐢 Performance

Sonari runs on CPU, so speed depends on your machine — expect roughly real-time
for speech and a few minutes to separate a song on a typical laptop. If the
computer has an **NVIDIA GPU**, Sonari uses it automatically and is far faster.

## 🧠 Make it smarter (fine-tuning)

Whisper is already excellent, but you can fine-tune it for **your** voice, accent,
language, or vocabulary. See [`training/`](training/) for a ready-to-run LoRA
script and a step-by-step guide using free cloud GPUs (Google Colab / Kaggle).

## 📁 What's in the folder

```
Sonari/
├─ app.py                     the visual app  (run via the Sonari icon · app.bat · app.sh)
├─ whisper_core.py            the speech-to-text engine
├─ lyrics.py chords.py media.py   vocal separation · chord detection · link downloads
├─ setup.bat / setup.sh       one-click installer
├─ assets/                    the app icon (sonari.ico / sonari.png)
├─ tools/                     command-line extras — transcribe, lyrics,
│                               download_models, make_icon, make_shortcut
├─ input/  output/            your audio in · your text + saved library out
└─ training/                  optional fine-tuning scripts
```

## 🛠️ Troubleshooting

- **`SSL: CERTIFICATE_VERIFY_FAILED`** during install — antivirus is inspecting
  HTTPS; setup installs `pip-system-certs` to fix it. Re-run setup if it persists.
- **It's slow** — expected on CPU. Use `--model base` for drafts, or run on a GPU.
- **A rare audio format won't open** — install ffmpeg (`winget install Gyan.FFmpeg`
  on Windows, `brew install ffmpeg` on macOS). Common formats work without it.

## 🙏 Built with

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) ·
[OpenAI Whisper](https://github.com/openai/whisper) ·
[Demucs](https://github.com/facebookresearch/demucs)

## 📄 License

Add a license of your choice (e.g. **MIT**). Note the underlying models have their
own licenses (Whisper: MIT, Demucs: MIT) — review them before any commercial use.
