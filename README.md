# Whisper Transcriber — speech & lyrics to text

A strong, reliable audio-to-text system built on the best free models:

- **Speech → text** with [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
  (`large-v3`), the state-of-the-art free speech model.
- **Music → lyrics** with [Demucs](https://github.com/facebookresearch/demucs)
  (removes the instruments, keeps the vocals) + Whisper.

---

## ⚠️ Read this first: what's realistic on your hardware

Your laptop has an **Intel integrated GPU (no NVIDIA/CUDA)**, an i5 low-power CPU,
and 16 GB RAM. That means:

- ✅ You can **run** the best models and get excellent transcriptions. (Just not
  instantly — see speed notes below.)
- ❌ You **cannot train / fine-tune** models here. Real training needs an NVIDIA
  GPU. The good news: you don't need to. Whisper `large-v3` is already top-tier.
- 🚀 If you later want a model specialized for *your* voice/accent/jargon, do it
  on a **free cloud GPU** — see [`training/README.md`](training/README.md). The
  training script is ready.

**Bottom line:** "improving Whisper" for you = using the biggest model + smart
audio handling now, and (optionally) domain fine-tuning in the cloud later.

---

## Quick start

Open a terminal in `C:\whisper-transcriber`.

### Transcribe speech
```bat
transcribe input\jfk.flac
```
or any file:
```bat
transcribe "C:\Users\rongo\Desktop\meeting.mp3"
```
Faster (recommended on this laptop), forcing English:
```bat
transcribe input\talk.m4a --model large-v3-turbo --language en
```
Whole folder at once:
```bat
transcribe input\
```

### Get lyrics from a song
```bat
lyrics "input\song.mp3"
```

Results are written to the **`output\`** folder as `.txt`, `.srt`, `.vtt`,
and `.json`.

---

## Choosing a model (accuracy vs speed)

| `--model` | Accuracy | Speed on your CPU | When to use |
|-----------|----------|-------------------|-------------|
| `large-v3-turbo` | ★★★★☆ | ~4–8× faster | **default — best everyday choice here** |
| `large-v3` | ★★★★★ | slow | when accuracy matters most (`--model large-v3`) |
| `medium` | ★★★☆ | faster | quick drafts |
| `small` / `base` / `tiny` | ★★ / ★ | fastest | testing only |

Speed rule of thumb on this laptop: `large-v3` is roughly **as slow as, or slower
than, real time** (a 10-min audio may take 10–30+ min). `large-v3-turbo` is much
faster. Be patient or use turbo.

---

## Useful options

```
--language en        force a language (skip auto-detect; faster, more reliable)
--task translate     output ENGLISH text from any spoken language
--model PATH         use a local fine-tuned model folder (see training/)
--prompt "..."       hint names/jargon/spellings to boost accuracy
--formats txt,srt    pick output formats
--threads 4          CPU threads (try 4 if it feels slow)
--no-word-timestamps slightly faster
```
Full help: `transcribe --help` / `lyrics --help`.

---

## How it works

```
SPEECH:  audio ── faster-whisper (large-v3) ──► text (.txt/.srt/.vtt/.json)
                  · voice-activity detection skips silence
                  · word-level timestamps
                  · temperature fallback for hard audio

MUSIC:   song ── Demucs (vocals/no_vocals) ──► vocals.wav ── Whisper ──► lyrics
```

---

## Troubleshooting

- **`SSL: CERTIFICATE_VERIFY_FAILED` on downloads** — your AVG antivirus inspects
  HTTPS. We fixed it by installing `pip-system-certs` (uses the Windows cert
  store). If it returns, reinstall it:
  `venv\Scripts\python.exe -m pip install pip-system-certs`.
- **It's very slow** — expected on this CPU. Use `--model large-v3-turbo`, set
  `--language`, and try `--threads 4`.
- **A rare format won't open** — common formats (mp3, wav, flac, m4a, ogg, most
  video) work out of the box (no ffmpeg needed). For an unusual container,
  install ffmpeg: `winget install Gyan.FFmpeg`, then reopen the terminal.
- **Lyrics: separation is slow** — normal on CPU. To preview a song quickly,
  process only the first part: `lyrics "input\song.mp3" --max-seconds 30`.
- **Lyrics quality** — singing is harder than speech, so expect good-not-perfect
  results. For better separation try `--demucs-model htdemucs_ft` (slower).

---

## Project layout

```
whisper-transcriber/
├─ setup.bat / setup.sh             one-click install (Windows / Mac+Linux)
├─ download_models.py               fetches all models
├─ transcribe.py  transcribe.bat  transcribe.sh    speech -> text
├─ lyrics.py      lyrics.bat      lyrics.sh        music  -> lyrics
├─ whisper_core.py                  shared engine + output writers
├─ requirements.txt                 inference dependencies
├─ input/                           put audio here
├─ output/                          results appear here
├─ models/                          downloaded model weights (auto, gitignored)
├─ separated/                       Demucs vocal stems (auto, gitignored)
└─ training/                        fine-tune on a cloud GPU (later)
   ├─ finetune_whisper.py
   ├─ requirements-training.txt
   └─ README.md
```

---

## Installing on a new computer (one click)

The repo only contains code — the big model files and the Python environment are
re-created automatically. After cloning (or unzipping) the project:

**Windows:** double-click **`setup.bat`**
**macOS / Linux:** open a terminal in the folder and run **`bash setup.sh`**
*(or rename it to `setup.command` and double-click)*

That single step:
1. finds Python 3.10–3.12,
2. creates the `venv`,
3. installs all dependencies,
4. downloads every model (~1.8 GB, from Hugging Face — fast).

Need Python first? Get 3.11 from <https://www.python.org/downloads/> (on Windows,
tick **"Add Python to PATH"**). To pre-download models without the big turbo
model, run `python download_models.py --skip-turbo`.

---

## Sharing this project (GitHub or Google Drive)

It's a git repo with a `.gitignore` that excludes `venv/`, `models/`, `separated/`,
and your audio — so what you share stays small (just code).

- **GitHub:** create an empty repo, then:
  ```bat
  git remote add origin https://github.com/<you>/<repo>.git
  git push -u origin main
  ```
- **Google Drive:** zip the folder **without** `venv/` and `models/` (they're huge
  and get re-downloaded). The recipient just runs `setup.bat` / `setup.sh`.

Either way, the other person runs the setup file and everything downloads itself.

---

## Reinstalling from scratch (manual)

```bat
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe download_models.py
```
