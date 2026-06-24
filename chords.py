#!/usr/bin/env python
"""
chords.py  -  Hear the chords + a little music-theory toolbox
=============================================================

This module has two halves that work together to turn a song into a printable
"lyrics + guitar chords" sheet — all offline, with no new heavy downloads (it
uses the torch you already have for Demucs/Whisper).

  HALF 1 — LISTEN (the imperfect, "AI ears" part)
      detect_chords(instrumental) -> {"key": "C", "spans": [{start,end,name}, ...]}
      We feed it the INSTRUMENTAL (drums+bass+other, no singing) that Demucs
      already produced, turn it into a "chromagram" (how much of each of the 12
      musical notes is sounding over time), and match every short slice against
      the 24 basic chords (a major and a minor on each note). Then we smooth out
      the jitter and merge it into a tidy list of timed chords.

      Reality check: this is roughly right for simple pop / rock / folk that use
      plain major & minor chords. It will not match a hand-made tab, and it only
      ever guesses major or minor triads (no 7ths / sus / slash chords).

  HALF 2 — THEORY (the exact, dependable part — just arithmetic on note names)
      transpose_chord("Am", +2)        -> "Bm"        (move everything up/down)
      simplify_chord("Cmaj7")          -> "C"         (the "easy" version)
      suggest_capo(["Bb","Eb","F"])    -> capo + the easy open shapes to play
      align_chords_to_lines(...)       -> chords sitting above the right words

Run it directly to sanity-check the detector on made-up audio (instant), or
point it at a real file:

    python chords.py                       # synthetic self-test (C F G Am)
    python chords.py "separated/.../no_vocals.wav"
"""
from __future__ import annotations

import math
import re

# --------------------------------------------------------------------------- #
# Note names. Two spellings of the same 12 notes: one using sharps (#) and one
# using flats (b). We pick whichever reads more naturally for the song's key
# (e.g. a song in F uses Bb, not A#).
# --------------------------------------------------------------------------- #
PITCH_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_FLAT  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# A lookup from any written note name to its number 0..11 (C=0, C#=1, …, B=11).
_NOTE_TO_PC: dict[str, int] = {}
for _i, _n in enumerate(PITCH_SHARP):
    _NOTE_TO_PC[_n] = _i
for _i, _n in enumerate(PITCH_FLAT):
    _NOTE_TO_PC[_n] = _i
# A few rare "enharmonic" spellings, just so we never choke on them.
_NOTE_TO_PC.update({"E#": 5, "B#": 0, "Cb": 11, "Fb": 4})

# Splits a chord name into  root letter + accidental + the rest (the "quality").
# e.g. "F#m7"  -> ("F", "#", "m7") ;  "Bb"  -> ("B", "b", "")
_CHORD_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)(.*)$")

# The classic beginner "campfire" open chords — the shapes a capo lets you reuse.
# (F and B are barre chords, so we deliberately leave them out of "easy".)
EASY_OPEN = {"C", "D", "E", "G", "A", "Em", "Am", "Dm"}


# --------------------------------------------------------------------------- #
# HALF 2 first — the pure music theory. It's small, exact, and the UI leans on
# it for the transpose / easy / capo buttons.
# --------------------------------------------------------------------------- #
def parse_chord(name: str) -> dict | None:
    """Break a chord name into pieces, or return None if it isn't a chord.

    Returns {"root": 0..11, "suffix": str, "bass": 0..11 or None}.
    "C" -> root 0, suffix "" ; "Am7" -> root 9, suffix "m7" ;
    "D/F#" -> root 2, suffix "", bass 6.
    """
    if not name or name == "N":
        return None
    main, _, bass = name.partition("/")            # split off a slash bass, if any
    m = _CHORD_RE.match(main)
    if not m:
        return None
    letter, acc, suffix = m.groups()
    root = _NOTE_TO_PC.get(letter.upper() + acc)
    if root is None:
        return None
    bass_pc = None
    if bass:
        mb = _CHORD_RE.match(bass)
        if mb:
            bass_pc = _NOTE_TO_PC.get(mb.group(1).upper() + mb.group(2))
    return {"root": root, "suffix": suffix, "bass": bass_pc}


def transpose_chord(name: str, semitones: int, prefer_flats: bool = False) -> str:
    """Shift a chord up (+) or down (-) by N semitones. "C" +2 -> "D".

    This is the engine behind the up/down buttons and the capo maths. The
    quality (m, 7, sus…) is kept; only the root (and any slash bass) moves.
    """
    p = parse_chord(name)
    if p is None:
        return name
    names = PITCH_FLAT if prefer_flats else PITCH_SHARP
    out = names[(p["root"] + semitones) % 12] + p["suffix"]
    if p["bass"] is not None:
        out += "/" + names[(p["bass"] + semitones) % 12]
    return out


def simplify_chord(name: str) -> str:
    """Reduce a fancy chord to a basic major or minor triad — the "easy" version.

    Cmaj7 -> C, Am7 -> Am, Dsus4 -> D, G7 -> G, F#m7b5 -> F#m, C/G -> C.
    (Our detector already only outputs plain major/minor, so this mostly matters
    if chords ever come from somewhere richer — but it keeps "Easy" honest.)
    """
    if not name or name == "N":
        return name
    m = _CHORD_RE.match(name.partition("/")[0])
    if not m:
        return name
    letter, acc, suffix = m.groups()
    s = suffix.lower()
    # minor = a leading "m" or "min", but NOT "maj". Everything else -> major.
    if s.startswith("dim") or s.startswith("°"):
        quality = "dim"
    elif (s.startswith("m") and not s.startswith("maj")) or s.startswith("min"):
        quality = "m"
    else:
        quality = ""
    return letter.upper() + acc + quality


def suggest_capo(chord_names, prefer_flats: bool = False):
    """Find a capo position that turns the song into easy open chords.

    Returns (capo_fret, shapes) where `shapes` maps each sounding chord to the
    open shape you actually finger with the capo on. With a capo on fret N you
    finger chords as if everything were N semitones LOWER, so we transpose each
    chord down by N and count how many become easy open shapes; we keep the
    position that makes the most chords easy (preferring the smallest capo).
    """
    uniq = [c for c in dict.fromkeys(chord_names) if c and c != "N"]
    best_count, best_capo, best_shapes = -1, 0, {}
    for n in range(0, 10):                          # capo 0 (none) .. 9
        shapes = {c: transpose_chord(c, -n) for c in uniq}
        easy = sum(1 for v in shapes.values() if v in EASY_OPEN)
        if easy > best_count or (easy == best_count and n < best_capo):
            best_count, best_capo, best_shapes = easy, n, shapes
    return best_capo, best_shapes


def key_long(name: str) -> str:
    """Human label for a key: "C" -> "C major", "Am" -> "A minor"."""
    if not name:
        return ""
    if name.endswith("m"):
        return name[:-1] + " minor"
    return name + " major"


def key_prefers_flats(name: str) -> bool:
    """Should chords in this key be spelled with flats (Bb) instead of sharps?"""
    p = parse_chord(name)
    if p is None:
        return False
    root, minor = p["root"], name.endswith("m")
    flat_major = {5, 10, 3, 8, 1, 6}               # F Bb Eb Ab Db Gb
    flat_minor = {2, 7, 0, 5, 10, 3}               # Dm Gm Cm Fm Bbm Ebm
    return (root in flat_minor) if minor else (root in flat_major)


# --------------------------------------------------------------------------- #
# Place the detected chords above the right words.
# --------------------------------------------------------------------------- #
def _chord_at(t: float, spans) -> str | None:
    """Which chord is sounding at time `t` (or None if a gap/silence)."""
    for s in spans:
        if s["start"] <= t < s["end"]:
            return s["name"]
    return None


def _changes_between(t0: float, t1: float, spans) -> list[str]:
    """The sequence of (de-duplicated) chord changes that start within [t0, t1)."""
    out: list[str] = []
    for s in spans:
        if t0 <= s["start"] < t1:
            if not out or out[-1] != s["name"]:
                out.append(s["name"])
    return out


def align_chords_to_lines(seg_dicts, spans) -> list[dict]:
    """Turn (lyric lines + timed chords) into rows ready to render as a sheet.

    Each returned row is one of:
      {"type": "lyric",  "cells": [[chord_or_None, word_text], ...]}
      {"type": "chords", "label": "Intro"/"Outro"/…, "chords": [name, ...]}

    A chord is shown above the word being sung when it first changes (and always
    on the first word of a line, so each line is self-contained to play). Purely
    instrumental stretches (intro, solos between verses, outro) become their own
    chord-only rows.
    """
    spans = sorted(
        [s for s in spans if s.get("name") and s["name"] != "N"],
        key=lambda s: s["start"],
    )
    lines: list[dict] = []
    if not seg_dicts:
        names = _changes_between(-1e9, 1e9, spans)
        if names:
            lines.append({"type": "chords", "label": "", "chords": names})
        return lines

    first_start = seg_dicts[0].get("start") or 0.0
    intro = _changes_between(-1e9, first_start, spans)
    if intro:
        lines.append({"type": "chords", "label": "Intro", "chords": intro})

    prev_end = first_start
    last_shown: str | None = None
    for seg in seg_dicts:
        seg_start = seg.get("start") or prev_end
        seg_end = seg.get("end") or seg_start
        # A long instrumental gap before this line (e.g. a solo) -> its own row.
        if seg_start - prev_end > 3.0:
            inter = _changes_between(prev_end, seg_start, spans)
            if inter:
                lines.append({"type": "chords", "label": "♪", "chords": inter})
                last_shown = inter[-1]

        words = seg.get("words") or []
        cells: list[list] = []
        if words:
            for wi, w in enumerate(words):
                ws = w.get("start")
                ws = ws if ws is not None else seg_start
                ch = _chord_at(ws, spans)
                # First word: always show the chord you start the line on.
                show = ch if (wi == 0 or (ch is not None and ch != last_shown)) else None
                if show is not None:
                    last_shown = show
                cells.append([show, w.get("word", "")])
        else:
            ch = _chord_at(seg_start, spans)
            if ch is not None:
                last_shown = ch
            cells.append([ch, seg.get("text", "")])
        lines.append({"type": "lyric", "cells": cells})
        prev_end = seg_end

    outro = _changes_between(prev_end, 1e9, spans)
    if outro:
        lines.append({"type": "chords", "label": "Outro", "chords": outro})
    return lines


def unique_chords(lines) -> list[str]:
    """Every distinct chord used across an aligned sheet (in first-seen order)."""
    seen: dict[str, None] = {}
    for ln in lines:
        if ln["type"] == "chords":
            for c in ln["chords"]:
                seen.setdefault(c, None)
        else:
            for chord, _word in ln["cells"]:
                if chord:
                    seen.setdefault(chord, None)
    return list(seen.keys())


# --------------------------------------------------------------------------- #
# HALF 1 — the detector. Everything below "listens" to the audio.
# --------------------------------------------------------------------------- #
# We work at a modest sample rate (22.05 kHz is plenty for harmony) and use a
# big FFT window so low notes are well resolved.
_SR = 22050
_N_FFT = 8192
_HOP = 2048
_FMIN = 55.0      # ~A1 — ignore sub-bass rumble below this
_FMAX = 2200.0    # ~C#7 — ignore hiss/cymbals above this


def _load_mono(path):
    """Load any audio file to a (mono float32 torch tensor, sample_rate)."""
    import numpy as np
    import torch

    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = torch.from_numpy(data.mean(axis=1).copy())
    except Exception:
        from faster_whisper.audio import decode_audio

        sr = 16000
        y = torch.from_numpy(np.asarray(decode_audio(str(path), sampling_rate=sr),
                                        dtype="float32"))
    return y, int(sr)


def _pcp_matrix(sr: int, n_fft: int):
    """A (12 x freq_bins) matrix that folds an FFT into the 12 pitch classes.

    Each usable frequency bin is mapped to the nearest musical note, and every
    note is collapsed onto one of the 12 names (so all the C's add together, all
    the C#'s, …). Multiplying this by a magnitude spectrum gives a chromagram.
    """
    import torch

    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr)
    M = torch.zeros(12, freqs.shape[0])
    for k, f in enumerate(freqs.tolist()):
        if f < _FMIN or f > _FMAX:
            continue
        midi = 69.0 + 12.0 * math.log2(f / 440.0)
        M[int(round(midi)) % 12, k] = 1.0
    return M


def _chromagram(y, sr):
    """How strong each of the 12 notes is over time -> tensor (12, frames)."""
    import torch

    win = torch.hann_window(_N_FFT)
    spec = torch.stft(y, n_fft=_N_FFT, hop_length=_HOP, window=win,
                      center=True, return_complex=True)
    mag = torch.log1p(spec.abs())                  # log tames very loud bins
    return _pcp_matrix(sr, _N_FFT) @ mag           # (12, frames)


def _templates():
    """The 24 chord "fingerprints": a major and a minor triad on each note."""
    import torch

    rows, labels = [], []
    for intervals, quality in (([0, 4, 7], "maj"), ([0, 3, 7], "min")):
        for root in range(12):
            v = [0.0] * 12
            for iv in intervals:
                v[(root + iv) % 12] = 1.0
            rows.append(v)
            labels.append((root, quality))
    T = torch.tensor(rows)
    return T / T.norm(dim=1, keepdim=True), labels


def _name(idx: int, labels, prefer_flats: bool) -> str:
    """Template index -> chord name (or "N" for the no-chord / silence slot)."""
    if idx < 0:
        return "N"
    root, quality = labels[idx]
    names = PITCH_FLAT if prefer_flats else PITCH_SHARP
    return names[root] + ("m" if quality == "min" else "")


# Krumhansl–Kessler key profiles: how "at home" each note feels in a major vs a
# minor key. Correlating these with the song's average chroma estimates the key.
_KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _estimate_key(chroma):
    """Best-fitting key as (root 0..11, "maj"/"min")."""
    import torch

    prof = chroma.mean(dim=1)
    prof = prof - prof.mean()
    best = None
    for mode, kk in (("maj", _KK_MAJOR), ("min", _KK_MINOR)):
        k = torch.tensor(kk)
        k = k - k.mean()
        for root in range(12):
            rot = torch.roll(k, root)
            corr = float((prof * rot).sum() / (prof.norm() * rot.norm() + 1e-9))
            if best is None or corr > best[0]:
                best = (corr, root, mode)
    return best[1], best[2]


def _smooth(idx, win):
    """Majority-vote each label against its neighbours to kill 1-frame jitter."""
    from collections import Counter

    n, half, out = len(idx), win // 2, []
    for i in range(n):
        window = idx[max(0, i - half):min(n, i + half + 1)]
        out.append(Counter(window).most_common(1)[0][0])
    return out


def _merge_to_spans(idx, labels, frame_dur, min_dur, prefer_flats):
    """Run-length encode the per-frame labels, drop blips, return timed chords."""
    spans, i, n = [], 0, len(idx)
    while i < n:
        j = i
        while j < n and idx[j] == idx[i]:
            j += 1
        spans.append([i * frame_dur, j * frame_dur, idx[i]])
        i = j

    # Repeatedly absorb any too-short chord into a neighbour, then re-merge
    # runs that became equal, until everything left is at least min_dur long.
    changed = True
    while changed and len(spans) > 1:
        changed = False
        for k in range(len(spans)):
            if spans[k][1] - spans[k][0] < min_dur:
                if k > 0:
                    spans[k - 1][1] = spans[k][1]
                    del spans[k]
                else:
                    spans[1][0] = spans[0][0]
                    del spans[0]
                changed = True
                break
        merged = [spans[0]]
        for s in spans[1:]:
            if s[2] == merged[-1][2]:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        spans = merged

    return [
        {"start": round(st, 2), "end": round(en, 2),
         "name": _name(ix, labels, prefer_flats)}
        for st, en, ix in spans
    ]


def detect_chords(audio_path, max_seconds: float = 0) -> dict:
    """LISTEN to an instrumental track and return its key + timed chords.

    Returns {"key": "C", "spans": [{"start", "end", "name"}, ...]}. Best fed the
    Demucs instrumental (no_vocals.wav): with the singing removed, the harmony is
    much clearer. Honest about its limits — see the module docstring.
    """
    import torch

    y, sr = _load_mono(audio_path)
    if max_seconds and max_seconds > 0:
        y = y[: int(max_seconds * sr)]
    if sr != _SR:
        import torchaudio

        y = torchaudio.functional.resample(y, sr, _SR)
        sr = _SR
    if y.numel() < _N_FFT:                          # too short to analyse
        return {"key": "", "spans": []}

    chroma = _chromagram(y, sr)                     # (12, frames)
    key_root, key_mode = _estimate_key(chroma)
    key_name = PITCH_SHARP[key_root] + ("m" if key_mode == "min" else "")
    prefer_flats = key_prefers_flats(key_name)
    if prefer_flats:
        key_name = PITCH_FLAT[key_root] + ("m" if key_mode == "min" else "")

    T, labels = _templates()
    energy = chroma.sum(dim=0)                      # crude loudness per frame
    unit = chroma / (chroma.norm(dim=0, keepdim=True) + 1e-9)
    best = (T @ unit).argmax(dim=0)                 # closest template per frame

    pos = energy[energy > 0]
    floor = 0.10 * float(torch.median(pos)) if pos.numel() else 0.0
    idx = [int(b) if float(e) >= floor else -1
           for b, e in zip(best.tolist(), energy.tolist())]

    frame_dur = _HOP / sr
    idx = _smooth(idx, max(1, round(0.5 / frame_dur)))
    spans = _merge_to_spans(idx, labels, frame_dur, min_dur=0.5,
                            prefer_flats=prefer_flats)
    return {"key": key_name, "spans": spans}


# --------------------------------------------------------------------------- #
# Self-test / quick CLI. Run `python chords.py` to check the detector on made-up
# audio, or pass a file path to dump the chords it hears from a real track.
# --------------------------------------------------------------------------- #
def _synth(progression, sr=_SR, dur=2.0):
    """Make a simple audio clip that plays each chord as three pure tones."""
    import numpy as np

    chunks = []
    for name in progression:
        p = parse_chord(name)
        intervals = [0, 3, 7] if p["suffix"].startswith("m") else [0, 4, 7]
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sig = np.zeros_like(t)
        for iv in intervals:
            f = 440.0 * 2 ** ((60 + p["root"] + iv - 69) / 12)   # ~middle octave
            sig += np.sin(2 * np.pi * f * t)
        chunks.append((sig * np.hanning(len(sig)) * 0.3).astype("float32"))
    return np.concatenate(chunks), sr


def _selftest():
    import tempfile
    from pathlib import Path

    import soundfile as sf

    print("Music-theory checks:")
    checks = [
        ("transpose C +2", transpose_chord("C", 2), "D"),
        ("transpose Am +3", transpose_chord("Am", 3), "Cm"),
        ("transpose G -2", transpose_chord("G", -2), "F"),
        ("transpose D/F# +1", transpose_chord("D/F#", 1), "D#/G"),
        ("simplify Cmaj7", simplify_chord("Cmaj7"), "C"),
        ("simplify Am7", simplify_chord("Am7"), "Am"),
        ("simplify G7sus4", simplify_chord("G7sus4"), "G"),
    ]
    ok = 0
    for label, got, want in checks:
        flag = "ok" if got == want else "FAIL"
        ok += got == want
        print(f"  [{flag}] {label}: {got}  (want {want})")
    capo, shapes = suggest_capo(["Bb", "Eb", "F"])
    print(f"  capo for Bb/Eb/F -> capo {capo}, shapes {shapes}")

    print("\nDetector check on synthetic C - F - G - Am:")
    prog = ["C", "F", "G", "Am"]
    audio, sr = _synth(prog, dur=2.0)
    tmp = Path(tempfile.gettempdir()) / "chords_selftest.wav"
    sf.write(str(tmp), audio, sr)
    res = detect_chords(tmp)
    print(f"  estimated key: {key_long(res['key'])}")
    hits = 0
    for i, want in enumerate(prog):
        center = i * 2.0 + 1.0
        got = _chord_at(center, res["spans"]) or "N"
        flag = "ok" if got == want else "??"
        hits += got == want
        print(f"  [{flag}] {center:4.1f}s  heard {got:>3}  (played {want})")
    print(f"\nTheory {ok}/{len(checks)} · detector {hits}/{len(prog)} chords matched.")


def _analyze(path):
    res = detect_chords(path)
    print(f"key: {key_long(res['key'])}\n")
    for s in res["spans"]:
        if s["name"] != "N":
            print(f"  {s['start']:7.2f} - {s['end']:7.2f}   {s['name']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        _analyze(sys.argv[1])
    else:
        _selftest()
