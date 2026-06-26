# Sonari UI/UX Flow Rework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate 3 processing tabs → 2 (Speech + Music), auto-reveal results, improve results layout, improve library UX.

**Architecture:** All changes in `app.py`. Auto-reveal via `pbar-done` CSS class + JS MutationObserver that clicks hidden `auto-reveal-btn` buttons. Music tab (new) replaces Song+YouTube tabs with a source toggle (file vs link). Speech tab gets same source toggle. Results drop accordions around karaoke/chords, use a styled row header. Library auto-opens on dropdown change.

**Tech Stack:** Python, Gradio, HTML/CSS/JS (no new dependencies)

## Global Constraints
- No new Python dependencies
- Engine functions `_process_speech`, `_process_lyrics`, `download_audio` unchanged
- All changes in `app.py`

---

### Task 1: Auto-reveal infrastructure

**Files:**
- Modify: `app.py` — `progress_html()`, `CUSTOM_CSS`, `INIT_JS`

- [ ] Update `progress_html` to add `pbar-done` class when percent==100
- [ ] Add `.auto-reveal-btn{display:none!important}` to CUSTOM_CSS
- [ ] Add MutationObserver to INIT_JS that clicks `.auto-reveal-btn` when `.pbar-done` appears
- [ ] Commit

---

### Task 2: Speech tab refactor

**Files:**
- Modify: `app.py` — Speech section of `build_ui()`

Changes:
- Add `sp_src` radio (Upload / Record | Paste a link) at top
- Wrap `sp_audio` in `gr.Group() as sp_file_grp`
- Add `gr.Group(visible=False) as sp_url_grp` with `sp_url = gr.Textbox`
- Make `sp_show` button use `elem_classes=["auto-reveal-btn"]`, remove instructional text
- Remove accordion around karaoke in results; transcript always visible; Details accordion collapsed
- Update `ui_speech()` to accept `src` + `url` inputs and download if link
- Wire `sp_src.change` to toggle group visibility

- [ ] Implement all above
- [ ] Commit

---

### Task 3: Music tab (replaces Song + YouTube)

**Files:**
- Modify: `app.py` — remove Song tab, remove YouTube tab, add Music tab

Changes:
- Single `🎵 Music → lyrics` tab
- Source toggle: "Upload a file" | "Paste a link"
- `ui_music()`: if file → `_process_lyrics` directly; if link → `download_audio` then `_process_lyrics`
- Results: no accordion around karaoke; Guitar chords accordion open; Details accordion collapsed
- `mu_show` button: `elem_classes=["auto-reveal-btn"]`

- [ ] Implement all above
- [ ] Commit

---

### Task 4: Library auto-open

**Files:**
- Modify: `app.py` — library section of `build_ui()`

Changes:
- Remove "Open ▶" button
- Wire `lib_pick.change → _open_library` (auto-open on select)
- `_open_library` returns `{lib_results: visible=False}` when item_id is None/empty

- [ ] Implement all above
- [ ] Commit

---

### Task 5: CSS polish + Guide update

**Files:**
- Modify: `app.py` — CUSTOM_CSS, GUIDE_HTML

CSS additions:
- `.res-row-header` — flex row, margin-bottom for results header
- `button.res-back` — border, transparent bg, indigo text, hover fill
- `.src-toggle .wrap` — subtle bg for source radio group

Guide update:
- Change "Song → lyrics" → "Music → lyrics"
- Remove the separate "From a link" guide section; fold YouTube tip into Music section
- Update "3 tabs" references

- [ ] Implement all above
- [ ] Commit
