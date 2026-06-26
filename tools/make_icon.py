#!/usr/bin/env python
"""
make_icon.py  -  generate Sonari's app icon (run once; re-run to tweak the look)
================================================================================

Draws a clean, modern app icon — an indigo→violet rounded square with a white
audio-waveform motif — and writes two files into ../assets:

    sonari.ico   multi-resolution Windows icon (256/128/64/48/32/16) used by the
                 desktop shortcut (see make_shortcut.ps1 / make_shortcut.bat)
    sonari.png   512px PNG used as the browser tab favicon in app.py

No design tools needed — it's pure Pillow + NumPy, so anyone can regenerate it:

    venv\\Scripts\\python.exe tools\\make_icon.py        (Windows)
    venv/bin/python tools/make_icon.py                 (macOS / Linux)
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

S = 1024                      # master canvas (super-sampled, then shrunk = crisp)
RADIUS = int(S * 0.235)       # corner radius of the rounded "app tile"

# Brand palette (matches the web UI's indigo→violet theme).
INDIGO = np.array([99, 102, 241])    # #6366f1
VIOLET = np.array([139, 92, 246])    # #8b5cf6
PURPLE = np.array([168, 85, 247])    # #a855f7


def _diagonal_gradient() -> Image.Image:
    """A smooth 3-stop diagonal gradient (indigo → violet → purple)."""
    yy, xx = np.mgrid[0:S, 0:S]
    t = (xx + yy) / (2 * (S - 1))                      # 0 (top-left) → 1 (bottom-right)
    lo = (np.clip(t / 0.5, 0, 1))[..., None]           # first half: indigo → violet
    hi = (np.clip((t - 0.5) / 0.5, 0, 1))[..., None]   # second half: violet → purple
    first = INDIGO * (1 - lo) + VIOLET * lo
    second = VIOLET * (1 - hi) + PURPLE * hi
    grad = np.where((t < 0.5)[..., None], first, second)
    return Image.fromarray(grad.astype("uint8"), "RGB")


def _rounded_mask() -> Image.Image:
    """Alpha mask shaped like the rounded app tile."""
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=RADIUS, fill=255)
    return mask


def _add_top_sheen(img: Image.Image) -> None:
    """Lay a faint top-down white sheen over the tile for a little depth."""
    yy = np.mgrid[0:S, 0:S][0]
    alpha = np.clip(1 - yy / (S * 0.62), 0, 1) ** 2 * 46   # strongest at the very top
    sheen = np.zeros((S, S, 4), "uint8")
    sheen[..., :3] = 255
    sheen[..., 3] = alpha.astype("uint8")
    img.alpha_composite(Image.fromarray(sheen, "RGBA"))


def _draw_waveform(img: Image.Image) -> None:
    """Draw centered, rounded white bars — a stylized audio waveform."""
    draw = ImageDraw.Draw(img)
    heights = [0.30, 0.55, 0.40, 0.86, 0.50, 0.66, 0.34]   # organic, not a triangle
    n = len(heights)
    span = S * 0.62                       # the bars occupy ~62% of the width
    gap = span / n * 0.42
    bar_w = (span - gap * (n - 1)) / n
    x0 = (S - span) / 2
    cy = S / 2
    r = bar_w / 2
    for i, h in enumerate(heights):
        bh = S * h
        left = x0 + i * (bar_w + gap)
        draw.rounded_rectangle(
            [left, cy - bh / 2, left + bar_w, cy + bh / 2], radius=r, fill=(255, 255, 255, 255)
        )


def build_icon() -> Image.Image:
    """Compose the full RGBA icon at master resolution."""
    icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    icon.paste(_diagonal_gradient(), (0, 0), _rounded_mask())
    _add_top_sheen(icon)
    _draw_waveform(icon)
    return icon


def main() -> None:
    icon = build_icon()

    ico_path = ASSETS / "sonari.ico"
    icon.resize((256, 256), Image.LANCZOS).save(
        ico_path, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    )

    png_path = ASSETS / "sonari.png"
    icon.resize((512, 512), Image.LANCZOS).save(png_path)

    print(f"[icon] wrote {ico_path}")
    print(f"[icon] wrote {png_path}")


if __name__ == "__main__":
    main()
