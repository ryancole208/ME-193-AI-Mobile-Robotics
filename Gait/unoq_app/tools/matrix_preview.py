"""Render every LED-matrix frame from sketch/sketch.ino without flashing the board.

The sketch stores its art as ASCII strings ('#' full, '+' dim, '.' off), so this
script parses those strings directly - what you see here is exactly what the
sketch draws. Output:
  * every frame printed in the terminal, composited into the full 13-column
    layout (animation in columns 0-10, model indicator in column 12)
  * reports/matrix_preview.html - all frames, with the animations playing at
    their real frame rates (open it in any browser)
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
SKETCH = APP / "sketch" / "sketch.ino"
ROWS, COLS, ART_W = 8, 13, 11
VALUES = {"#": 7, "+": 3, ".": 0}

# frame period in ms, as in drawActivity() in the sketch
ANIMATIONS = {"STILL_FRAMES": 2000, "WALK_FRAMES": 333, "RUN_FRAMES": 167, "BIKE_FRAMES": 200}

_ARRAY = re.compile(r"static const char\* const (\w+)(?:\[(\d+)\])?\[ROWS\]\s*=\s*\{(.*?)\};", re.S)
_STR = re.compile(r'"([^"]*)"')


def parse_sketch(path: Path = SKETCH) -> dict[str, list[list[str]]]:
    """name -> list of frames; each frame is 8 strings."""
    text = path.read_text()
    out = {}
    for name, count, body in _ARRAY.findall(text):
        strings = _STR.findall(body)
        if len(strings) % ROWS:
            raise ValueError(f"{name}: {len(strings)} strings is not a multiple of {ROWS} rows")
        frames = [strings[i:i + ROWS] for i in range(0, len(strings), ROWS)]
        if count and int(count) != len(frames):
            raise ValueError(f"{name}: declared {count} frames but has {len(frames)}")
        out[name] = frames
    return out


def check_frames(frames: dict) -> list[str]:
    """Problems that would make the sketch draw garbage (tests call this)."""
    problems = []
    for name, fl in frames.items():
        width = COLS if name in ("ICON_CHECK", "ICON_BATTERY") else ART_W
        for i, f in enumerate(fl):
            for r, row in enumerate(f):
                if len(row) != width:
                    problems.append(f"{name}[{i}] row {r} has {len(row)} chars, expected {width}")
                bad = set(row) - set(VALUES)
                if bad:
                    problems.append(f"{name}[{i}] row {r} has characters {sorted(bad)} (use # + .)")
    return problems


def compose(frame: list[str], model: str | None = "A") -> list[list[int]]:
    """Frame -> 8x13 brightness grid with the column-12 model indicator."""
    g = [[0] * COLS for _ in range(ROWS)]
    for r, row in enumerate(frame):
        for c, ch in enumerate(row[:COLS]):
            g[r][c] = VALUES.get(ch, 0)
    if model and len(frame[0]) == ART_W:
        for r in ((0, 1) if model == "A" else (6, 7)):
            g[r][12] = 7
    return g


def ascii_grid(g) -> list[str]:
    return ["".join("█" if v >= 6 else ("▒" if v > 0 else "·") for v in row) for row in g]


def to_html(frames: dict) -> str:
    def grid_html(g):
        cells = "".join(f'<i style="opacity:{0.08 + 0.92 * v / 7:.2f}"></i>' for row in g for v in row)
        return f'<div class="m">{cells}</div>'

    parts = []
    for name, fl in frames.items():
        period = ANIMATIONS.get(name)
        grids = [grid_html(compose(f, "A")) for f in fl]
        if period:
            inner = "".join(f'<div class="f" style="display:{"block" if i == 0 else "none"}">{g}</div>'
                            for i, g in enumerate(grids))
            parts.append(f'<section><h2>{html.escape(name)} <small>{len(fl)} frames, {1000 / period:.1f} fps</small></h2>'
                         f'<div class="anim" data-period="{period}">{inner}</div>'
                         f'<div class="row">{"".join(grids)}</div></section>')
        else:
            parts.append(f'<section><h2>{html.escape(name)}</h2><div class="row">{"".join(grids)}</div></section>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Matrix preview</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#f6f7f9;--fg:#1d2330;--led:#1f6fff;--off:#c9d3e6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#11151c;--fg:#e8ecf3;--led:#5b9bff;--off:#243048}}}}
body{{background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif;margin:16px}}
.m{{display:grid;grid-template-columns:repeat(13,12px);gap:3px;padding:8px;background:#0b0f16;border-radius:6px;width:max-content}}
.m i{{width:12px;height:12px;border-radius:50%;background:var(--led);display:block}}
.row{{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px}} section{{margin-bottom:22px}} small{{opacity:.6;font-weight:400}}
</style></head><body><h1>LED matrix preview</h1>
<p>Parsed from sketch/sketch.ino. Columns 0-10 animation, column 12 = model indicator (A shown).</p>
{''.join(parts)}
<script>
document.querySelectorAll('.anim').forEach(a=>{{const f=a.querySelectorAll('.f');let i=0;
setInterval(()=>{{f[i].style.display='none';i=(i+1)%f.length;f[i].style.display='block'}},+a.dataset.period)}});
</script></body></html>"""


def main() -> int:
    frames = parse_sketch()
    problems = check_frames(frames)
    for name, fl in frames.items():
        print(f"\n{name} ({len(fl)} frame{'s' if len(fl) != 1 else ''})")
        grids = [ascii_grid(compose(f, "A")) for f in fl]
        for r in range(ROWS):
            print("   ".join(g[r] for g in grids))
    out = APP / "reports" / "matrix_preview.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_html(frames))
    print(f"\nwrote {out}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
