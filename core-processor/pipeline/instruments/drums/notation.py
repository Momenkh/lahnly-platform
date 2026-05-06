"""
Stage 8 (Drums): ASCII Drum Grid Notation
Input:  [{hit_class, start, confidence, velocity}]  from drums_onset.py
Output: text saved to outputs/08_drums_grid.txt

Grid layout:
  Rows     = drum voices (top → bottom: CY HH SN TM K)
  Columns  = 16th-note subdivisions per bar
  -        = rest
  x        = normal/hard hit (velocity >= 0.3)
  ·        = ghost hit (velocity < 0.3)
  O        = open hi-hat (hihat_open class)
  |        = bar separator

Example (1 bar, 4/4 at 120 BPM):
  [0:00 — 0:02]
  CY |----|----|----|----|
  HH |x-x-|O-x-|x-x-|O-x-|
  SN |----|x---|----|·---|
  TM |----|----|----|--x-|
  K  |x---|x---|x---|x---|
"""

import os
from pipeline.config import get_instrument_dir, get_shared_dir
from pipeline.settings import DRUMS_NOTATION_ROW_ORDER, DRUMS_GRID_SUBDIVISIONS

_ROW_LABELS = {
    "cymbal": "CY",
    "hihat":  "HH",
    "snare":  "SN",
    "tom":    "TM",
    "kick":   "K ",
}

# hihat_open / hihat_closed both render on the "hihat" row
_HIHAT_ROW_ALIASES = {"hihat_open", "hihat_closed", "hihat"}

_VELOCITY_GHOST = 0.3   # hits below this velocity render as ghost notes (·)

_BARS_PER_SECTION = 4


def generate_drum_grid(
    hits: list[dict],
    tempo_bpm: float | None = None,
    save: bool = True,
) -> str:
    """
    Convert detected drum hits to an ASCII 16th-note grid.

    Parameters
    ----------
    hits       : list of {hit_class, start, confidence}
    tempo_bpm  : detected BPM (loaded from 04_tempo.json if None)
    save       : write text to 08_drums_grid.txt

    Returns
    -------
    Full grid string.
    """
    if tempo_bpm is None:
        tempo_bpm = _load_bpm() or 120.0

    subdiv_s  = 60.0 / (tempo_bpm * (DRUMS_GRID_SUBDIVISIONS / 4))
    bar_s     = subdiv_s * DRUMS_GRID_SUBDIVISIONS

    if not hits:
        text = "[no drum hits detected]\n"
    else:
        total_dur = max(h["start"] for h in hits) + 0.5
        n_bars    = max(1, int(total_dur / bar_s) + 1)
        sections  = [
            range(i, min(i + _BARS_PER_SECTION, n_bars))
            for i in range(0, n_bars, _BARS_PER_SECTION)
        ]
        lines = []
        for sec_bars in sections:
            t_start = sec_bars[0] * bar_s
            t_end   = (sec_bars[-1] + 1) * bar_s
            lines.append(f"[{_fmt(t_start)} — {_fmt(t_end)}]")
            lines.extend(_render_section(hits, sec_bars, subdiv_s))
            lines.append("")
        text = "\n".join(lines)

    if save:
        out_path = os.path.join(get_instrument_dir("drums"), "08_grid.txt")
        with open(out_path, "w") as f:
            f.write(text)
        print(f"[Stage 8D] Saved -> {out_path}")

    return text


def _render_section(
    hits: list[dict],
    bar_range: range,
    subdiv_s: float,
) -> list[str]:
    n_bars   = len(bar_range)
    n_cols   = n_bars * DRUMS_GRID_SUBDIVISIONS
    first_bar = bar_range[0]

    # Build grid: row label → list of cells
    grid: dict[str, list[str]] = {cls: ["-"] * n_cols for cls in DRUMS_NOTATION_ROW_ORDER}

    bar_start_s = first_bar * DRUMS_GRID_SUBDIVISIONS * subdiv_s

    for hit in hits:
        t        = hit["start"]
        cls      = hit["hit_class"]
        velocity = float(hit.get("velocity", 0.7))

        # Resolve grid row — hihat sub-classes render on the "hihat" row
        if cls in _HIHAT_ROW_ALIASES:
            row = "hihat"
        elif cls in grid:
            row = cls
        else:
            continue

        col = int(round((t - bar_start_s) / subdiv_s))
        if not (0 <= col < n_cols):
            continue

        # Choose cell symbol
        if cls == "hihat_open":
            symbol = "O"
        elif velocity < _VELOCITY_GHOST:
            symbol = "·"   # · ghost note
        else:
            symbol = "x"

        grid[row][col] = symbol

    rows = []
    for cls in DRUMS_NOTATION_ROW_ORDER:
        label = _ROW_LABELS.get(cls, cls[:2].upper())
        # Insert "|" at bar boundaries
        cells = grid[cls]
        parts = []
        for bar_i in range(n_bars):
            s = bar_i * DRUMS_GRID_SUBDIVISIONS
            e = s + DRUMS_GRID_SUBDIVISIONS
            # Group into beats of 4 subdivisions: x-x-|x-x-|...
            beat_groups = []
            for beat in range(4):
                bs = s + beat * (DRUMS_GRID_SUBDIVISIONS // 4)
                be = bs + (DRUMS_GRID_SUBDIVISIONS // 4)
                beat_groups.append("".join(cells[bs:be]))
            parts.append("|".join(beat_groups))
        row_str = "|".join(f"|{p}|" for p in parts)
        rows.append(f"  {label} {row_str}")

    return rows


def _fmt(t: float) -> str:
    m = int(t) // 60
    s = int(t) % 60
    return f"{m}:{s:02d}"


def _load_bpm() -> float | None:
    import json
    path = os.path.join(get_shared_dir(), "04_tempo.json")
    try:
        with open(path) as f:
            return json.load(f).get("bpm")
    except FileNotFoundError:
        return None


def load_drum_grid() -> str:
    path = os.path.join(get_instrument_dir("drums"), "08_grid.txt")
    with open(path) as f:
        return f.read()
