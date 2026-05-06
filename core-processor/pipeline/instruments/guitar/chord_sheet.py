"""
Stage 11: Chord Sheet
Input:  chord_groups [{notes, chord_name, start, duration}]
Output: PNG grid of chord box diagrams saved to outputs/11_chord_sheet.png

Each diagram shows:
  - 6 vertical string lines (low E on left, high e on right)
  - 5 fret rows
  - Filled dot with fret number at each played position
  - Barre bar drawn when 4+ strings share the same fret
  - "O" above open strings, "x" above unplayed strings
  - Fret marker on the left if the shape starts above fret 1
  - Chord name above the box

Quality improvements:
  - Barre chord detection: if 4+ strings are at the same fret, draw a horizontal
    bar across those strings instead of individual dots
  - Multiple voicings: if a chord name appears with 2+ distinct fingering patterns,
    both voicings are drawn side by side (labelled v1, v2)
  - Chord duration in progression: beat count shown as subscript after the name
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    CHORD_SHEET_DISPLAY_FRETS,
    CHORD_SHEET_COLS_PER_ROW,
    CHORD_SHEET_PROGRESSION_PER_LINE,
    CHORD_SHEET_BARRE_MIN_STRINGS,
    CHORD_SHEET_DPI,
)

# String order in the diagram: left = low E (6), right = high e (1)
STRINGS_LR   = [6, 5, 4, 3, 2, 1]   # left to right
STRING_NAMES = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}


def plot_chord_sheet(
    chord_groups: list[dict],
    tempo_info: dict | None = None,
    save: bool = True,
    show: bool = False,
) -> str:
    if not chord_groups:
        print("[Stage 11] No chords to display.")
        return ""

    named = [cg for cg in chord_groups if cg["chord_name"] != "?"]
    if not named:
        print("[Stage 11] No named chords to display.")
        return ""

    beat_s = tempo_info["beat_s"] if tempo_info else None

    # ── Collect unique voicings per chord name ────────────────────────────────
    # voicings[name] = list of (frozenset of (string, fret) tuples, notes list)
    voicings: dict[str, list[tuple[frozenset, list[dict]]]] = {}
    counts:   dict[str, int] = {}

    for cg in named:
        name = cg["chord_name"]
        counts[name] = counts.get(name, 0) + 1
        shape = frozenset((n["string"], n["fret"]) for n in cg["notes"])
        if name not in voicings:
            voicings[name] = []
        # Track unique shapes (up to 2 voicings)
        existing_shapes = [v[0] for v in voicings[name]]
        if shape not in existing_shapes and len(voicings[name]) < 2:
            voicings[name].append((shape, cg["notes"]))

    # Sort by frequency, then alphabetically
    unique_names = sorted(voicings.keys(), key=lambda n: (-counts[n], n))

    # Build flat list of diagram slots (name, notes, voicing_label)
    diagram_slots = []
    for name in unique_names:
        vs = voicings[name]
        if len(vs) == 1:
            diagram_slots.append((name, list(vs[0][1]), ""))
        else:
            diagram_slots.append((f"{name}", list(vs[0][1]), "v1"))
            diagram_slots.append((f"{name}", list(vs[1][1]), "v2"))

    n_diagrams = len(diagram_slots)
    print(f"[Stage 11] Drawing {n_diagrams} chord diagrams + "
          f"progression ({len(named)} chords)...")

    diag_rows = max(1, (n_diagrams + CHORD_SHEET_COLS_PER_ROW - 1) // CHORD_SHEET_COLS_PER_ROW)

    # ── Progression section ───────────────────────────────────────────────────
    progression = [(cg["chord_name"], cg["start"], cg["duration"]) for cg in named]
    prog_lines  = [
        progression[i : i + CHORD_SHEET_PROGRESSION_PER_LINE]
        for i in range(0, len(progression), CHORD_SHEET_PROGRESSION_PER_LINE)
    ]
    n_prog_lines = len(prog_lines)

    diag_height = diag_rows * 3.0
    prog_height = max(1.5, n_prog_lines * 0.45 + 1.0)
    total_h     = diag_height + prog_height

    fig = plt.figure(figsize=(CHORD_SHEET_COLS_PER_ROW * 2.2, total_h))
    fig.suptitle("Chord Reference Sheet", fontsize=13, fontweight="bold")

    gs = GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[diag_height, prog_height],
        hspace=0.35,
    )

    # Diagram subplots
    diag_gs = gs[0].subgridspec(diag_rows, CHORD_SHEET_COLS_PER_ROW, hspace=0.6, wspace=0.3)
    for idx, (name, notes, v_label) in enumerate(diagram_slots):
        row, col = divmod(idx, CHORD_SHEET_COLS_PER_ROW)
        ax = fig.add_subplot(diag_gs[row, col])
        display_name = f"{name} {v_label}" if v_label else name
        _draw_diagram(ax, display_name, notes,
                      counts[name.split()[0] if " " in name else name])

    # Hide unused diagram cells
    for idx in range(n_diagrams, diag_rows * CHORD_SHEET_COLS_PER_ROW):
        row, col = divmod(idx, CHORD_SHEET_COLS_PER_ROW)
        ax = fig.add_subplot(diag_gs[row, col])
        ax.set_visible(False)

    # ── Progression text ──────────────────────────────────────────────────────
    prog_ax = fig.add_subplot(gs[1])
    prog_ax.axis("off")
    prog_ax.set_xlim(0, 1)
    prog_ax.set_ylim(0, 1)

    prog_ax.text(
        0.0, 1.0, "Chord Progression",
        fontsize=11, fontweight="bold", va="top", ha="left",
        transform=prog_ax.transAxes,
    )

    line_h = 0.85 / max(n_prog_lines, 1)
    for line_idx, line_chords in enumerate(prog_lines):
        y = 0.88 - line_idx * line_h

        t     = line_chords[0][1]
        mins  = int(t // 60)
        secs  = int(t % 60)
        label = f"[{mins}:{secs:02d}]"
        prog_ax.text(
            0.0, y, label,
            fontsize=8, color="#888888", va="center", ha="left",
            transform=prog_ax.transAxes,
        )

        slot_w = 0.88 / CHORD_SHEET_PROGRESSION_PER_LINE
        for slot, (chord_name, _, dur) in enumerate(line_chords):
            x = 0.10 + slot * slot_w

            # Beat count subscript
            if beat_s and beat_s > 0:
                beats = round(dur / beat_s)
                beat_label = f"{chord_name}" + (f"({beats}b)" if beats > 0 else "")
            else:
                beat_label = chord_name

            prog_ax.text(
                x, y, beat_label,
                fontsize=10, fontweight="bold", va="center", ha="left",
                color="#111111", transform=prog_ax.transAxes,
            )
            if slot < len(line_chords) - 1:
                prog_ax.text(
                    x + slot_w - 0.01, y, "|",
                    fontsize=9, color="#cccccc", va="center", ha="center",
                    transform=prog_ax.transAxes,
                )

    out_path = ""
    if save:
        out_path = os.path.join(get_instrument_dir("guitar"), "11_chord_sheet.png")
        plt.savefig(out_path, dpi=CHORD_SHEET_DPI, bbox_inches="tight")
        print(f"[Stage 11] Saved -> {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return out_path


# ── Diagram drawing ───────────────────────────────────────────────────────────

def _draw_diagram(ax, chord_name: str, notes: list[dict], count: int) -> None:
    positions = {n["string"]: n["fret"] for n in notes}  # string -> fret

    played_frets = [f for f in positions.values() if f > 0]

    if played_frets:
        min_fret = min(played_frets)
        disp_start = 1 if min_fret <= 2 else min_fret
    else:
        disp_start = 1

    disp_end = disp_start + CHORD_SHEET_DISPLAY_FRETS - 1

    ax.set_facecolor("#faf8f4")

    # Fret lines (horizontal)
    for fret_row in range(CHORD_SHEET_DISPLAY_FRETS + 1):
        is_nut = (disp_start == 1 and fret_row == 0)
        ax.axhline(
            y=fret_row,
            color="#111111" if is_nut else "#888888",
            linewidth=3.5 if is_nut else 0.8,
            zorder=2,
        )

    # String lines (vertical)
    for x, s in enumerate(STRINGS_LR):
        ax.axvline(x=x, color="#555555", linewidth=0.9, zorder=2)

    # Open / muted indicators above diagram
    for x, s in enumerate(STRINGS_LR):
        if s in positions:
            if positions[s] == 0:
                ax.text(x, -0.55, "O", ha="center", va="center",
                        fontsize=9, color="#222222")
        else:
            ax.text(x, -0.55, "x", ha="center", va="center",
                    fontsize=9, color="#888888")

    # Fret marker on left if not at nut
    if disp_start > 1:
        ax.text(-0.6, 0.5, str(disp_start), ha="center", va="center",
                fontsize=7.5, color="#444444")

    # ── Detect barre ─────────────────────────────────────────────────────────
    # Group fretted positions by fret number;
    # if ≥ CHORD_SHEET_BARRE_MIN_STRINGS share a fret, draw a bar
    fret_to_strings: dict[int, list] = {}
    for s, f in positions.items():
        if f > 0 and disp_start <= f <= disp_end:
            fret_to_strings.setdefault(f, []).append(s)

    barre_frets: dict[int, tuple[int, int]] = {}   # fret -> (min_x, max_x)
    for fret, strings in fret_to_strings.items():
        if len(strings) >= CHORD_SHEET_BARRE_MIN_STRINGS:
            xs = [STRINGS_LR.index(s) for s in strings if s in STRINGS_LR]
            if xs:
                barre_frets[fret] = (min(xs), max(xs))

    # Draw barre bars
    for fret, (x_lo, x_hi) in barre_frets.items():
        row = fret - disp_start + 0.5
        bar = mpatches.FancyBboxPatch(
            (x_lo - 0.3, row - 0.3),
            (x_hi - x_lo) + 0.6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor="#222222", edgecolor="none", zorder=4,
        )
        ax.add_patch(bar)
        # Fret number on bar
        ax.text((x_lo + x_hi) / 2, row, str(fret),
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=5)

    # Draw individual dots for strings NOT covered by a barre
    for x, s in enumerate(STRINGS_LR):
        if s not in positions:
            continue
        fret = positions[s]
        if fret == 0:
            continue
        row = fret - disp_start + 0.5
        if not (0 <= row <= CHORD_SHEET_DISPLAY_FRETS):
            continue
        if fret in barre_frets:
            continue   # already drawn as barre
        circle = plt.Circle((x, row), 0.36, color="#222222", zorder=4)
        ax.add_patch(circle)
        ax.text(x, row, str(fret),
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white", zorder=5)

    # Axes
    ax.set_xlim(-0.8, len(STRINGS_LR) - 0.2)
    ax.set_ylim(CHORD_SHEET_DISPLAY_FRETS + 0.4, -0.9)
    ax.set_aspect("equal")
    ax.axis("off")

    # String name labels at bottom
    for x, s in enumerate(STRINGS_LR):
        ax.text(x, CHORD_SHEET_DISPLAY_FRETS + 0.3, STRING_NAMES[s],
                ha="center", va="center", fontsize=7, color="#666666")

    ax.set_title(
        f"{chord_name}  x{count}",
        fontsize=10, fontweight="bold", pad=14, color="#111111",
    )
