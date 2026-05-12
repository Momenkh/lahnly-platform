"""
Stage 8 (Piano): Piano Roll
Input:  mapped piano notes [{hand, key_index, pitch, start, duration, confidence}]
Output: PNG piano roll saved to outputs/08_piano_roll.png

Layout:
  X axis : time (seconds)
  Y axis : MIDI pitch (PIANO_MIDI_MIN at bottom, PIANO_MIDI_MAX at top)

Left hand notes are rendered in blue, right hand in orange.
Confidence maps to note opacity.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    PIANO_MIDI_MIN,
    PIANO_MIDI_MAX,
    PIANO_LH_RH_SPLIT_MIDI,
    VIZ_DPI,
    VIZ_SECTION_TARGET_ROWS,
    VIZ_SECTION_MIN_S,
    VIZ_SECTION_MAX_S,
    VIZ_ROW_HEIGHT_INCHES,
    VIZ_MIN_NOTE_WIDTH_S,
)

# Color palette: left hand blue, right hand orange
_LH_COLOR = "#3a7abf"
_RH_COLOR = "#e8802e"
_OCTAVE_LINES = {n % 12 == 0 for n in range(PIANO_MIDI_MIN, PIANO_MIDI_MAX + 1)}


def _fmt_mmss(x: float, _) -> str:
    m = int(x) // 60
    s = int(x) % 60
    return f"{m}:{s:02d}"


def generate_piano_roll(
    mapped_notes: list[dict],
    key_info: dict | None = None,
    save: bool = True,
    show: bool = False,
) -> str:
    if not mapped_notes:
        print("[Stage 8P] No notes to render.")
        return ""

    print(f"[Stage 8P] Rendering piano roll for {len(mapped_notes)} notes...")

    total_dur = max(n["start"] + n["duration"] for n in mapped_notes)
    section_s  = max(VIZ_SECTION_MIN_S, min(VIZ_SECTION_MAX_S,
                                             total_dur / VIZ_SECTION_TARGET_ROWS))
    n_sections = max(1, int(total_dur / section_s) + 1)

    key_label = f"  -  Key: {key_info['key_str']}" if key_info else ""

    # Piano roll height: each MIDI semitone gets a fixed pixel height
    pitch_range = PIANO_MIDI_MAX - PIANO_MIDI_MIN + 1
    row_h = VIZ_ROW_HEIGHT_INCHES * 1.5   # taller rows since pitch axis is dense

    fig, axes = plt.subplots(
        n_sections, 1,
        figsize=(22, row_h * n_sections),
        squeeze=False,
    )
    fig.suptitle(
        f"Piano Transcription — Roll View{key_label}",
        fontsize=12, y=1.002, fontweight="bold",
    )

    for sec_idx in range(n_sections):
        ax        = axes[sec_idx][0]
        t_start   = sec_idx * section_s
        t_end     = t_start + section_s
        sec_notes = [n for n in mapped_notes
                     if n["start"] < t_end and n["start"] + n["duration"] > t_start]

        ax.set_facecolor("#1a1a2e")   # dark background for piano roll

        # Octave guide lines
        for midi in range(PIANO_MIDI_MIN, PIANO_MIDI_MAX + 1):
            if midi % 12 == 0:   # C notes
                ax.axhline(y=midi, color="#444466", linewidth=0.6, zorder=1)
            elif midi % 12 in {1, 3, 6, 8, 10}:   # black keys
                ax.axhspan(midi - 0.5, midi + 0.5,
                           facecolor="#22223a", alpha=0.5, zorder=0)

        # Middle C split line
        ax.axhline(y=PIANO_LH_RH_SPLIT_MIDI, color="#888888",
                   linewidth=1.2, linestyle="--", zorder=2, alpha=0.7)

        # Notes
        for note in sec_notes:
            pitch  = note["pitch"]
            t      = note["start"]
            dur    = max(note["duration"], VIZ_MIN_NOTE_WIDTH_S)
            conf   = float(note.get("confidence", 0.5))
            color  = _LH_COLOR if note["hand"] == "left" else _RH_COLOR

            rect = FancyBboxPatch(
                (t, pitch - 0.45),
                dur, 0.90,
                boxstyle="round,pad=0.01",
                linewidth=0.5,
                edgecolor="white",
                facecolor=color,
                alpha=max(0.4, conf),
                zorder=3,
            )
            ax.add_patch(rect)

        # Y axis: show note names at C positions
        c_ticks = [m for m in range(PIANO_MIDI_MIN, PIANO_MIDI_MAX + 1) if m % 12 == 0]
        c_labels = [f"C{(m - 12) // 12}" for m in c_ticks]

        ax.set_xlim(t_start, t_end)
        ax.set_ylim(PIANO_MIDI_MIN - 0.5, PIANO_MIDI_MAX + 0.5)
        ax.set_yticks(c_ticks)
        ax.set_yticklabels(c_labels, fontsize=7, fontfamily="monospace", color="white")
        ax.tick_params(axis="y", length=0, colors="white")
        ax.tick_params(axis="x", labelsize=7, colors="white")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_mmss))
        ax.set_title(
            f"{_fmt_mmss(t_start, None)} — {_fmt_mmss(t_end, None)}",
            fontsize=8, loc="left", pad=3, color="white",
        )
        for spine in ax.spines.values():
            spine.set_color("#444466")

    # Legend
    from matplotlib.patches import Patch
    axes[0][0].legend(
        handles=[Patch(color=_LH_COLOR, label="Left hand"),
                 Patch(color=_RH_COLOR, label="Right hand")],
        loc="upper right", fontsize=8,
        facecolor="#1a1a2e", edgecolor="#666688", labelcolor="white",
    )

    plt.tight_layout(h_pad=1.2)

    out_path = ""
    if save:
        out_path = os.path.join(get_instrument_dir("piano"), "08_piano_roll.png")
        plt.savefig(out_path, dpi=VIZ_DPI, bbox_inches="tight",
                    facecolor="#1a1a2e")
        print(f"[Stage 8P] Saved -> {out_path}  "
              f"({n_sections} rows x {section_s:.0f}s each)")

    if show:
        plt.show()

    plt.close(fig)
    return out_path
