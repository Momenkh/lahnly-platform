"""
Stage 10: Fretboard Visualization
Input:  mapped notes [{string, fret, pitch, start, duration, confidence}]
Output: PNG saved to outputs/10_fretboard.png

Layout — 6-string neck / piano-roll hybrid:
  X axis : time (seconds within section)
  Y axis : 6 strings  (string 1 = high e at top, string 6 = low E at bottom)

Quality improvements:
  - Confidence-colored pills: dark = high confidence, light = low
  - Adaptive section length: clamps to VIZ_SECTION_TARGET_ROWS rows
    regardless of song length
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    VIZ_NOTE_HEIGHT_STRING_UNITS,
    VIZ_MIN_NOTE_WIDTH_S,
    VIZ_ROW_HEIGHT_INCHES,
    VIZ_DPI,
    VIZ_SECTION_TARGET_ROWS,
    VIZ_SECTION_MIN_S,
    VIZ_SECTION_MAX_S,
    VIZ_PILL_COLOR_HIGH,
    VIZ_PILL_COLOR_LOW,
    VIZ_TEXT_LIGHT_CONF_THRESHOLD,
    VIZ_FONT_SIZE_WIDE_NOTE,
    VIZ_FONT_SIZE_NARROW_NOTE,
    VIZ_WIDE_NOTE_THRESHOLD_S,
)

STRING_NAMES = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}


def _fmt_mmss(x: float, _) -> str:
    m = int(x) // 60
    s = int(x) % 60
    return f"{m}:{s:02d}"

# Visual thickness per string (thin at top -> thick at bottom)
STRING_LW = {1: 0.8, 2: 1.1, 3: 1.5, 4: 2.0, 5: 2.6, 6: 3.3}


def _confidence_color(conf: float) -> str:
    """Map confidence [0, 1] to a grayscale hex: low conf = light, high conf = dark."""
    v = int(VIZ_PILL_COLOR_LOW - conf * (VIZ_PILL_COLOR_LOW - VIZ_PILL_COLOR_HIGH))
    return f"#{v:02x}{v:02x}{v:02x}"


def plot_fretboard(
    mapped_notes: list[dict],
    key_info: dict | None = None,
    chord_groups: list[dict] | None = None,   # kept for signature compat, not used
    save: bool = True,
    show: bool = False,
) -> str:
    if not mapped_notes:
        print("[Stage 10] No notes to visualize.")
        return ""

    print(f"[Stage 10] Rendering fretboard diagram for {len(mapped_notes)} notes...")

    total_dur = max(n["start"] + n["duration"] for n in mapped_notes)

    # Adaptive section length: aim for VIZ_SECTION_TARGET_ROWS rows
    section_s  = max(VIZ_SECTION_MIN_S, min(VIZ_SECTION_MAX_S,
                                             total_dur / VIZ_SECTION_TARGET_ROWS))
    n_sections = max(1, int(total_dur / section_s) + 1)

    key_label = f"  -  Key: {key_info['key_str']}" if key_info else ""

    fig, axes = plt.subplots(
        n_sections, 1,
        figsize=(22, VIZ_ROW_HEIGHT_INCHES * n_sections),
        squeeze=False,
    )
    fig.suptitle(
        f"Guitar Transcription{key_label}",
        fontsize=12, y=1.002, fontweight="bold",
    )

    for sec_idx in range(n_sections):
        ax      = axes[sec_idx][0]
        t_start = sec_idx * section_s
        t_end   = t_start + section_s
        sec_notes  = [n for n in mapped_notes if t_start <= n["start"] < t_end]

        # ── Neck background ──────────────────────────────────────────────────
        ax.set_facecolor("#f0e8d8")   # warm wood tone

        # ── String lines ─────────────────────────────────────────────────────
        for s in range(1, 7):
            ax.axhline(
                y=s,
                color="#888888",
                linewidth=STRING_LW[s],
                zorder=2,
                solid_capstyle="round",
            )

        # ── Subtle time grid ─────────────────────────────────────────────────
        ax.grid(axis="x", color="#ccbbaa", linewidth=0.4, alpha=0.6, zorder=1)

        # ── Notes ────────────────────────────────────────────────────────────
        for note in sec_notes:
            s    = note["string"]
            fret = note["fret"]
            t    = note["start"]
            dur  = max(note["duration"], VIZ_MIN_NOTE_WIDTH_S)
            conf = float(note.get("confidence", 0.5))
            color = _confidence_color(conf)

            pill = FancyBboxPatch(
                (t, s - VIZ_NOTE_HEIGHT_STRING_UNITS / 2),
                dur, VIZ_NOTE_HEIGHT_STRING_UNITS,
                boxstyle="round,pad=0.02",
                linewidth=0.8,
                edgecolor="white",
                facecolor=color,
                alpha=0.90,
                zorder=4,
            )
            ax.add_patch(pill)

            # Fret number inside pill — white text for dark pills, dark for light
            text_color = "white" if conf >= VIZ_TEXT_LIGHT_CONF_THRESHOLD else "#333333"
            fontsize = (VIZ_FONT_SIZE_WIDE_NOTE if dur > VIZ_WIDE_NOTE_THRESHOLD_S
                        else VIZ_FONT_SIZE_NARROW_NOTE)
            ax.text(
                t + dur / 2, s,
                str(fret),
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=text_color, zorder=5,
            )

        # ── Axes formatting ───────────────────────────────────────────────────
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(6.55, 0.45)           # string 1 (e) at top, 6 (E) at bottom

        ax.set_yticks(range(1, 7))
        ax.set_yticklabels(
            [f"{STRING_NAMES[s]}  " for s in range(1, 7)],
            fontsize=9, fontfamily="monospace",
        )
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_mmss))

        ax.set_xlabel(
            f"[{_fmt_mmss(t_start, None)} — {_fmt_mmss(t_end, None)}]",
            fontsize=7,
        )
        ax.set_title(
            f"{_fmt_mmss(t_start, None)} — {_fmt_mmss(t_end, None)}",
            fontsize=8, loc="left", pad=3,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#aaaaaa")

    plt.tight_layout(h_pad=1.2)

    out_path = ""
    if save:
        out_path = os.path.join(get_instrument_dir("guitar"), "10_fretboard.png")
        plt.savefig(out_path, dpi=VIZ_DPI, bbox_inches="tight")
        print(f"[Stage 10] Saved -> {out_path}  "
              f"({n_sections} rows x {section_s:.0f}s each)")

    if show:
        plt.show()

    plt.close(fig)
    return out_path
