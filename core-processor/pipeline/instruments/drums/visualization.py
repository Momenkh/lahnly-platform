"""
Stage 10 (Drums): Drum Pattern Visualization
Input:  [{hit_class, start, confidence}]  from drums_onset.py
Output: PNG saved to outputs/10_drums_pattern.png

Piano-roll-style grid:
  Rows    = drum voices (y-axis)
  Columns = time (x-axis, seconds)
  Markers = vertical lines at each detected hit, coloured by class
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pipeline.config import get_instrument_dir
from pipeline.settings import DRUMS_NOTATION_ROW_ORDER, VIZ_DPI

_VOICE_COLORS = {
    "cymbal":       "#f39c12",   # gold
    "hihat":        "#2ecc71",   # green
    "hihat_open":   "#2ecc71",   # green (same row as hihat)
    "hihat_closed": "#2ecc71",   # green (same row as hihat)
    "snare":        "#3498db",   # blue
    "tom":          "#9b59b6",   # purple
    "kick":         "#e74c3c",   # red
}

_VOICE_LABELS = {
    "cymbal": "Cymbal / Ride",
    "hihat":  "Hi-Hat",
    "snare":  "Snare",
    "tom":    "Tom",
    "kick":   "Kick",
}

# Map sub-classes to their visualization row (the canonical row name)
_ROW_ALIAS = {
    "hihat_open":   "hihat",
    "hihat_closed": "hihat",
}

_SECTION_S = 8.0   # seconds per row


def plot_drum_pattern(
    hits: list[dict],
    save: bool = True,
    show: bool = False,
) -> str:
    if not hits:
        print("[Stage 10D] No drum hits to visualize.")
        return ""

    print(f"[Stage 10D] Rendering drum pattern for {len(hits)} hits...")

    total_dur = max(h["start"] for h in hits) + 1.0
    n_sections = max(1, int(total_dur / _SECTION_S) + 1)
    n_voices   = len(DRUMS_NOTATION_ROW_ORDER)

    fig, axes = plt.subplots(
        n_sections, 1,
        figsize=(22, 1.8 * n_sections),
        squeeze=False,
    )
    fig.suptitle("Drum Pattern — Hit Detection", fontsize=12, fontweight="bold",
                 y=1.002, color="white")
    fig.patch.set_facecolor("#0d1117")

    for sec_idx in range(n_sections):
        ax      = axes[sec_idx][0]
        t_start = sec_idx * _SECTION_S
        t_end   = t_start + _SECTION_S
        sec_hits = [h for h in hits if t_start <= h["start"] < t_end]

        ax.set_facecolor("#161b22")

        # Horizontal lane separators
        for vi in range(n_voices + 1):
            ax.axhline(vi, color="#30363d", linewidth=0.5, zorder=1)

        # Voice labels on y-axis
        voice_order = DRUMS_NOTATION_ROW_ORDER  # top = cymbal, bottom = kick
        y_pos = {cls: (n_voices - 1 - i) + 0.5 for i, cls in enumerate(voice_order)}

        ax.set_ylim(0, n_voices)
        ax.set_yticks([y_pos[cls] for cls in voice_order])
        ax.set_yticklabels(
            [_VOICE_LABELS.get(cls, cls) for cls in voice_order],
            fontsize=8, color="white", fontfamily="monospace",
        )
        ax.tick_params(axis="y", length=0)

        # Plot each hit as a vertical marker + horizontal line
        for hit in sec_hits:
            t    = hit["start"]
            cls  = hit["hit_class"]
            row  = _ROW_ALIAS.get(cls, cls)   # resolve sub-classes to their grid row
            if row not in y_pos:
                continue
            y        = y_pos[row]
            conf     = float(hit.get("confidence", 0.7))
            velocity = float(hit.get("velocity", conf))   # use velocity for visual intensity
            color    = _VOICE_COLORS.get(cls, "#ffffff")
            lane_bottom = y - 0.5
            # Full-height lane marker — faint flash, scaled by velocity
            ax.axvspan(t - 0.005, t + 0.020, ymin=(lane_bottom) / n_voices,
                       ymax=(lane_bottom + 1) / n_voices,
                       color=color, alpha=max(0.05, 0.15 * velocity), zorder=2)
            # Sharp hit line — width from confidence, alpha from velocity
            ax.plot([t, t], [lane_bottom + 0.05, lane_bottom + 0.95],
                    color=color, linewidth=max(0.8, 2.0 * conf),
                    alpha=min(1.0, 0.3 + 0.7 * velocity), zorder=3)
            # Dot at hit position — size from confidence, alpha from velocity
            ax.scatter([t], [y], color=color, s=20 * conf,
                       alpha=min(1.0, 0.4 + 0.6 * velocity), zorder=4, linewidths=0)

        # X axis
        ax.set_xlim(t_start, t_end)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x)//60}:{int(x)%60:02d}"
        ))
        ax.tick_params(axis="x", labelsize=7, colors="white")
        ax.set_title(
            f"{_fmt(t_start)} — {_fmt(t_end)}",
            fontsize=8, loc="left", pad=3, color="white",
        )
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    plt.tight_layout(h_pad=0.8)

    out_path = ""
    if save:
        out_path = os.path.join(get_instrument_dir("drums"), "10_pattern.png")
        plt.savefig(out_path, dpi=VIZ_DPI, bbox_inches="tight", facecolor="#0d1117")
        print(f"[Stage 10D] Saved -> {out_path}  ({n_sections} rows)")

    if show:
        plt.show()

    plt.close(fig)
    return out_path


def _fmt(t: float) -> str:
    m = int(t) // 60
    s = int(t) % 60
    return f"{m}:{s:02d}"
