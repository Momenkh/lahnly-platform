"""
Stage 10 (Piano): Piano Keyboard Diagram
Input:  mapped piano notes [{hand, key_index, pitch, start, duration, confidence}]
Output: PNG saved to outputs/10_piano_keyboard.png

Layout: one octave-wide keyboard per section row, showing which keys are
active in that time window. White keys are labeled with note names.
Left hand = blue highlight, right hand = orange, both = purple.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle, FancyBboxPatch

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    PIANO_MIDI_MIN,
    PIANO_MIDI_MAX,
    PIANO_LH_RH_SPLIT_MIDI,
    VIZ_DPI,
    VIZ_SECTION_MIN_S,
    VIZ_SECTION_MAX_S,
)

# Black key pitch classes (within octave)
_BLACK_PCS = {1, 3, 6, 8, 10}

# Colours
_LH_COLOR   = "#3a7abf"
_RH_COLOR   = "#e8802e"
_BOTH_COLOR = "#8e44ad"
_WHITE_KEY  = "#f5f5f5"
_BLACK_KEY  = "#222222"

# Note names for white keys
_NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
_WHITE_ORDER = [0, 2, 4, 5, 7, 9, 11]   # semitone offsets within octave for white keys


def _is_black(midi: int) -> bool:
    return (midi % 12) in _BLACK_PCS


def _white_key_x(midi: int) -> float:
    """X position of a white key's left edge (in white-key units from A0)."""
    octave  = (midi - 21) // 12   # octave offset from A0
    pc      = midi % 12
    # White keys within a C-based octave
    # A0=21 → pc=9, so need to handle the partial first octave
    # Simplify: count white keys below this midi
    return sum(1 for m in range(PIANO_MIDI_MIN, midi) if not _is_black(m))


def generate_piano_keyboard(
    mapped_notes: list[dict],
    key_info: dict | None = None,
    section_s: float = 4.0,
    save: bool = True,
    show: bool = False,
) -> str:
    """
    Draw a snapshot of active piano keys in each time section.
    section_s: duration of each time window shown per row.
    """
    if not mapped_notes:
        print("[Stage 10P] No notes to visualize.")
        return ""

    print(f"[Stage 10P] Rendering piano keyboard diagram for {len(mapped_notes)} notes...")

    total_dur = max(n["start"] + n["duration"] for n in mapped_notes)
    section_s  = max(VIZ_SECTION_MIN_S, min(VIZ_SECTION_MAX_S, section_s))
    n_sections = max(1, int(total_dur / section_s) + 1)

    # Total white keys on standard piano
    n_white = sum(1 for m in range(PIANO_MIDI_MIN, PIANO_MIDI_MAX + 1)
                  if not _is_black(m))

    key_label = f"  —  Key: {key_info['key_str']}" if key_info else ""
    fig_h     = 2.8 * n_sections
    fig, axes = plt.subplots(n_sections, 1, figsize=(24, fig_h), squeeze=False)
    fig.suptitle(f"Piano Keyboard Diagram{key_label}",
                 fontsize=12, y=1.002, fontweight="bold")

    for sec_idx in range(n_sections):
        ax      = axes[sec_idx][0]
        t_start = sec_idx * section_s
        t_end   = t_start + section_s

        # Collect active notes in this window, build midi → hand set
        active: dict[int, set] = {}
        for note in mapped_notes:
            n_start = note["start"]
            n_end   = n_start + note["duration"]
            if n_end < t_start or n_start > t_end:
                continue
            midi = note["pitch"]
            active.setdefault(midi, set()).add(note["hand"])

        ax.set_xlim(0, n_white)
        ax.set_ylim(0, 1.0)
        ax.set_aspect("auto")
        ax.axis("off")

        # Draw white keys first
        x = 0
        for midi in range(PIANO_MIDI_MIN, PIANO_MIDI_MAX + 1):
            if _is_black(midi):
                continue
            hands = active.get(midi, set())
            if "left" in hands and "right" in hands:
                face = _BOTH_COLOR
            elif "left" in hands:
                face = _LH_COLOR
            elif "right" in hands:
                face = _RH_COLOR
            else:
                face = _WHITE_KEY

            rect = Rectangle((x, 0), 0.92, 0.97,
                              linewidth=0.6, edgecolor="#aaaaaa", facecolor=face, zorder=1)
            ax.add_patch(rect)

            # Label C notes
            if midi % 12 == 0:
                octave_num = (midi // 12) - 1
                ax.text(x + 0.46, 0.06, f"C{octave_num}",
                        ha="center", va="bottom", fontsize=5,
                        color="#555555" if face == _WHITE_KEY else "white", zorder=3)
            x += 1

        # Draw black keys on top
        x = 0
        for midi in range(PIANO_MIDI_MIN, PIANO_MIDI_MAX + 1):
            if _is_black(midi):
                # Find x from the white key before it
                prev_white_x = sum(1 for m in range(PIANO_MIDI_MIN, midi)
                                   if not _is_black(m))
                bx = prev_white_x - 0.32
                hands = active.get(midi, set())
                if "left" in hands and "right" in hands:
                    face = _BOTH_COLOR
                elif "left" in hands:
                    face = _LH_COLOR
                elif "right" in hands:
                    face = _RH_COLOR
                else:
                    face = _BLACK_KEY

                rect = Rectangle((bx, 0.38), 0.64, 0.60,
                                  linewidth=0.4, edgecolor="#000000",
                                  facecolor=face, zorder=2)
                ax.add_patch(rect)

        # Time label
        def _fmt(s):
            return f"{int(s)//60}:{int(s)%60:02d}"

        ax.set_title(f"{_fmt(t_start)} — {_fmt(t_end)}",
                     fontsize=8, loc="left", pad=2)

    # Legend
    from matplotlib.patches import Patch
    axes[0][0].legend(
        handles=[Patch(color=_LH_COLOR,   label="Left hand"),
                 Patch(color=_RH_COLOR,   label="Right hand"),
                 Patch(color=_BOTH_COLOR,  label="Both hands")],
        loc="upper right", fontsize=8,
        facecolor="white", edgecolor="#cccccc",
    )

    plt.tight_layout(h_pad=0.8)

    out_path = ""
    if save:
        out_path = os.path.join(get_instrument_dir("piano"), "10_keyboard.png")
        plt.savefig(out_path, dpi=VIZ_DPI, bbox_inches="tight")
        print(f"[Stage 10P] Saved -> {out_path}  ({n_sections} rows)")

    if show:
        plt.show()

    plt.close(fig)
    return out_path
