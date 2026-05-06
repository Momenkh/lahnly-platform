"""
Stage 10 (Vocals): Melodic Contour + Spectrogram Overlay
Input:  mapped vocal notes [{pitch, start, duration, confidence}]
Output: PNG saved to outputs/10_vocals_contour.png

Layout per section row:
  - Spectrogram background (CQT, muted)
  - Detected F0 contour overlaid as a colored line
  - Note segments highlighted as horizontal bars with note names
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    VOCALS_MIDI_MIN, VOCALS_MIDI_MAX, VOCALS_HZ_MIN, VOCALS_HZ_MAX,
    VIZ_DPI, VIZ_SECTION_MIN_S, VIZ_SECTION_MAX_S, VIZ_SECTION_TARGET_ROWS,
    VIZ_ROW_HEIGHT_INCHES, VIZ_MIN_NOTE_WIDTH_S,
)

_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_CONTOUR_COLOR = "#e74c3c"   # red contour line
_NOTE_COLOR    = "#3498db"   # blue note bars


def _midi_to_name(midi: int) -> str:
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _fmt_mmss(x: float, _) -> str:
    return f"{int(x)//60}:{int(x)%60:02d}"


def plot_vocals_contour(
    mapped_notes: list[dict],
    stem_path: str | None = None,
    key_info: dict | None = None,
    save: bool = True,
    show: bool = False,
) -> str:
    if not mapped_notes:
        print("[Stage 10V] No notes to visualize.")
        return ""

    print(f"[Stage 10V] Rendering vocal contour for {len(mapped_notes)} notes...")

    total_dur  = max(n["start"] + n["duration"] for n in mapped_notes)
    section_s  = max(VIZ_SECTION_MIN_S, min(VIZ_SECTION_MAX_S,
                                             total_dur / VIZ_SECTION_TARGET_ROWS))
    n_sections = max(1, int(total_dur / section_s) + 1)
    key_label  = f"  -  Key: {key_info['key_str']}" if key_info else ""

    # Load stem audio for spectrogram if available
    spec_data = None
    if stem_path and os.path.isfile(stem_path):
        try:
            import librosa
            y, sr = librosa.load(stem_path, sr=22050, mono=True)
            spec_data = (y, sr)
        except Exception:
            pass

    fig, axes = plt.subplots(
        n_sections, 1,
        figsize=(22, VIZ_ROW_HEIGHT_INCHES * n_sections),
        squeeze=False,
    )
    fig.suptitle(f"Vocals Transcription — Melodic Contour{key_label}",
                 fontsize=12, y=1.002, fontweight="bold")

    for sec_idx in range(n_sections):
        ax        = axes[sec_idx][0]
        t_start   = sec_idx * section_s
        t_end     = t_start + section_s
        sec_notes = [n for n in mapped_notes if t_start <= n["start"] < t_end]

        ax.set_facecolor("#0d1117")   # dark background

        # Spectrogram background
        if spec_data is not None:
            try:
                import librosa
                y, sr = spec_data
                start_sample = int(t_start * sr)
                end_sample   = int(t_end   * sr)
                y_sec = y[start_sample:end_sample]
                if len(y_sec) > 0:
                    D = librosa.amplitude_to_db(
                        np.abs(librosa.cqt(y_sec, sr=sr, hop_length=512,
                                           fmin=VOCALS_HZ_MIN, n_bins=48)),
                        ref=np.max)
                    times = librosa.times_like(D, sr=sr, hop_length=512) + t_start
                    freqs = librosa.cqt_frequencies(48, fmin=VOCALS_HZ_MIN)
                    midis = 69 + 12 * np.log2(np.maximum(freqs, 1e-6) / 440.0)
                    ax.pcolormesh(times, midis, D,
                                  cmap="magma", vmin=-60, vmax=0,
                                  alpha=0.5, shading="auto", zorder=1)
            except Exception:
                pass

        # Note bars
        for note in sec_notes:
            t   = note["start"]
            dur = max(note["duration"], VIZ_MIN_NOTE_WIDTH_S)
            p   = note["pitch"]
            conf = float(note.get("confidence", 0.5))
            alpha = max(0.5, conf)

            bar = FancyBboxPatch(
                (t, p - 0.4), dur, 0.8,
                boxstyle="round,pad=0.01",
                linewidth=0.5, edgecolor="white",
                facecolor=_NOTE_COLOR, alpha=alpha, zorder=3,
            )
            ax.add_patch(bar)

            if dur > 0.15:
                ax.text(t + dur / 2, p, _midi_to_name(p),
                        ha="center", va="center", fontsize=6,
                        fontweight="bold", color="white", zorder=4)

        # F0 contour line connecting note midpoints
        if len(sec_notes) > 1:
            xs = [n["start"] + n["duration"] / 2 for n in sec_notes]
            ys = [n["pitch"] for n in sec_notes]
            ax.plot(xs, ys, color=_CONTOUR_COLOR, linewidth=1.2,
                    alpha=0.8, zorder=2, marker="o", markersize=2)

        # Y axis: show octave C labels
        c_ticks  = [m for m in range(VOCALS_MIDI_MIN, VOCALS_MIDI_MAX + 1) if m % 12 == 0]
        c_labels = [f"C{m // 12 - 1}" for m in c_ticks]
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(VOCALS_MIDI_MIN - 0.5, VOCALS_MIDI_MAX + 0.5)
        ax.set_yticks(c_ticks)
        ax.set_yticklabels(c_labels, fontsize=8, fontfamily="monospace", color="white")
        ax.tick_params(axis="y", length=0, colors="white")
        ax.tick_params(axis="x", labelsize=7, colors="white")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_mmss))
        ax.set_title(f"{_fmt_mmss(t_start, None)} — {_fmt_mmss(t_end, None)}",
                     fontsize=8, loc="left", pad=3, color="white")
        for spine in ax.spines.values():
            spine.set_color("#333355")

    plt.tight_layout(h_pad=1.2)

    out_path = ""
    if save:
        out_path = os.path.join(get_instrument_dir("vocals"), "10_contour.png")
        plt.savefig(out_path, dpi=VIZ_DPI, bbox_inches="tight", facecolor="#0d1117")
        print(f"[Stage 10V] Saved -> {out_path}  ({n_sections} rows)")

    if show:
        plt.show()

    plt.close(fig)
    return out_path
