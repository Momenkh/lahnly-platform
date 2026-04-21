"""
Stage 5: Key & Scale Detection
Input:  cleaned notes [{pitch, start, duration, confidence}]
Output: key/scale analysis saved to outputs/05_key_analysis.json

Algorithm: Krumhansl-Schmuckler (KS) key profiles.
Histogram is dual-weighted: 50% note duration + 50% onset count, which is more
robust than pure duration weighting (avoids sustained open strings dominating).

Quality improvements:
  - Dual-weighted histogram (duration + onset count)
  - Top-3 key candidates returned with confidence scores
  - Guitar-specific pentatonic minor bias (most common lead scale on guitar)
"""

import json
import os
import numpy as np

from pipeline.config import get_outputs_dir
from pipeline.settings import KEY_PENTATONIC_MINOR_BIAS

CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

SCALES = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":    [0, 2, 3, 5, 7, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
}

SCALE_PARENT = {
    "pentatonic_major": "major",
    "pentatonic_minor": "natural_minor",
    "blues":            "natural_minor",
    "dorian":           "natural_minor",
    "mixolydian":       "major",
}


def analyze_key(cleaned_notes: list[dict], save: bool = True) -> dict:
    """
    Detect key, mode, and scale. Returns dict with top-3 candidates.
    """
    print(f"[Stage 5] Analysing key for {len(cleaned_notes)} notes...")

    histogram = _build_histogram(cleaned_notes)
    root, mode, confidence, candidates = _detect_key(histogram)
    scale_name = _best_scale(histogram, root, mode)
    scale_pcs  = _scale_pitch_classes(root, scale_name)

    root_name  = CHROMATIC[root]
    mode_label = "major" if mode == "major" else "minor"
    penta_label = " (pentatonic)" if "pentatonic" in scale_name else \
                  " (blues)"      if scale_name == "blues" else ""
    key_str = f"{root_name} {mode_label}{penta_label}"

    result = {
        "root":       root_name,
        "root_midi":  root,
        "mode":       mode,
        "scale":      scale_name,
        "scale_pcs":  scale_pcs,
        "key_str":    key_str,
        "confidence": round(float(confidence), 4),
        "candidates": candidates,
        "histogram":  {CHROMATIC[i]: round(float(histogram[i]), 4) for i in range(12)},
    }

    print(f"[Stage 5] Detected key: {key_str}  (confidence {confidence:.2f})")
    print(f"[Stage 5] Top candidates: {[c['key_str'] for c in candidates[:3]]}")
    print(f"[Stage 5] Scale notes: {[CHROMATIC[pc] for pc in scale_pcs]}")

    if save:
        os.makedirs(get_outputs_dir(), exist_ok=True)
        out_path = os.path.join(get_outputs_dir(), "05_key_analysis.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Stage 5] Saved -> {out_path}")

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_histogram(notes: list[dict]) -> np.ndarray:
    """
    Dual-weighted pitch class histogram: 50% duration + 50% onset count.
    Normalised to sum=1.
    """
    dur_hist    = np.zeros(12)
    onset_hist  = np.zeros(12)

    for n in notes:
        pc = n["pitch"] % 12
        dur_hist[pc]   += n["duration"]
        onset_hist[pc] += 1

    # Normalise each component independently then blend
    if dur_hist.sum() > 0:
        dur_hist /= dur_hist.sum()
    if onset_hist.sum() > 0:
        onset_hist /= onset_hist.sum()

    return 0.5 * dur_hist + 0.5 * onset_hist


def _detect_key(histogram: np.ndarray) -> tuple[int, str, float, list[dict]]:
    """
    KS correlation against all 24 key profiles.
    Returns (root, mode, best_r, top3_candidates).
    """
    scores = []
    for root in range(12):
        for mode, profile in [("major", KS_MAJOR), ("minor", KS_MINOR)]:
            rotated = np.roll(profile, root)
            r = float(np.corrcoef(histogram, rotated)[0, 1])
            scores.append((r, root, mode))

    scores.sort(reverse=True)

    best_r, best_root, best_mode = scores[0]

    candidates = []
    for r, root, mode in scores[:5]:
        rn      = CHROMATIC[root]
        ml      = "major" if mode == "major" else "minor"
        candidates.append({"key_str": f"{rn} {ml}", "confidence": round(r, 4)})

    return best_root, best_mode, best_r, candidates


def _best_scale(histogram: np.ndarray, root: int, mode: str) -> str:
    """Pick the scale whose notes account for the highest weighted coverage."""
    diatonic   = "major" if mode == "major" else "natural_minor"
    parent_key = "major" if mode == "major" else "natural_minor"
    candidates = [
        name for name, parent in SCALE_PARENT.items()
        if parent == parent_key
    ] + [diatonic]

    best_score = -1.0
    best_name  = diatonic

    for name in candidates:
        pcs   = set(_scale_pitch_classes(root, name))
        score = sum(histogram[pc] for pc in pcs)
        # Guitar-specific bias for pentatonic minor
        if name == "pentatonic_minor":
            score += KEY_PENTATONIC_MINOR_BIAS
        if score > best_score:
            best_score = score
            best_name  = name

    return best_name


def _scale_pitch_classes(root: int, scale_name: str) -> list[int]:
    intervals = SCALES[scale_name]
    return [(root + interval) % 12 for interval in intervals]


def get_scale_pitch_classes(root_midi: int, scale_name: str) -> list[int]:
    return _scale_pitch_classes(root_midi, scale_name)


def load_key_analysis() -> dict:
    path = os.path.join(get_outputs_dir(), "05_key_analysis.json")
    with open(path) as f:
        return json.load(f)
