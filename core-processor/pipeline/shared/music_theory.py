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

from pipeline.config import get_shared_dir
from pipeline.settings import (
    KEY_PENTATONIC_MINOR_BIAS, KEY_PENTA_BIAS_GATE,
    KEY_MAQAM_MIN_GAP, KEY_MAQAM_MIN_NOTES,
)

CHROMATIC      = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHROMATIC_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Keys whose key signature uses flats rather than sharps.
# Enharmonic equivalents (e.g. F#/Gb major) are assigned to the flat side.
_FLAT_MAJOR_ROOTS = frozenset({1, 3, 5, 6, 8, 10})   # Db Eb F Gb Ab Bb
_FLAT_MINOR_ROOTS = frozenset({0, 2, 3, 5, 7, 10})   # C  D  Eb F  G  Bb
# Maqam keys: b2 is universal in all three maqams below, so flat notation is
# always more readable (Eb rather than D#, Bb rather than A#, etc.).
_FLAT_MAQAM_ROOTS = frozenset({0, 1, 2, 3, 5, 7, 8, 10})  # broad flat preference


def key_uses_flats(root: int, mode: str) -> bool:
    """Return True when the key signature for (root, mode) uses flat accidentals."""
    if mode.startswith("maqam_"):
        return root in _FLAT_MAQAM_ROOTS
    return root in (_FLAT_MAJOR_ROOTS if mode == "major" else _FLAT_MINOR_ROOTS)


def note_name(pitch_class: int, use_flats: bool) -> str:
    """Return the canonical note name for a pitch class given the key context."""
    return (CHROMATIC_FLAT if use_flats else CHROMATIC)[pitch_class % 12]

KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# ── Arabic maqam KS profiles ──────────────────────────────────────────────────
# Each profile is a 12-element root-position weight vector (position 0 = root).
# Weights follow the KS convention: higher = more characteristic of the mode.
# Non-scale tones get ~2.2 (not zero — chromatic passing tones are common).
#
# Maqam Hijaz: 1 b2 3 4 5 b6 b7  (intervals: 0 1 4 5 7 8 10)
#   Defining feature: augmented 2nd between b2 and 3 (e.g. Eb→F# on root D).
#   Most common maqam in Arabic pop.
KS_MAQAM_HIJAZ = np.array([6.30, 3.50, 2.20, 2.20, 4.50, 4.00,
                            2.20, 5.00, 3.20, 2.20, 3.50, 2.20])

# Maqam Kurd: 1 b2 b3 4 5 b6 b7  (intervals: 0 1 3 5 7 8 10)
#   Equivalent to Phrygian in Western theory. Common in Turkish and Arabic music.
#   The b2 is its most characteristic element relative to natural minor.
KS_MAQAM_KURD  = np.array([6.30, 4.00, 2.20, 3.50, 2.20, 3.80,
                            2.20, 4.80, 3.20, 2.20, 3.50, 2.20])

# Maqam Saba: 1 b2 b3 b4 5 b6 b7  (intervals: 0 1 3 4 7 8 10)
#   Defining feature: chromatic cluster b2-b3-b4 (semitone steps Eb-F-Gb on D).
#   The diminished 4th (tritone from root to b4) is unique to Saba.
KS_MAQAM_SABA  = np.array([6.30, 3.20, 2.20, 3.80, 3.50, 2.20,
                            2.20, 4.80, 3.20, 2.20, 3.50, 2.20])

# Map from mode string → KS profile (used in _detect_key)
MAQAM_PROFILES: dict[str, np.ndarray] = {
    "maqam_hijaz": KS_MAQAM_HIJAZ,
    "maqam_kurd":  KS_MAQAM_KURD,
    "maqam_saba":  KS_MAQAM_SABA,
}

# Human-readable labels for maqam mode strings
MAQAM_LABELS: dict[str, str] = {
    "maqam_hijaz": "Hijaz",
    "maqam_kurd":  "Kurd",
    "maqam_saba":  "Saba",
}

SCALES = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":    [0, 2, 3, 5, 7, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    # Maqam scales — intervals from root in semitones
    "maqam_hijaz":      [0, 1, 4, 5, 7, 8, 10],
    "maqam_kurd":       [0, 1, 3, 5, 7, 8, 10],
    "maqam_saba":       [0, 1, 3, 4, 7, 8, 10],
}

SCALE_PARENT = {
    "pentatonic_major": "major",
    "pentatonic_minor": "natural_minor",
    "blues":            "natural_minor",
    "dorian":           "natural_minor",
    "mixolydian":       "major",
    # Maqam scales are self-contained — no Western parent
}


def analyze_key(cleaned_notes: list[dict], save: bool = True, instrument: str = "guitar") -> dict:
    """
    Detect key, mode, and scale. Returns dict with top-3 candidates.
    instrument: controls instrument-specific scale biases (e.g. pentatonic_minor for guitar only)
    """
    print(f"[Stage 5] Analysing key for {len(cleaned_notes)} notes...")

    histogram = _build_histogram(cleaned_notes)
    root, mode, confidence, candidates = _detect_key(histogram, len(cleaned_notes))
    is_maqam   = mode.startswith("maqam_")
    scale_name = mode if is_maqam else _best_scale(histogram, root, mode, instrument=instrument)
    scale_pcs  = _scale_pitch_classes(root, scale_name)

    use_flats  = key_uses_flats(root, mode)
    root_name  = note_name(root, use_flats)
    if is_maqam:
        key_str = f"{root_name} {MAQAM_LABELS[mode]}"
    else:
        mode_label  = "major" if mode == "major" else "minor"
        penta_label = " (pentatonic)" if "pentatonic" in scale_name else \
                      " (blues)"      if scale_name == "blues" else ""
        key_str = f"{root_name} {mode_label}{penta_label}"

    result = {
        "root":       root_name,
        "root_midi":  root,
        "mode":       mode,
        "use_flats":  use_flats,
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
        out_path = os.path.join(get_shared_dir(), "05_key_analysis.json")
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


def _detect_key(histogram: np.ndarray, n_notes: int) -> tuple[int, str, float, list[dict]]:
    """
    KS correlation against all key profiles: 24 Western + 3 maqam × 12 roots.
    Maqam wins only when n_notes >= KEY_MAQAM_MIN_NOTES AND the maqam score
    beats the best Western score by at least KEY_MAQAM_MIN_GAP.
    Returns (root, mode, best_r, top5_candidates).
    """
    scores = []
    western = [("major", KS_MAJOR), ("minor", KS_MINOR)]
    for root in range(12):
        for mode, profile in western:
            rotated = np.roll(profile, root)
            r = float(np.corrcoef(histogram, rotated)[0, 1])
            scores.append((r, root, mode))
        for mode, profile in MAQAM_PROFILES.items():
            rotated = np.roll(profile, root)
            r = float(np.corrcoef(histogram, rotated)[0, 1])
            scores.append((r, root, mode))

    scores.sort(reverse=True)

    western_scores = [(r, rt, m) for r, rt, m in scores if not m.startswith("maqam_")]
    maqam_scores   = [(r, rt, m) for r, rt, m in scores if m.startswith("maqam_")]

    best_western_r = western_scores[0][0] if western_scores else -1.0
    best_maqam_r, best_maqam_root, best_maqam_mode = maqam_scores[0] if maqam_scores else (-1.0, 0, "")

    maqam_eligible = (
        n_notes >= KEY_MAQAM_MIN_NOTES
        and best_maqam_r - best_western_r >= KEY_MAQAM_MIN_GAP
    )

    if maqam_eligible:
        best_r, best_root, best_mode = best_maqam_r, best_maqam_root, best_maqam_mode
    else:
        best_r, best_root, best_mode = western_scores[0]

    candidates = []
    for r, root, mode in scores[:5]:
        rn = note_name(root, key_uses_flats(root, mode))
        if mode.startswith("maqam_"):
            label = f"{rn} {MAQAM_LABELS[mode]}"
        else:
            label = f"{rn} {'major' if mode == 'major' else 'minor'}"
        candidates.append({"key_str": label, "confidence": round(r, 4)})

    return best_root, best_mode, best_r, candidates


def _best_scale(histogram: np.ndarray, root: int, mode: str, instrument: str = "guitar") -> str:
    """Pick the scale whose notes account for the highest weighted coverage."""
    if mode.startswith("maqam_"):
        return mode   # maqam IS the scale — no further subdivision
    diatonic   = "major" if mode == "major" else "natural_minor"
    parent_key = "major" if mode == "major" else "natural_minor"
    candidates = [
        name for name, parent in SCALE_PARENT.items()
        if parent == parent_key
    ] + [diatonic]

    # First pass: raw scores (no bias)
    raw_scores: dict[str, float] = {}
    for name in candidates:
        pcs = set(_scale_pitch_classes(root, name))
        raw_scores[name] = sum(histogram[pc] for pc in pcs)

    best_raw    = max(raw_scores.values())
    penta_score = raw_scores.get("pentatonic_minor", -1.0)

    # Pentatonic minor bias: guitar-only. Electric guitar music strongly favours
    # pentatonic minor for solos/riffs even when the full minor scale fits equally
    # well. This bias should not apply to vocals, piano, or bass key detection.
    scores = dict(raw_scores)
    if (
        instrument == "guitar"
        and "pentatonic_minor" in scores
        and (best_raw - penta_score) <= KEY_PENTA_BIAS_GATE
    ):
        scores["pentatonic_minor"] += KEY_PENTATONIC_MINOR_BIAS

    best_name = max(scores, key=lambda n: scores[n])
    return best_name


def _scale_pitch_classes(root: int, scale_name: str) -> list[int]:
    intervals = SCALES[scale_name]
    return [(root + interval) % 12 for interval in intervals]


def get_scale_pitch_classes(root_midi: int, scale_name: str) -> list[int]:
    return _scale_pitch_classes(root_midi, scale_name)


def load_key_analysis() -> dict:
    path = os.path.join(get_shared_dir(), "05_key_analysis.json")
    with open(path) as f:
        return json.load(f)
