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
    CAPO_SEARCH_MAX, CAPO_MIN_FRIENDLY_STRINGS,
    KEY_SEGMENT_MIN_GAP_S, KEY_SEGMENT_MIN_NOTES, KEY_SEGMENT_MIN_DURATION_S,
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

    capo_fret = detect_capo(root, mode) if instrument == "guitar" else 0

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
        "capo_fret":  capo_fret,
    }

    print(f"[Stage 5] Detected key: {key_str}  (confidence {confidence:.2f})")
    print(f"[Stage 5] Top candidates: {[c['key_str'] for c in candidates[:3]]}")
    print(f"[Stage 5] Scale notes: {[CHROMATIC[pc] for pc in scale_pcs]}")
    if capo_fret:
        print(f"[Stage 5] Capo inferred: fret {capo_fret}  "
              f"(key becomes guitar-friendly after transposing down {capo_fret} semitone(s))")

    if save:
        out_path = os.path.join(get_shared_dir(), "05_key_analysis.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Stage 5] Saved -> {out_path}")

    return result


def analyze_key_segmented(
    notes: list[dict],
    instrument: str = "guitar",
) -> tuple[dict, list[dict]]:
    """
    Per-section key detection.

    Splits the note stream at silence gaps >= KEY_SEGMENT_MIN_GAP_S, then detects
    the key independently per segment.  Returns the global key (full-song histogram)
    alongside a list of per-segment key dicts for use in the stage 5b filter.

    Each segment dict has the same structure as analyze_key() plus:
      "seg_start": float  — earliest note start time in the segment
      "seg_end":   float  — latest note end time in the segment

    Segments with fewer than KEY_SEGMENT_MIN_NOTES notes inherit the key from the
    nearest reliable segment (the global key is the final fallback).

    The global key_info is returned as the first element and should be used for:
      - Tab header / key_str display
      - Guitar mapping open-string bonuses (single Viterbi pass for the whole song)
      - Capo detection

    The segments list is used only by apply_key_confidence_filter_segmented().
    """
    # Global key — always run first; also populates capo_fret
    global_key = analyze_key(notes, save=True, instrument=instrument)

    # ── Build time-based segments ──────────────────────────────────────────────
    if not notes:
        return global_key, []

    sorted_notes = sorted(notes, key=lambda n: n["start"])
    # Find gap-based split points
    split_times: list[float] = []
    for i in range(1, len(sorted_notes)):
        prev_end = sorted_notes[i - 1]["start"] + sorted_notes[i - 1]["duration"]
        gap      = sorted_notes[i]["start"] - prev_end
        if gap >= KEY_SEGMENT_MIN_GAP_S:
            split_times.append(sorted_notes[i]["start"])

    # Build segment note lists from split points
    segs_notes: list[list[dict]] = []
    seg_start_times: list[float] = []
    prev_split = 0.0
    seg_buf: list[dict] = []
    split_idx = 0

    for note in sorted_notes:
        if split_idx < len(split_times) and note["start"] >= split_times[split_idx]:
            if seg_buf:
                segs_notes.append(seg_buf)
                seg_start_times.append(seg_buf[0]["start"])
                seg_buf = []
            split_idx += 1
        seg_buf.append(note)
    if seg_buf:
        segs_notes.append(seg_buf)
        seg_start_times.append(seg_buf[0]["start"])

    # Merge short segments (duration < MIN_DURATION_S) into a neighbor
    def _seg_duration(notes_list: list[dict]) -> float:
        if not notes_list:
            return 0.0
        end = max(n["start"] + n["duration"] for n in notes_list)
        return end - notes_list[0]["start"]

    merged: list[list[dict]] = []
    for seg in segs_notes:
        if merged and _seg_duration(seg) < KEY_SEGMENT_MIN_DURATION_S:
            merged[-1] = merged[-1] + seg  # absorb into previous
        else:
            merged.append(seg)

    # If we ended up with only one segment (no meaningful gap found), return early
    if len(merged) <= 1:
        return global_key, []

    # ── Per-segment key detection ──────────────────────────────────────────────
    segments: list[dict] = []
    reliable_keys: list[dict] = []  # segments with enough notes

    for seg_notes in merged:
        seg_s = seg_notes[0]["start"]
        seg_e = max(n["start"] + n["duration"] for n in seg_notes)

        if len(seg_notes) >= KEY_SEGMENT_MIN_NOTES:
            hist = _build_histogram(seg_notes)
            root, mode, confidence, candidates = _detect_key(hist, len(seg_notes))
            is_maqam   = mode.startswith("maqam_")
            scale_name = mode if is_maqam else _best_scale(hist, root, mode, instrument=instrument)
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

            seg_key = {
                "root": root_name, "root_midi": root, "mode": mode,
                "use_flats": use_flats, "scale": scale_name,
                "scale_pcs": scale_pcs, "key_str": key_str,
                "confidence": round(float(confidence), 4),
                "seg_start": seg_s, "seg_end": seg_e,
            }
            reliable_keys.append(seg_key)
        else:
            seg_key = {"seg_start": seg_s, "seg_end": seg_e, "inherited": True}

        segments.append(seg_key)

    # Fill unreliable segments with the nearest reliable neighbour
    fallback = global_key  # final fallback if no reliable segments at all
    for i, seg in enumerate(segments):
        if seg.get("inherited"):
            nearest = _nearest_reliable(i, segments, reliable_keys, fallback)
            seg.update({k: v for k, v in nearest.items() if k not in ("seg_start", "seg_end")})
            seg.pop("inherited", None)

    # Log detected segments
    print(f"[Stage 5] Segmented key analysis: {len(segments)} section(s)")
    for s in segments:
        print(f"[Stage 5]   {s['seg_start']:.1f}s - {s['seg_end']:.1f}s  ->  {s['key_str']}")

    return global_key, segments


def _nearest_reliable(
    idx: int,
    segments: list[dict],
    reliable: list[dict],
    fallback: dict,
) -> dict:
    """Return the nearest segment that has a proper key detection."""
    if not reliable:
        return fallback
    # Find the reliable segment with smallest start-time distance
    mid = segments[idx]["seg_start"]
    return min(reliable, key=lambda s: abs(s["seg_start"] - mid))


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


# Open string pitch classes for standard guitar tuning: E A D G B
_OPEN_STRING_PCS: frozenset[int] = frozenset({4, 9, 2, 7, 11})


def detect_capo(root_midi: int, mode: str) -> int:
    """
    Infer capo position (1–CAPO_SEARCH_MAX) for guitar.

    A capo on fret N shifts all open strings up by N semitones, making the
    guitarist's chord shapes N semitones lower than the concert pitch.
    If the detected key has few open strings in its scale (guitar-unfriendly),
    we test each capo position to see if transposing the root down by N gives
    a friendlier key.

    Returns the lowest capo fret where the key scores >= CAPO_MIN_FRIENDLY_STRINGS
    open string pitch classes AND scores strictly higher than the detected key.
    Returns 0 (no capo) if no improvement is found or the key is already friendly.

    Only applies to Western major/minor keys — maqam songs are not capo-annotated.
    """
    if mode.startswith("maqam_"):
        return 0

    diatonic = "major" if mode == "major" else "natural_minor"

    def _open_string_score(root: int) -> int:
        pcs = set(_scale_pitch_classes(root, diatonic))
        return sum(1 for pc in _OPEN_STRING_PCS if pc in pcs)

    base_score = _open_string_score(root_midi)
    if base_score >= CAPO_MIN_FRIENDLY_STRINGS:
        return 0  # already guitar-friendly — no capo needed

    best_fret  = 0
    best_score = base_score
    for capo in range(1, CAPO_SEARCH_MAX + 1):
        transposed_root = (root_midi - capo) % 12
        score = _open_string_score(transposed_root)
        if score > best_score:
            best_score = score
            best_fret  = capo
            if score == len(_OPEN_STRING_PCS):
                break  # perfect score — no need to test higher frets

    return best_fret if best_score >= CAPO_MIN_FRIENDLY_STRINGS else 0


def get_scale_pitch_classes(root_midi: int, scale_name: str) -> list[int]:
    return _scale_pitch_classes(root_midi, scale_name)


def load_key_analysis() -> dict:
    path = os.path.join(get_shared_dir(), "05_key_analysis.json")
    with open(path) as f:
        return json.load(f)
