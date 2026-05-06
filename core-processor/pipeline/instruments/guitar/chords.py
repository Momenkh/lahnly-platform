"""
Stage 7: Chord Detection
Input:  mapped notes [{string, fret, pitch, start, duration}]
Output: (solo_notes, chord_groups)
        solo_notes   — notes not part of any chord (< 3 simultaneous)
        chord_groups — [{notes, chord_name, start, duration}]

Saved to outputs/07_chords.json

Quality improvements:
  - Tempo-scaled strum tolerance (20% of a quarter note, capped at 60ms)
    avoids grouping notes that aren't strummed together at fast tempos
  - Key-aware chord naming: when two templates score equally, prefer the
    chord diatonic to the detected key
  - More templates: dim7, half-dim (min7b5), add9, min9, dom9
  - Penalty scoring: matches - 0.3 × extra pitch classes → less over-naming
"""

import json
import os

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    CHORD_DEFAULT_STRUM_S,
    CHORD_STRUM_BEAT_FRACTION,
    CHORD_MIN_NOTES,
    CHORD_MIN_UNIQUE_PCS,
    CHORD_MIN_DURATION_S,
    CHORD_EXTRA_PC_PENALTY,
    CHORD_POWER_INTERVALS,
)

CHROMATIC      = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHROMATIC_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

CHORD_TEMPLATES = {
    "maj":      [0, 4, 7],
    "min":      [0, 3, 7],
    "dim":      [0, 3, 6],
    "aug":      [0, 4, 8],
    "5":        [0, 7],
    "7":        [0, 4, 7, 10],
    "maj7":     [0, 4, 7, 11],
    "min7":     [0, 3, 7, 10],
    "dim7":     [0, 3, 6, 9],
    "hdim7":    [0, 3, 6, 10],   # half-diminished / min7b5
    "sus2":     [0, 2, 7],
    "sus4":     [0, 5, 7],
    "add9":     [0, 2, 4, 7],    # major + 9th
    "min9":     [0, 3, 7, 10, 2],
    "9":        [0, 4, 7, 10, 2],
}


def detect_chords(
    mapped_notes: list[dict],
    tempo_info: dict | None = None,
    key_info: dict | None = None,
    save: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (solo_notes, chord_groups).
    chord_groups items: {notes, chord_name, start, duration}
    """
    if not mapped_notes:
        return [], []

    # Tempo-scaled strum tolerance: fraction of a beat, capped at default
    if tempo_info and tempo_info.get("beat_s"):
        strum_s = min(CHORD_DEFAULT_STRUM_S, tempo_info["beat_s"] * CHORD_STRUM_BEAT_FRACTION)
    else:
        strum_s = CHORD_DEFAULT_STRUM_S

    scale_pcs = set(key_info["scale_pcs"]) if key_info else set()
    use_flats = bool(key_info.get("use_flats", False)) if key_info else False

    print(f"[Stage 7] Detecting chords in {len(mapped_notes)} mapped notes "
          f"(strum window: {strum_s*1000:.0f}ms)...")

    sorted_notes = sorted(mapped_notes, key=lambda n: n["start"])
    groups = _group_simultaneous(sorted_notes, strum_s)

    solo_notes   = []
    chord_groups = []

    for group in groups:
        unique_pcs = set(n["pitch"] % 12 for n in group)
        duration = (max(n["start"] + n["duration"] for n in group)
                    - min(n["start"] for n in group))

        is_chord = False

        if len(unique_pcs) >= CHORD_MIN_UNIQUE_PCS and duration >= CHORD_MIN_DURATION_S:
            if len(group) >= CHORD_MIN_NOTES:
                # Standard chord: 3+ simultaneous notes
                is_chord = True
            elif len(group) == 2 and len(unique_pcs) == 2:
                # 2-note group: only qualify as a power chord if the
                # pitch-class interval is a perfect 5th or perfect 4th.
                pcs_sorted = sorted(unique_pcs)
                interval   = (pcs_sorted[1] - pcs_sorted[0]) % 12
                if interval in CHORD_POWER_INTERVALS:
                    is_chord = True

        if is_chord:
            t_start = min(n["start"] for n in group)
            t_end   = max(n["start"] + n["duration"] for n in group)
            name    = _name_chord(unique_pcs, scale_pcs, use_flats)
            chord_groups.append({
                "notes":      group,
                "chord_name": name,
                "start":      round(t_start, 4),
                "duration":   round(t_end - t_start, 4),
            })
        else:
            solo_notes.extend(group)

    print(
        f"[Stage 7] Found {len(chord_groups)} chords, "
        f"{len(solo_notes)} solo notes"
    )
    if chord_groups:
        names = [c["chord_name"] for c in chord_groups[:8]]
        print(f"[Stage 7] Chord names (first 8): {names}")

    if save:
        _save(solo_notes, chord_groups)

    return solo_notes, chord_groups


def get_chord_tone_pcs(
    notes: list[dict],
    tempo_info: dict | None = None,
) -> set[int]:
    """
    Lightweight pre-pass to collect pitch classes that appear as chord tones.

    Runs the same grouping + quality gates as detect_chords but on cleaned
    (un-mapped) notes, so it can be called before Stage 6. Returns the set of
    pitch classes (0-11) found in any qualifying chord group.

    Used by apply_key_confidence_filter to protect legitimate chromatic chord
    tones (e.g. the b7 in a V7 chord) from being deleted as "off-key".
    """
    if not notes:
        return set()

    if tempo_info and tempo_info.get("beat_s"):
        strum_s = min(CHORD_DEFAULT_STRUM_S, tempo_info["beat_s"] * CHORD_STRUM_BEAT_FRACTION)
    else:
        strum_s = CHORD_DEFAULT_STRUM_S

    sorted_notes = sorted(notes, key=lambda n: n["start"])
    groups       = _group_simultaneous(sorted_notes, strum_s)

    protected: set[int] = set()
    for group in groups:
        unique_pcs = set(n["pitch"] % 12 for n in group)
        duration   = (max(n["start"] + n["duration"] for n in group)
                      - min(n["start"] for n in group))

        qualifies = False
        if len(unique_pcs) >= CHORD_MIN_UNIQUE_PCS and duration >= CHORD_MIN_DURATION_S:
            if len(group) >= CHORD_MIN_NOTES:
                qualifies = True
            elif len(group) == 2 and len(unique_pcs) == 2:
                pcs_sorted = sorted(unique_pcs)
                interval   = (pcs_sorted[1] - pcs_sorted[0]) % 12
                if interval in CHORD_POWER_INTERVALS:
                    qualifies = True

        if qualifies:
            protected |= unique_pcs

    return protected


def load_chord_detection() -> tuple[list[dict], list[dict]]:
    path = os.path.join(get_instrument_dir("guitar"), "07_chords.json")
    with open(path) as f:
        data = json.load(f)
    return data["solo_notes"], data["chord_groups"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _group_simultaneous(sorted_notes: list[dict], strum_s: float) -> list[list[dict]]:
    """
    Group notes that start within strum_s of each other.

    Rolling anchor: the window reference updates to each new note added to the
    group, so a slow strum (e.g. 6 strings across 90ms) stays in one group even
    when consecutive inter-note gaps are each small but cumulative gap is wide.
    """
    if not sorted_notes:
        return []

    groups = []
    current_group = [sorted_notes[0]]
    last_start    = sorted_notes[0]["start"]

    for note in sorted_notes[1:]:
        if note["start"] - last_start <= strum_s:
            current_group.append(note)
            last_start = note["start"]   # rolling anchor
        else:
            groups.append(current_group)
            current_group = [note]
            last_start    = note["start"]

    groups.append(current_group)
    return groups


def _name_chord(pcs: set, scale_pcs: set, use_flats: bool = False) -> str:
    """
    Name a chord from its pitch class set.
    - 2 unique PCs: detect power chord, 3rd, etc.
    - 3+ unique PCs: template matching with penalty for extras.
    Key-aware tie-breaking: prefer root+quality diatonic to the key.
    Returns e.g. "Am", "G7", "Eb5", "Db", or "?" if nothing fits.
    """
    names = CHROMATIC_FLAT if use_flats else CHROMATIC

    if len(pcs) == 2:
        pc_list  = sorted(pcs)
        lo, hi   = pc_list
        interval = (hi - lo) % 12
        INTERVAL_MAP = {
            3: ("min", 0),   # minor 3rd   -> lower note is root
            4: ("",    0),   # major 3rd   -> lower note is root
            5: ("5",   1),   # perfect 4th -> upper note is root (inv. 5th)
            7: ("5",   0),   # perfect 5th -> lower note is root
            8: ("",    1),   # minor 6th   -> upper note is root (inv. maj 3rd)
            9: ("min", 1),   # major 6th   -> upper note is root (inv. min 3rd)
        }
        if interval in INTERVAL_MAP:
            suffix, root_idx = INTERVAL_MAP[interval]
            root = pc_list[root_idx]
            label = "min" if suffix == "min" else suffix
            name = names[root] + label
            return name if suffix else names[root]
        return "?"

    # 3+ pitch classes — try all roots and templates
    best_score  = -999.0
    best_name   = "?"
    best_diat   = False   # is current best diatonic to key?

    for root in range(12):
        for quality, intervals in CHORD_TEMPLATES.items():
            template_pcs = set((root + i) % 12 for i in intervals)
            matches = len(template_pcs & pcs)
            extras  = len(pcs - template_pcs)
            if matches < 3:
                continue
            score = matches - CHORD_EXTRA_PC_PENALTY * extras

            root_name  = names[root]
            chord_name = root_name if quality == "maj" else f"{root_name}{quality}"

            # Key-aware tie-breaking
            template_roots_diatonic = all(
                pc in scale_pcs for pc in template_pcs if pc in pcs
            )
            diatonic = bool(scale_pcs) and template_roots_diatonic

            # Prefer higher score; break ties by diatonicity
            if (score > best_score) or (score == best_score and diatonic and not best_diat):
                best_score = score
                best_name  = chord_name
                best_diat  = diatonic

    return best_name


def _save(solo_notes: list[dict], chord_groups: list[dict]) -> None:
    out_path = os.path.join(get_instrument_dir("guitar"), "07_chords.json")
    with open(out_path, "w") as f:
        json.dump({"solo_notes": solo_notes, "chord_groups": chord_groups}, f, indent=2)
    print(f"[Stage 7] Saved -> {out_path}")
