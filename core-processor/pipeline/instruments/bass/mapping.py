"""
Stage 6 (Bass): Bass Mapping
Input:  cleaned bass notes + key analysis
Output: mapped notes [{string, fret, pitch, start, duration, confidence}]
        saved to outputs/06_bass_mapped_notes.json

Standard 4-string bass tuning (string -> open MIDI pitch):
  String 4 (low E):  E1 = 28
  String 3 (A):      A1 = 33
  String 2 (D):      D2 = 38
  String 1 (G):      G2 = 43

Viterbi DP algorithm is identical to guitar; only the tuning and string count differ.
Bass has 24 frets (vs 21 on guitar) — allows notes up to G4 on the G string.
"""

import json
import os

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    MAPPING_HAND_SPAN,
    MAPPING_CHORD_GAP_S,
    MAPPING_SHIFT_COST_PER_FRET,
    MAPPING_OPEN_STRING_IN_KEY_BONUS,
    MAPPING_FRET_HEIGHT_WEIGHT,
    MAPPING_SAME_STRING_CHORD_PENALTY,
    MAPPING_LONG_NOTE_S,
    MAPPING_SHORT_NOTE_S,
    MAPPING_STRING_CHANGE_PENALTY_LONG,
    MAPPING_STRING_CHANGE_PENALTY_MEDIUM,
    MAPPING_STRING_CHANGE_PENALTY_SHORT,
    MAPPING_CONTEXT_WINDOW_S,
    MAPPING_CONTEXT_TOLERANCE,
    MAPPING_CONTEXT_PENALTY_PER_FRET,
    MAPPING_CONTEXT_OPEN_STRING_EXEMPT,
    MAPPING_BEND_SEMITONE_COST,
)

# ── Bass-specific fretboard constants ─────────────────────────────────────────
BASS_MAX_FRET   = 24   # most bass guitars have 24 frets
BASS_BEND_TOL   = 2    # semitones — bass bends are small (mainly slides)
BASS_HAND_SPAN  = 4    # same as guitar (one finger per fret)

BASS_TUNING = {
    4: 28,   # E1
    3: 33,   # A1
    2: 38,   # D2
    1: 43,   # G2
}

OPEN_STRING_PITCH_BASS = {v: k for k, v in BASS_TUNING.items()}

# String preference (lower = more preferred) — inner strings for runs
BASS_STRING_PREF = {3: 0, 2: 1, 4: 2, 1: 3}


def map_to_bass(
    cleaned_notes: list[dict],
    key_info: dict | None = None,
    save: bool = True,
) -> list[dict]:
    if key_info:
        print(f"[Stage 6B] Mapping {len(cleaned_notes)} notes  "
              f"(key: {key_info['key_str']})")
    else:
        print(f"[Stage 6B] Mapping {len(cleaned_notes)} notes")

    scale_pcs    = set(key_info["scale_pcs"]) if key_info else set()
    notes_sorted = sorted(cleaned_notes, key=lambda n: n["start"])

    mapped, unmapped = _viterbi_map_bass(notes_sorted, scale_pcs)

    if unmapped:
        print(f"[Stage 6B] Warning: {unmapped} notes could not be mapped (out of range)")

    print(f"[Stage 6B] Mapped {len(mapped)} notes  "
          f"(fret range: {_fret_range(mapped)})")

    if save:
        out_path = os.path.join(get_instrument_dir("bass"), "06_mapped_notes.json")
        with open(out_path, "w") as f:
            json.dump(mapped, f, indent=2)
        print(f"[Stage 6B] Saved -> {out_path}")

    return mapped


# ── Viterbi DP mapping ────────────────────────────────────────────────────────

def _viterbi_map_bass(
    notes: list[dict],
    scale_pcs: set,
) -> tuple[list[dict], int]:
    INF = float("inf")

    note_cands: list[list[tuple[int, int]]] = []
    valid_idx: list[int] = []
    unmapped = 0

    for i, note in enumerate(notes):
        cands = _all_positions_bass(note["pitch"])
        if cands:
            note_cands.append(cands)
            valid_idx.append(i)
        else:
            unmapped += 1

    if not note_cands:
        return [], unmapped

    n = len(note_cands)

    context_centers = _compute_context_centers_bass(notes, valid_idx)

    p0 = notes[valid_idx[0]]["pitch"]
    dp   = [_emission_cost_bass(s, f, scale_pcs, context_centers[0], p0) for s, f in note_cands[0]]
    back: list[list[int]] = [[0] * len(c) for c in note_cands]

    for i in range(1, n):
        note_cur  = notes[valid_idx[i]]
        note_prev = notes[valid_idx[i - 1]]
        cands_cur  = note_cands[i]
        cands_prev = note_cands[i - 1]

        gap_s = note_cur["start"] - note_prev["start"]

        new_dp: list[float] = []
        for j, (s2, f2) in enumerate(cands_cur):
            emit = _emission_cost_bass(s2, f2, scale_pcs, context_centers[i], note_cur["pitch"])
            best_cost = INF
            best_k    = 0
            for k, (s1, f1) in enumerate(cands_prev):
                cost = dp[k] + _transition_cost_bass(
                    s1, f1, s2, f2,
                    prev_dur=note_prev["duration"],
                    gap_s=gap_s,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_k    = k
            new_dp.append(best_cost + emit)
            back[i][j] = best_k

        dp = new_dp

    path = [0] * n
    path[-1] = int(min(range(len(dp)), key=lambda j: dp[j]))
    for i in range(n - 2, -1, -1):
        path[i] = back[i + 1][path[i + 1]]

    assigned: dict[int, tuple[int, int]] = {}
    for vi, orig_i in enumerate(valid_idx):
        s, f = note_cands[vi][path[vi]]
        assigned[orig_i] = (s, f)

    mapped = []
    for i, note in enumerate(notes):
        if i in assigned:
            s, f = assigned[i]
            mapped.append({
                "string":     s,
                "fret":       f,
                "pitch":      note["pitch"],
                "start":      note["start"],
                "duration":   note["duration"],
                "confidence": note.get("confidence", 0.5),
            })

    return mapped, unmapped


def _emission_cost_bass(
    s: int,
    f: int,
    scale_pcs: set,
    context_center: float | None = None,
    note_pitch: int | None = None,
) -> float:
    cost = float(BASS_STRING_PREF.get(s, 9))
    cost += f * MAPPING_FRET_HEIGHT_WEIGHT
    if f == 0 and BASS_TUNING[s] % 12 in scale_pcs:
        cost -= MAPPING_OPEN_STRING_IN_KEY_BONUS

    if note_pitch is not None:
        natural_pitch = BASS_TUNING[s] + f
        bend_st = note_pitch - natural_pitch
        if bend_st > 0:
            cost += bend_st * MAPPING_BEND_SEMITONE_COST

    if context_center is not None:
        if f > 0 or not MAPPING_CONTEXT_OPEN_STRING_EXEMPT:
            deviation = abs(f - context_center)
            cost += max(0.0, deviation - MAPPING_CONTEXT_TOLERANCE) * MAPPING_CONTEXT_PENALTY_PER_FRET

    return cost


def _transition_cost_bass(
    s1: int, f1: int,
    s2: int, f2: int,
    prev_dur: float,
    gap_s: float,
) -> float:
    cost = 0.0

    if f1 > 0 and f2 > 0:
        shift = max(0, abs(f1 - f2) - BASS_HAND_SPAN)
        cost += shift * MAPPING_SHIFT_COST_PER_FRET

    if gap_s < MAPPING_CHORD_GAP_S:
        if s1 == s2:
            cost += MAPPING_SAME_STRING_CHORD_PENALTY
    else:
        if s1 != s2:
            if prev_dur >= MAPPING_LONG_NOTE_S:
                cost += MAPPING_STRING_CHANGE_PENALTY_LONG
            elif prev_dur >= MAPPING_SHORT_NOTE_S:
                cost += MAPPING_STRING_CHANGE_PENALTY_MEDIUM
            else:
                cost += MAPPING_STRING_CHANGE_PENALTY_SHORT

    return cost


def _pitch_ref_fret_bass(midi_pitch: int) -> float:
    cands = _all_positions_bass(midi_pitch)
    if not cands:
        return 0.0
    return sum(f for _, f in cands) / len(cands)


def _compute_context_centers_bass(
    notes: list[dict],
    valid_idx: list[int],
) -> list[float]:
    import statistics

    ref_frets = [_pitch_ref_fret_bass(notes[orig_i]["pitch"]) for orig_i in valid_idx]
    times     = [notes[orig_i]["start"] for orig_i in valid_idx]

    centers: list[float] = []
    for i, t_i in enumerate(times):
        window_refs = [
            ref_frets[j]
            for j in range(len(valid_idx))
            if abs(times[j] - t_i) <= MAPPING_CONTEXT_WINDOW_S
        ]
        centers.append(statistics.median(window_refs))
    return centers


# ── Position helpers ──────────────────────────────────────────────────────────

def _all_positions_bass(midi_pitch: int) -> list[tuple[int, int]]:
    positions = []
    for string_num, open_pitch in BASS_TUNING.items():
        fret = midi_pitch - open_pitch
        if 0 <= fret <= BASS_MAX_FRET:
            positions.append((string_num, fret))
        elif BASS_MAX_FRET < fret <= BASS_MAX_FRET + BASS_BEND_TOL:
            positions.append((string_num, BASS_MAX_FRET))
    return positions


def _fret_range(mapped: list[dict]) -> str:
    if not mapped:
        return "n/a"
    frets = [n["fret"] for n in mapped]
    return f"{min(frets)}-{max(frets)}"


def load_mapped_notes_bass() -> list[dict]:
    path = os.path.join(get_instrument_dir("bass"), "06_mapped_notes.json")
    with open(path) as f:
        return json.load(f)
