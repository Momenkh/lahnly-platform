"""
Stage 6: Guitar Mapping
Input:  cleaned notes + key analysis
Output: mapped notes [{string, fret, pitch, start, duration, confidence}]
        saved to outputs/06_mapped_notes.json

Standard tuning (string -> open MIDI pitch):
  String 1 (high e): E4 = 64
  String 2 (B):      B3 = 59
  String 3 (G):      G3 = 55
  String 4 (D):      D3 = 50
  String 5 (A):      A2 = 45
  String 6 (low E):  E2 = 40

Quality improvements:
  - Viterbi DP replaces greedy sliding window: globally optimal (string, fret)
    assignment minimises total hand-shift + string-change cost over the whole
    note sequence, instead of making locally optimal per-note decisions.
  - Open-string preference: if a note matches an open string AND is in the key's
    scale, strongly prefer fret 0 on that string (most natural guitar playing).
  - Simultaneous-note safety: two chord notes on the same string receive a large
    penalty (physically impossible on a real guitar).
  - confidence is preserved from pitch extraction through to the mapped output.
"""

import json
import os

from pipeline.config import get_outputs_dir
from pipeline.settings import (
    MAPPING_MAX_FRET,
    MAPPING_HAND_SPAN,
    MAPPING_BEND_TOLERANCE,
    MAPPING_BEND_TOLERANCE_LEAD,
    MAPPING_LONG_NOTE_S,
    MAPPING_SHORT_NOTE_S,
    MAPPING_STRING_CHANGE_PENALTY_LONG,
    MAPPING_STRING_CHANGE_PENALTY_MEDIUM,
    MAPPING_STRING_CHANGE_PENALTY_SHORT,
    MAPPING_OPEN_STRING_IN_KEY_BONUS,
    MAPPING_SHIFT_COST_PER_FRET,
    MAPPING_CHORD_GAP_S,
    MAPPING_SAME_STRING_CHORD_PENALTY,
    MAPPING_FRET_HEIGHT_WEIGHT,
    MAPPING_CONTEXT_WINDOW_S,
    MAPPING_CONTEXT_TOLERANCE,
    MAPPING_CONTEXT_PENALTY_PER_FRET,
    MAPPING_CONTEXT_OPEN_STRING_EXEMPT,
    MAPPING_BEND_SEMITONE_COST,
    MELODY_MIN_PITCH,
)

STANDARD_TUNING = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}

# Open string pitches for quick lookup (pitch -> string number)
OPEN_STRING_PITCH = {v: k for k, v in STANDARD_TUNING.items()}

# String preference (lower = more preferred) — middle strings for runs
STRING_PREF = {4: 0, 3: 1, 5: 2, 2: 3, 6: 4, 1: 5}


def map_to_guitar(
    cleaned_notes: list[dict],
    key_info: dict | None = None,
    guitar_type: str = "rhythm",
    save: bool = True,
) -> list[dict]:
    if key_info:
        print(f"[Stage 6] Mapping {len(cleaned_notes)} notes  "
              f"(key: {key_info['key_str']})")
    else:
        print(f"[Stage 6] Mapping {len(cleaned_notes)} notes")

    bend_tol  = MAPPING_BEND_TOLERANCE_LEAD if guitar_type == "lead" else MAPPING_BEND_TOLERANCE
    scale_pcs = set(key_info["scale_pcs"]) if key_info else set()
    notes_sorted = sorted(cleaned_notes, key=lambda n: n["start"])

    mapped, unmapped = _viterbi_map(notes_sorted, scale_pcs, bend_tol)

    if unmapped:
        print(f"[Stage 6] Warning: {unmapped} notes could not be mapped (out of range)")

    print(f"[Stage 6] Mapped {len(mapped)} notes  "
          f"(fret range: {_fret_range(mapped)})")

    if save:
        os.makedirs(get_outputs_dir(), exist_ok=True)
        out_path = os.path.join(get_outputs_dir(), "06_mapped_notes.json")
        with open(out_path, "w") as f:
            json.dump(mapped, f, indent=2)
        print(f"[Stage 6] Saved -> {out_path}")

    return mapped


# ── Viterbi DP mapping ────────────────────────────────────────────────────────

def _viterbi_map(
    notes: list[dict],
    scale_pcs: set,
    bend_tol: int,
) -> tuple[list[dict], int]:
    """
    Globally optimal (string, fret) assignment via Viterbi DP.

    State per note: one of its valid (string, fret) candidate positions.
    Total cost = sum of emission + transition costs over the whole sequence.

    Emission  — intrinsic desirability of a position (string preference,
                open-string-in-key bonus).
    Transition — hand shift (frets beyond span) + string-change penalty.
                 Simultaneous notes (gap < MAPPING_CHORD_GAP_S) get a large
                 penalty for sharing a string — physically impossible on guitar.

    Returns (mapped_notes, n_unmapped).
    """
    INF = float("inf")

    # Build candidate list; skip notes with no valid position
    note_cands: list[list[tuple[int, int]]] = []
    valid_idx: list[int] = []
    unmapped = 0

    for i, note in enumerate(notes):
        cands = _all_positions(note["pitch"], bend_tol)
        if cands:
            note_cands.append(cands)
            valid_idx.append(i)
        else:
            unmapped += 1

    if not note_cands:
        return [], unmapped

    n = len(note_cands)

    # ── Context-aware fret window precomputation ──────────────────────────────
    # Compute the median expected fret for the surrounding passage before the
    # forward pass, so each emission cost can be biased toward the local position.
    context_centers = _compute_context_centers(notes, valid_idx, bend_tol)

    # ── Forward pass ──────────────────────────────────────────────────────────
    p0 = notes[valid_idx[0]]["pitch"]
    dp   = [_emission_cost(s, f, scale_pcs, context_centers[0], p0) for s, f in note_cands[0]]
    back: list[list[int]] = [[0] * len(c) for c in note_cands]

    for i in range(1, n):
        note_cur  = notes[valid_idx[i]]
        note_prev = notes[valid_idx[i - 1]]
        cands_cur  = note_cands[i]
        cands_prev = note_cands[i - 1]

        # Start-to-start gap: notes starting within MAPPING_CHORD_GAP_S of each
        # other are treated as a simultaneous strum (chord), not sequential notes.
        gap_s = note_cur["start"] - note_prev["start"]

        new_dp: list[float] = []
        for j, (s2, f2) in enumerate(cands_cur):
            emit = _emission_cost(s2, f2, scale_pcs, context_centers[i], note_cur["pitch"])
            best_cost = INF
            best_k    = 0
            for k, (s1, f1) in enumerate(cands_prev):
                cost = dp[k] + _transition_cost(
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

    # ── Backtrack ─────────────────────────────────────────────────────────────
    path = [0] * n
    path[-1] = int(min(range(len(dp)), key=lambda j: dp[j]))
    for i in range(n - 2, -1, -1):
        path[i] = back[i + 1][path[i + 1]]

    # ── Build output in original note order ───────────────────────────────────
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


def _emission_cost(
    s: int,
    f: int,
    scale_pcs: set,
    context_center: float | None = None,
    note_pitch: int | None = None,
) -> float:
    """
    Intrinsic desirability of playing at (string, fret). Lower = preferred.

    Components:
      - String preference: middle strings slightly preferred for runs.
      - Fret height penalty: playing high on the neck is harder; steers the mapper
        toward lower frets without hard-coding string choices.
      - Open string in-key bonus: subtracted when fret=0 AND the open pitch is
        diatonic in the detected key (most natural guitar voicing).
      - Bend penalty: if note_pitch > natural pitch at (string, fret), each semitone
        of required bend adds MAPPING_BEND_SEMITONE_COST.  Prevents the mapper from
        choosing a clamped high-fret position on a middle string (e.g., G-string
        fret 21 bent +6 semitones) over the natural high-e position.
      - Context fret penalty (optional): adds a soft cost for fret assignments that
        deviate more than MAPPING_CONTEXT_TOLERANCE frets from the passage's median
        expected position.  Open strings are exempt when
        MAPPING_CONTEXT_OPEN_STRING_EXEMPT is True.
    """
    cost = float(STRING_PREF.get(s, 9))
    cost += f * MAPPING_FRET_HEIGHT_WEIGHT
    if f == 0 and STANDARD_TUNING[s] % 12 in scale_pcs:
        cost -= MAPPING_OPEN_STRING_IN_KEY_BONUS

    if note_pitch is not None:
        natural_pitch = STANDARD_TUNING[s] + f
        bend_st = note_pitch - natural_pitch
        if bend_st > 0:
            cost += bend_st * MAPPING_BEND_SEMITONE_COST

    if context_center is not None:
        if f > 0 or not MAPPING_CONTEXT_OPEN_STRING_EXEMPT:
            deviation = abs(f - context_center)
            cost += max(0.0, deviation - MAPPING_CONTEXT_TOLERANCE) * MAPPING_CONTEXT_PENALTY_PER_FRET

    return cost


def _transition_cost(
    s1: int, f1: int,
    s2: int, f2: int,
    prev_dur: float,
    gap_s: float,
) -> float:
    """
    Cost of moving from (s1, f1) to (s2, f2).

    Simultaneous notes (gap_s < MAPPING_CHORD_GAP_S):
      - Same-string assignment is penalised heavily (physically impossible).
      - Hand shift still applies.

    Sequential notes (gap_s >= MAPPING_CHORD_GAP_S):
      - String change penalised proportional to previous note duration.
      - Hand shift applies for any fret distance beyond the hand span.

    Open strings (fret 0) don't anchor the hand, so shifts from/to fret 0
    are not penalised.
    """
    cost = 0.0

    # Hand shift: cost per fret beyond the reachable span
    if f1 > 0 and f2 > 0:
        shift = max(0, abs(f1 - f2) - MAPPING_HAND_SPAN)
        cost += shift * MAPPING_SHIFT_COST_PER_FRET

    if gap_s < MAPPING_CHORD_GAP_S:
        # Chord / simultaneous — same string is physically impossible
        if s1 == s2:
            cost += MAPPING_SAME_STRING_CHORD_PENALTY
    else:
        # Sequential — penalise string changes scaled by previous duration
        if s1 != s2:
            if prev_dur >= MAPPING_LONG_NOTE_S:
                cost += MAPPING_STRING_CHANGE_PENALTY_LONG
            elif prev_dur >= MAPPING_SHORT_NOTE_S:
                cost += MAPPING_STRING_CHANGE_PENALTY_MEDIUM
            else:
                cost += MAPPING_STRING_CHANGE_PENALTY_SHORT

    return cost


# ── Context-aware fret window helpers ────────────────────────────────────────

def _pitch_ref_fret(midi_pitch: int, bend_tol: int) -> float:
    """
    Pitch-based reference fret: mean fret across all valid (string, fret) candidates.

    This is the best pitch-only proxy for "what fret height does this note imply?":
      - Monotonically increasing with pitch (higher pitch → higher reference fret)
      - String-agnostic (no single-string projection bias)
      - Computable before any DP assignment decisions

    Used exclusively by _compute_context_centers.
    Returns 0.0 if the pitch has no valid positions (should not occur in practice).
    """
    cands = _all_positions(midi_pitch, bend_tol)
    if not cands:
        return 0.0
    return sum(f for _, f in cands) / len(cands)


def _compute_context_centers(
    notes: list[dict],
    valid_idx: list[int],
    bend_tol: int,
) -> list[float]:
    """
    Precompute the context fret center for each valid note.

    For each note i in valid_idx, gather all other valid notes j whose start
    time falls within ±MAPPING_CONTEXT_WINDOW_S seconds of note i's start,
    and return the median of their reference frets.

    The note itself is always included, so the result is never empty.
    Returns a list of length len(valid_idx), one center per valid note.
    """
    import statistics

    ref_frets = [
        _pitch_ref_fret(notes[orig_i]["pitch"], bend_tol)
        for orig_i in valid_idx
    ]
    times = [notes[orig_i]["start"] for orig_i in valid_idx]

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

def _all_positions(midi_pitch: int, bend_tolerance: int = MAPPING_BEND_TOLERANCE) -> list[tuple[int, int]]:
    """
    All valid (string, fret) positions for a MIDI pitch.
    Pitches up to bend_tolerance semitones above MAX_FRET are clamped to fret 21.
    """
    positions = []
    for string_num, open_pitch in STANDARD_TUNING.items():
        fret = midi_pitch - open_pitch
        if 0 <= fret <= MAPPING_MAX_FRET:
            positions.append((string_num, fret))
        elif MAPPING_MAX_FRET < fret <= MAPPING_MAX_FRET + bend_tolerance:
            positions.append((string_num, MAPPING_MAX_FRET))
    return positions


def _fret_range(mapped: list[dict]) -> str:
    if not mapped:
        return "n/a"
    frets = [n["fret"] for n in mapped]
    return f"{min(frets)}-{max(frets)}"


def isolate_melody(
    mapped_notes: list[dict],
    min_pitch: int = 0,
) -> tuple[list[dict], list[dict]]:
    """
    Split mapped notes into (melody_notes, harmony_notes).

    Melody = the top voice: the highest-pitched note active at each instant.
    When multiple notes overlap in time, only the highest-pitch one is melody;
    the rest are harmony (accompaniment / chord tones).
    Non-overlapping notes are always melody, UNLESS their pitch is below min_pitch.

    min_pitch: notes below this MIDI pitch go to harmony regardless of overlap.
               Use MELODY_MIN_PITCH[guitar_type] from settings to prevent
               low-string / chord-harmonic bleed into the melody track.

    Returns (melody_notes, harmony_notes).
    """
    if not mapped_notes:
        return [], []

    sorted_notes = sorted(mapped_notes, key=lambda n: (n["start"], -n["pitch"]))
    n = len(sorted_notes)
    is_melody = [False] * n

    for i, note in enumerate(sorted_notes):
        t_start = note["start"]
        t_end   = t_start + note["duration"]
        pitch_i = note["pitch"]

        # Notes below min_pitch floor go straight to harmony (e.g. bass/low-E bleed)
        if pitch_i < min_pitch:
            is_melody[i] = False
            continue

        # Check if any other note overlaps this one AND has a higher pitch
        dominated = False
        for j, other in enumerate(sorted_notes):
            if j == i:
                continue
            o_start = other["start"]
            o_end   = o_start + other["duration"]
            if o_start < t_end and o_end > t_start:
                if other["pitch"] > pitch_i:
                    dominated = True
                    break
                elif other["pitch"] == pitch_i and other["confidence"] > note["confidence"]:
                    dominated = True
                    break

        is_melody[i] = not dominated

    melody  = [n for n, m in zip(sorted_notes, is_melody) if m]
    harmony = [n for n, m in zip(sorted_notes, is_melody) if not m]

    print(f"[Melody] Isolated {len(melody)} melody notes, {len(harmony)} harmony notes")
    return melody, harmony


def get_all_positions(midi_pitch: int, guitar_type: str = "rhythm") -> list[tuple[int, int]]:
    bend_tol = MAPPING_BEND_TOLERANCE_LEAD if guitar_type == "lead" else MAPPING_BEND_TOLERANCE
    return _all_positions(midi_pitch, bend_tol)


def load_mapped_notes() -> list[dict]:
    path = os.path.join(get_outputs_dir(), "06_mapped_notes.json")
    with open(path) as f:
        return json.load(f)
