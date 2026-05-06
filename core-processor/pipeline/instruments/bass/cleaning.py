"""
Stage 3 (Bass): Note Cleaning
Input:  raw bass notes [{pitch, start, duration, confidence}]
Output: cleaned notes, saved to outputs/03_bass_cleaned_notes.json

Key differences from guitar cleaning:
  - No "bass bleed gate" — we ARE processing the bass stem, so low-frequency
    notes are the target, not bleed to be removed.
  - Lower polyphony limit (bass is mostly monophonic: 1-2 notes).
  - Wider merge gaps — bass notes sustain longer with detection dips.
  - Bend merging always applied (slides are common on bass).
"""

import json
import os

from pipeline.config import get_instrument_dir, get_shared_dir
from pipeline.settings import (
    BASS_MIDI_MIN,
    BASS_MIDI_MAX,
    BASS_CLEANING_TYPE_PARAMS,
    BASS_DEFAULT_MERGE_GAP_S,
    BASS_MERGE_BEAT_DIVISOR,
    BASS_MERGE_GAP_FLOOR_S,
    BASS_MERGE_GAP_CEILING_S,
    BASS_BEND_MAX_SEMITONES,
    BASS_BEND_MAX_GAP_S,
    BASS_KEY_CONFIDENCE_CUTOFF,
    BASS_BPM_MIN_DUR_CLAMP_MIN_S,
    BASS_BPM_MIN_DUR_CLAMP_MAX_S,
    BASS_LOCAL_PITCH_WINDOW_S,
    BASS_LOCAL_PITCH_MIN_CONTEXT,
    BASS_LOCAL_PITCH_MAX_DEV,
    BASS_OCTAVE_CONF_GATE,
    BASS_OCTAVE_WINDOW_S,
    BASS_OCTAVE_WIDE_WINDOW_S,
    BASS_OCTAVE_MIN_NEIGHBORS,
    BASS_OCTAVE_SHIFT_THRESHOLD,
    BASS_STEM_GATE_ENABLED,
    BASS_STEM_GATE_WINDOW,
    BASS_STEM_GATE_THRESH,
    BASS_ISOLATION_ENABLED,
    BASS_ISOLATION_WINDOW_S,
    BASS_ISOLATION_MIN_TOTAL,
    BASS_HIGH_PITCH_FLOOR_1,
    BASS_HIGH_PITCH_REDUCTION_1,
    BASS_HIGH_PITCH_FLOOR_2,
    BASS_HIGH_PITCH_REDUCTION_2,
    BASS_HIGH_PITCH_MIN_DUR_SCALE,
)


def clean_notes_bass(
    raw_notes: list[dict],
    bass_style: str = "bass_fingered",
    save: bool = True,
) -> list[dict]:
    """
    Clean raw pitch-detection output into stable, distinct bass notes.
    Loads stem_confidence from Stage 1 metadata to tune thresholds.
    bass_style: "bass_fingered" | "bass_picked" | "bass_slap"
    """
    from pipeline.shared.separation import load_stem_meta_for, gate_notes_by_stem_energy_for

    if bass_style not in BASS_CLEANING_TYPE_PARAMS:
        print(f"[Stage 3B] Unknown style '{bass_style}' — defaulting to 'bass_fingered'")
        bass_style = "bass_fingered"

    min_dur_fallback, conf_floor, conf_stem_scale, max_poly, bpm_subdiv, merge_ratio = (
        BASS_CLEANING_TYPE_PARAMS[bass_style]
    )

    stem_meta = load_stem_meta_for("bass")
    stem_conf = stem_meta.get("stem_confidence", 0.5)
    bpm       = _load_bpm()

    # Adaptive confidence threshold
    conf_thresh = round(conf_floor + (1.0 - stem_conf) * conf_stem_scale, 3)

    # Adaptive merge gap: half a 32nd note at detected tempo
    if bpm:
        beat_s    = 60.0 / bpm
        merge_gap = round(max(BASS_MERGE_GAP_FLOOR_S,
                              min(BASS_MERGE_GAP_CEILING_S, beat_s / BASS_MERGE_BEAT_DIVISOR)), 4)
    else:
        merge_gap = BASS_DEFAULT_MERGE_GAP_S

    # Adaptive min duration
    if bpm:
        bpm_min_dur = 60.0 / (bpm * bpm_subdiv)
        min_dur = round(
            max(BASS_BPM_MIN_DUR_CLAMP_MIN_S,
                min(BASS_BPM_MIN_DUR_CLAMP_MAX_S, bpm_min_dur)),
            4,
        )
    else:
        min_dur = min_dur_fallback

    print(f"[Stage 3B] Cleaning {len(raw_notes)} raw notes  (style: {bass_style})")
    print(f"[Stage 3B] stem_conf={stem_conf:.2f}  conf_thresh={conf_thresh}  "
          f"merge_gap={merge_gap*1000:.0f}ms  max_poly={max_poly}  min_dur={min_dur*1000:.0f}ms"
          + (f"  bpm={bpm:.1f}" if bpm else "  bpm=unknown"))

    notes = list(raw_notes)

    # 0. Stem energy gate
    if BASS_STEM_GATE_ENABLED:
        before = len(notes)
        notes  = gate_notes_by_stem_energy_for(
            notes, "bass",
            window_s=BASS_STEM_GATE_WINDOW,
            thresh=BASS_STEM_GATE_THRESH,
        )
        print(f"[Stage 3B] After stem energy gate   : {len(notes):4d}  (removed {before - len(notes)})")

    # 1. Duration filter (high notes scaled shorter — upper-register bass is real but brief)
    before = len(notes)
    def _eff_min_dur_b(pitch):
        if pitch >= BASS_HIGH_PITCH_FLOOR_1:
            return min_dur * BASS_HIGH_PITCH_MIN_DUR_SCALE
        return min_dur
    notes  = [n for n in notes if n["duration"] >= _eff_min_dur_b(n["pitch"])]
    print(f"[Stage 3B] After duration filter    : {len(notes):4d}  (removed {before - len(notes)})")

    # 2. Confidence filter (two-tier graduated reduction for upper-register)
    def _eff_conf_thresh_b(pitch):
        if pitch >= BASS_HIGH_PITCH_FLOOR_2:
            return max(0.0, conf_thresh - BASS_HIGH_PITCH_REDUCTION_2)
        if pitch >= BASS_HIGH_PITCH_FLOOR_1:
            return max(0.0, conf_thresh - BASS_HIGH_PITCH_REDUCTION_1)
        return conf_thresh
    before = len(notes)
    notes  = [n for n in notes if n["confidence"] >= _eff_conf_thresh_b(n["pitch"])]
    print(f"[Stage 3B] After confidence filter  : {len(notes):4d}  (removed {before - len(notes)})"
          f"  [gate: {conf_thresh} / tier1: {_eff_conf_thresh_b(BASS_HIGH_PITCH_FLOOR_1):.3f}"
          f" / tier2: {_eff_conf_thresh_b(BASS_HIGH_PITCH_FLOOR_2):.3f}]")

    # 3. Bend / slide detection — always apply for bass (slides are common)
    before = len(notes)
    notes  = _merge_bends_bass(notes,
                                max_semitones=BASS_BEND_MAX_SEMITONES,
                                max_gap_s=BASS_BEND_MAX_GAP_S)
    print(f"[Stage 3B] After bend/slide merge   : {len(notes):4d}  (merged  {before - len(notes)})")

    # 4. Merge nearby same-pitch notes
    before = len(notes)
    notes  = _merge_nearby_bass(notes, gap_threshold=merge_gap, merge_ratio=merge_ratio)
    print(f"[Stage 3B] After merging nearby     : {len(notes):4d}  (merged  {before - len(notes)})")

    # 5. Local pitch context filter
    local_max_dev = BASS_LOCAL_PITCH_MAX_DEV.get(bass_style)
    if local_max_dev is not None:
        before = len(notes)
        notes  = _local_pitch_filter_bass(
            notes,
            window_s=BASS_LOCAL_PITCH_WINDOW_S,
            max_deviation=local_max_dev,
            min_context=BASS_LOCAL_PITCH_MIN_CONTEXT,
        )
        removed = before - len(notes)
        if removed:
            print(f"[Stage 3B] After local pitch filter : {len(notes):4d}  (removed {removed})")

    # 6. Ghost note isolation filter
    if BASS_ISOLATION_ENABLED:
        before = len(notes)
        notes  = _filter_isolated_notes_bass(notes, BASS_ISOLATION_WINDOW_S, BASS_ISOLATION_MIN_TOTAL)
        removed = before - len(notes)
        if removed:
            print(f"[Stage 3B] After isolation filter   : {len(notes):4d}  (removed {removed})")

    # 7. Polyphony limit (bass is typically monophonic)
    before = len(notes)
    notes  = _limit_polyphony_bass(notes, max_poly=max_poly)
    print(f"[Stage 3B] After polyphony limit    : {len(notes):4d}  (removed {before - len(notes)})")

    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 3B] Final: {len(notes)} notes")

    if save:
        out_path = os.path.join(get_instrument_dir("bass"), "03_cleaned_notes.json")
        with open(out_path, "w") as f:
            json.dump(notes, f, indent=2)
        print(f"[Stage 3B] Saved -> {out_path}")

        meta_path = os.path.join(get_instrument_dir("bass"), "03_clean_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"bass_style": bass_style, "conf_thresh": conf_thresh,
                       "max_poly": max_poly}, f, indent=2)

    return notes


# ── Post-key feedback filter ──────────────────────────────────────────────────

def apply_key_confidence_filter_bass(
    notes: list[dict],
    key_info: dict,
    conf_cutoff: float = BASS_KEY_CONFIDENCE_CUTOFF,
) -> list[dict]:
    """
    Retroactive cleanup: remove notes that are BOTH off-key AND low confidence.
    Bass often plays chromatic walks so the cutoff is lower than guitar.
    """
    if not key_info:
        return notes
    scale_pcs = set(key_info.get("scale_pcs", []))
    if not scale_pcs:
        return notes

    kept, removed = [], 0
    for note in notes:
        in_key = note["pitch"] % 12 in scale_pcs
        if not in_key and note["confidence"] < conf_cutoff:
            removed += 1
        else:
            kept.append(note)

    if removed:
        print(f"[Stage 5bB] Key-confidence filter: removed {removed} off-key "
              f"low-confidence notes  (conf < {conf_cutoff})")
    return kept


def apply_key_octave_correction_bass(notes: list[dict], key_info: dict) -> list[dict]:
    """
    Second-pass octave correction using local pitch context (bass version).
    Same algorithm as guitar but uses BASS_* thresholds.
    """
    import bisect

    if not notes:
        return notes

    sorted_order   = sorted(range(len(notes)), key=lambda i: notes[i]["start"])
    sorted_starts  = [notes[i]["start"] for i in sorted_order]
    sorted_pitches = [notes[i]["pitch"] for i in sorted_order]
    orig_to_sorted = {orig: si for si, orig in enumerate(sorted_order)}

    result    = list(notes)
    corrected = 0

    for orig_i, note in enumerate(notes):
        if note["confidence"] >= BASS_OCTAVE_CONF_GATE:
            continue

        si = orig_to_sorted[orig_i]
        t  = note["start"]

        lo = bisect.bisect_left( sorted_starts, t - BASS_OCTAVE_WINDOW_S)
        hi = bisect.bisect_right(sorted_starts, t + BASS_OCTAVE_WINDOW_S)

        neighbour_pitches = [sorted_pitches[j] for j in range(lo, hi) if j != si]
        if len(neighbour_pitches) < BASS_OCTAVE_MIN_NEIGHBORS:
            lo2 = bisect.bisect_left( sorted_starts, t - BASS_OCTAVE_WIDE_WINDOW_S)
            hi2 = bisect.bisect_right(sorted_starts, t + BASS_OCTAVE_WIDE_WINDOW_S)
            neighbour_pitches = [sorted_pitches[j] for j in range(lo2, hi2) if j != si]
            if len(neighbour_pitches) < BASS_OCTAVE_MIN_NEIGHBORS:
                continue

        median = sorted(neighbour_pitches)[len(neighbour_pitches) // 2]
        p      = note["pitch"]
        dist   = abs(p - median)

        if dist < BASS_OCTAVE_SHIFT_THRESHOLD:
            continue

        best_shift = None
        best_dist  = dist
        for delta in (12, -12):
            candidate = p + delta
            if BASS_MIDI_MIN <= candidate <= BASS_MIDI_MAX:
                shifted_dist = abs(candidate - median)
                if shifted_dist < best_dist:
                    best_dist  = shifted_dist
                    best_shift = delta

        if best_shift is not None:
            result[orig_i] = dict(note, pitch=p + best_shift)
            corrected += 1

    if corrected:
        print(f"[Stage 5bB] Octave correction: shifted {corrected} low-confidence notes")
    return result


# ── Filter implementations ────────────────────────────────────────────────────

def _merge_bends_bass(
    notes: list[dict],
    max_semitones: int,
    max_gap_s: float,
) -> list[dict]:
    """Merge bass slides / bends — consecutive semitone-adjacent notes with a tiny gap."""
    if len(notes) < 2:
        return notes

    notes = sorted(notes, key=lambda n: n["start"])
    merged = [dict(notes[0])]

    for nxt in notes[1:]:
        cur = merged[-1]
        cur_end       = cur["start"] + cur["duration"]
        gap           = nxt["start"] - cur_end
        semitone_dist = abs(nxt["pitch"] - cur["pitch"])

        if gap <= max_gap_s and 1 <= semitone_dist <= max_semitones:
            new_end = max(cur_end, nxt["start"] + nxt["duration"])
            merged[-1] = {
                **cur,
                "pitch":      min(cur["pitch"], nxt["pitch"]),
                "duration":   round(new_end - cur["start"], 4),
                "confidence": round(max(cur["confidence"], nxt["confidence"]), 4),
            }
        else:
            merged.append(dict(nxt))

    return merged


def _merge_nearby_bass(
    notes: list[dict],
    gap_threshold: float,
    merge_ratio: float = 0.0,
) -> list[dict]:
    """Merge consecutive same-pitch bass notes separated by a small gap."""
    if not notes:
        return notes

    by_pitch: dict[int, list[dict]] = {}
    for n in notes:
        by_pitch.setdefault(n["pitch"], []).append(n)

    merged = []
    for pitch, group in by_pitch.items():
        group.sort(key=lambda n: n["start"])
        current = dict(group[0])
        for nxt in group[1:]:
            current_end = current["start"] + current["duration"]
            gap         = nxt["start"] - current_end
            ratio_limit = merge_ratio * min(current["duration"], nxt["duration"])
            if gap <= gap_threshold or (gap > 0 and gap <= ratio_limit):
                new_end = max(current_end, nxt["start"] + nxt["duration"])
                current["duration"]   = round(new_end - current["start"], 4)
                current["confidence"] = round(max(current["confidence"], nxt["confidence"]), 4)
            else:
                merged.append(current)
                current = dict(nxt)
        merged.append(current)

    return merged


def _local_pitch_filter_bass(
    notes: list[dict],
    window_s: float,
    max_deviation: int,
    min_context: int = 4,
) -> list[dict]:
    import bisect

    if not notes:
        return notes

    notes_sorted = sorted(notes, key=lambda n: n["start"])
    starts  = [n["start"] for n in notes_sorted]
    pitches = [n["pitch"] for n in notes_sorted]

    kept = []
    for i, note in enumerate(notes_sorted):
        t  = note["start"]
        lo = bisect.bisect_left(starts,  t - window_s)
        hi = bisect.bisect_right(starts, t + window_s)

        context = [pitches[j] for j in range(lo, hi) if j != i]
        if len(context) < min_context:
            kept.append(note)
            continue

        context.sort()
        median = context[len(context) // 2]
        if abs(note["pitch"] - median) <= max_deviation:
            kept.append(note)

    return kept


def _filter_isolated_notes_bass(
    notes: list[dict],
    window_s: float,
    min_total: int,
) -> list[dict]:
    import bisect
    if len(notes) < min_total:
        return notes
    starts = sorted(n["start"] for n in notes)
    return [
        n for n in notes
        if (bisect.bisect_right(starts, n["start"] + window_s)
            - bisect.bisect_left(starts, n["start"] - window_s)) > 1
    ]


def _limit_polyphony_bass(notes: list[dict], max_poly: int) -> list[dict]:
    """Keep at most max_poly simultaneously active notes (evict lowest confidence first)."""
    if not notes:
        return notes

    events = []
    for i, note in enumerate(notes):
        events.append((note["start"],                    0, i))
        events.append((note["start"] + note["duration"], 1, i))
    events.sort(key=lambda e: (e[0], e[1]))

    active    = {}
    to_remove = set()

    for _time, event_type, idx in events:
        if idx in to_remove:
            continue
        if event_type == 0:
            active[idx] = notes[idx]
            while len(active) > max_poly:
                worst = min(active, key=lambda i: (
                    active[i]["confidence"],
                    -active[i]["pitch"],
                ))
                to_remove.add(worst)
                del active[worst]
        else:
            active.pop(idx, None)

    return [n for i, n in enumerate(notes) if i not in to_remove]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_bpm() -> float | None:
    path = os.path.join(get_shared_dir(), "04_tempo.json")
    try:
        with open(path) as f:
            return json.load(f).get("bpm")
    except FileNotFoundError:
        return None


def load_cleaned_notes_bass() -> list[dict]:
    path = os.path.join(get_instrument_dir("bass"), "03_cleaned_notes.json")
    with open(path) as f:
        return json.load(f)
