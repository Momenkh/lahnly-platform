"""
Stage 3 (Piano): Note Cleaning
Input:  raw piano notes [{pitch, start, duration, confidence}]
Output: cleaned notes, saved to outputs/03_piano_cleaned_notes.json
"""

import json
import os

from pipeline.config import get_instrument_dir, get_shared_dir
from pipeline.settings import (
    PIANO_MIDI_MIN,
    PIANO_MIDI_MAX,
    PIANO_CLEANING_TYPE_PARAMS,
    PIANO_DEFAULT_MERGE_GAP_S,
    PIANO_MERGE_BEAT_DIVISOR,
    PIANO_MERGE_GAP_FLOOR_S,
    PIANO_MERGE_GAP_CEILING_S,
    PIANO_KEY_CONFIDENCE_CUTOFF,
    PIANO_BPM_MIN_DUR_CLAMP_MIN_S,
    PIANO_BPM_MIN_DUR_CLAMP_MAX_S,
    PIANO_LOCAL_PITCH_WINDOW_S,
    PIANO_LOCAL_PITCH_MIN_CONTEXT,
    PIANO_LOCAL_PITCH_MAX_DEV,
    PIANO_OCTAVE_CONF_GATE,
    PIANO_OCTAVE_WINDOW_S,
    PIANO_OCTAVE_WIDE_WINDOW_S,
    PIANO_OCTAVE_MIN_NEIGHBORS,
    PIANO_OCTAVE_SHIFT_THRESHOLD,
    PIANO_STEM_GATE_ENABLED,
    PIANO_STEM_GATE_WINDOW,
    PIANO_STEM_GATE_THRESH,
    PIANO_ISOLATION_ENABLED,
    PIANO_ISOLATION_WINDOW_S,
    PIANO_ISOLATION_MIN_TOTAL,
    PIANO_ISOLATION_PITCH_AWARE,
    PIANO_ISOLATION_MAX_INTERVAL_ST,
    PIANO_POLY_ROOT_PROTECTION,
    PIANO_POLY_HEIGHT_PENALTY_ST,
    PIANO_POLY_HEIGHT_PENALTY_COEF,
    PIANO_HARMONIC_DOMINANCE_RATIO,
    PIANO_BPM_FAST_THRESHOLD_BPM,
    PIANO_BPM_FAST_MELODY_SUBDIV,
    PIANO_BPM_FAST_CHORD_SUBDIV,
    PIANO_SUSTAIN_MIN_CONCURRENT,
    PIANO_SUSTAIN_MERGE_GAP_SCALE,
    HARMONIC_COHERENCE_ENABLED,
    HARMONIC_CONF_OVERRIDE,
    SPECTRAL_PRESENCE_ENABLED,
    SPECTRAL_PRESENCE_MIN_ENERGY,
    SPECTRAL_PRESENCE_CONF_OVERRIDE,
    PIANO_ATTACK_GATE_ENABLED,
    ATTACK_GATE_PRE_WINDOW_S,
    ATTACK_GATE_POST_WINDOW_S,
    ATTACK_GATE_MIN_RATIO,
    ATTACK_GATE_CONF_OVERRIDE,
    PIANO_HIGH_PITCH_FLOOR_1,
    PIANO_HIGH_PITCH_REDUCTION_1,
    PIANO_HIGH_PITCH_FLOOR_2,
    PIANO_HIGH_PITCH_REDUCTION_2,
    PIANO_HIGH_PITCH_MIN_DUR_SCALE,
    PIANO_CHORD_RECOVERY_ENABLED,
    PIANO_CHORD_RECOVERY_WINDOW_S,
    PIANO_CHORD_RECOVERY_CHROMA_TOP_N,
    PIANO_CHORD_RECOVERY_CHROMA_MIN_ENERGY,
    PIANO_CHORD_RECOVERY_MAX_NOTES_TRIGGER,
    PIANO_CHORD_RECOVERY_MAX_PER_WINDOW,
    PIANO_CHORD_RECOVERY_MIN_RAW_CONF,
)


def clean_notes_piano(
    raw_notes: list[dict],
    piano_mode: str = "piano_chord",
    save: bool = True,
) -> list[dict]:
    """
    Clean raw piano notes.
    piano_mode: "piano_melody" | "piano_chord"
    """
    from pipeline.shared.separation import load_stem_meta_for, gate_notes_by_stem_energy_for

    if piano_mode not in PIANO_CLEANING_TYPE_PARAMS:
        print(f"[Stage 3P] Unknown mode '{piano_mode}' — defaulting to 'piano_chord'")
        piano_mode = "piano_chord"

    min_dur_fallback, conf_floor, conf_stem_scale, max_poly, bpm_subdiv, merge_ratio = (
        PIANO_CLEANING_TYPE_PARAMS[piano_mode]
    )

    stem_meta = load_stem_meta_for("piano")
    stem_conf = stem_meta.get("stem_confidence", 0.5)
    bpm       = _load_bpm()

    conf_thresh = round(conf_floor + (1.0 - stem_conf) * conf_stem_scale, 3)

    if bpm:
        beat_s    = 60.0 / bpm
        merge_gap = round(max(PIANO_MERGE_GAP_FLOOR_S,
                              min(PIANO_MERGE_GAP_CEILING_S, beat_s / PIANO_MERGE_BEAT_DIVISOR)), 4)
    else:
        merge_gap = PIANO_DEFAULT_MERGE_GAP_S

    if bpm:
        _is_melody = piano_mode == "piano_melody"
        _fast_sub  = PIANO_BPM_FAST_MELODY_SUBDIV if _is_melody else PIANO_BPM_FAST_CHORD_SUBDIV
        _eff_sub   = _fast_sub if bpm >= PIANO_BPM_FAST_THRESHOLD_BPM else bpm_subdiv
        min_dur = round(
            max(PIANO_BPM_MIN_DUR_CLAMP_MIN_S,
                min(PIANO_BPM_MIN_DUR_CLAMP_MAX_S, 60.0 / (bpm * _eff_sub))), 4)
    else:
        min_dur = min_dur_fallback

    print(f"[Stage 3P] Cleaning {len(raw_notes)} raw notes  (mode: {piano_mode})")
    print(f"[Stage 3P] stem_conf={stem_conf:.2f}  conf_thresh={conf_thresh}  "
          f"merge_gap={merge_gap*1000:.0f}ms  max_poly={max_poly}  min_dur={min_dur*1000:.0f}ms"
          + (f"  bpm={bpm:.1f}" if bpm else "  bpm=unknown"))

    notes = list(raw_notes)

    if PIANO_STEM_GATE_ENABLED:
        before = len(notes)
        notes  = gate_notes_by_stem_energy_for(
            notes, "piano",
            window_s=PIANO_STEM_GATE_WINDOW,
            thresh=PIANO_STEM_GATE_THRESH,
        )
        print(f"[Stage 3P] After stem energy gate   : {len(notes):4d}  (removed {before - len(notes)})")

    before = len(notes)
    def _eff_min_dur_p(pitch):
        if pitch >= PIANO_HIGH_PITCH_FLOOR_1:
            return min_dur * PIANO_HIGH_PITCH_MIN_DUR_SCALE
        return min_dur
    notes  = [n for n in notes if n["duration"] >= _eff_min_dur_p(n["pitch"])]
    print(f"[Stage 3P] After duration filter    : {len(notes):4d}  (removed {before - len(notes)})")

    def _eff_conf_thresh_p(pitch):
        if pitch >= PIANO_HIGH_PITCH_FLOOR_2:
            return max(0.0, conf_thresh - PIANO_HIGH_PITCH_REDUCTION_2)
        if pitch >= PIANO_HIGH_PITCH_FLOOR_1:
            return max(0.0, conf_thresh - PIANO_HIGH_PITCH_REDUCTION_1)
        return conf_thresh
    before = len(notes)
    notes  = [n for n in notes if n["confidence"] >= _eff_conf_thresh_p(n["pitch"])]
    print(f"[Stage 3P] After confidence filter  : {len(notes):4d}  (removed {before - len(notes)})"
          f"  [gate: {conf_thresh} / tier1: {_eff_conf_thresh_p(PIANO_HIGH_PITCH_FLOOR_1):.3f}"
          f" / tier2: {_eff_conf_thresh_p(PIANO_HIGH_PITCH_FLOOR_2):.3f}]")

    # No bend merging for piano — pianos don't bend pitches

    # Detect pedal-use sections: if many notes are simultaneously active, widen merge gap
    eff_merge_gap = merge_gap
    max_concurrent = _max_concurrent_notes_piano(notes)
    if max_concurrent >= PIANO_SUSTAIN_MIN_CONCURRENT:
        eff_merge_gap = round(merge_gap * PIANO_SUSTAIN_MERGE_GAP_SCALE, 4)
        print(f"[Stage 3P] Pedal section detected (max_concurrent={max_concurrent}) "
              f"→ merge_gap widened to {eff_merge_gap*1000:.0f}ms")

    before = len(notes)
    notes  = _merge_nearby_piano(notes, gap_threshold=eff_merge_gap, merge_ratio=merge_ratio)
    print(f"[Stage 3P] After merging nearby     : {len(notes):4d}  (merged  {before - len(notes)})")

    local_max_dev = PIANO_LOCAL_PITCH_MAX_DEV.get(piano_mode)
    if local_max_dev is not None:
        before = len(notes)
        notes  = _local_pitch_filter_piano(notes,
                                           window_s=PIANO_LOCAL_PITCH_WINDOW_S,
                                           max_deviation=local_max_dev,
                                           min_context=PIANO_LOCAL_PITCH_MIN_CONTEXT)
        removed = before - len(notes)
        if removed:
            print(f"[Stage 3P] After local pitch filter : {len(notes):4d}  (removed {removed})")

    if PIANO_ISOLATION_ENABLED:
        before = len(notes)
        if PIANO_ISOLATION_PITCH_AWARE:
            from pipeline.shared.spectral import filter_isolated_notes_pitch_aware
            notes = filter_isolated_notes_pitch_aware(
                notes,
                window_s=PIANO_ISOLATION_WINDOW_S,
                min_neighbors=2,
                max_interval_st=PIANO_ISOLATION_MAX_INTERVAL_ST,
            )
        else:
            notes = _filter_isolated_notes_piano(notes, PIANO_ISOLATION_WINDOW_S, PIANO_ISOLATION_MIN_TOTAL)
        removed = before - len(notes)
        if removed:
            print(f"[Stage 3P] After isolation filter   : {len(notes):4d}  (removed {removed})")

    # Spectral gates — pitch-specific ghost note removal (especially for piano overtones)
    _stem_p = os.path.join(get_instrument_dir("piano"), "01_stem.wav")
    if os.path.isfile(_stem_p):
        from pipeline.shared.spectral import (
            compute_stem_cqt,
            check_harmonic_coherence,
            spectral_presence_gate,
            attack_envelope_gate,
        )
        import librosa as _librosa
        _cqt, _sr, _hop = compute_stem_cqt(_stem_p)
        _stem_audio, _sr2 = _librosa.load(_stem_p, sr=22050, mono=True)

        if HARMONIC_COHERENCE_ENABLED:
            before = len(notes)
            notes  = check_harmonic_coherence(notes, _cqt, _sr, _hop,
                                              dominance_ratio=PIANO_HARMONIC_DOMINANCE_RATIO,
                                              conf_override=HARMONIC_CONF_OVERRIDE)
            removed = before - len(notes)
            if removed:
                print(f"[Stage 3P] After harmonic coherence : {len(notes):4d}  (removed {removed} overtones)")

        if SPECTRAL_PRESENCE_ENABLED:
            before = len(notes)
            notes  = spectral_presence_gate(notes, _cqt, _sr, _hop,
                                            min_energy=SPECTRAL_PRESENCE_MIN_ENERGY,
                                            conf_override=SPECTRAL_PRESENCE_CONF_OVERRIDE)
            removed = before - len(notes)
            if removed:
                print(f"[Stage 3P] After spectral presence  : {len(notes):4d}  (removed {removed} absent pitches)")

        if PIANO_ATTACK_GATE_ENABLED:
            before = len(notes)
            notes  = attack_envelope_gate(notes, _stem_audio, _sr2,
                                          pre_window_s=ATTACK_GATE_PRE_WINDOW_S,
                                          post_window_s=ATTACK_GATE_POST_WINDOW_S,
                                          min_ratio=ATTACK_GATE_MIN_RATIO,
                                          conf_override=ATTACK_GATE_CONF_OVERRIDE)
            removed = before - len(notes)
            if removed:
                print(f"[Stage 3P] After attack gate        : {len(notes):4d}  (removed {removed} no-ramp notes)")

    before = len(notes)
    notes  = _limit_polyphony_piano(notes, max_poly=max_poly)
    print(f"[Stage 3P] After polyphony limit    : {len(notes):4d}  (removed {before - len(notes)})")

    # Chord-guided recovery: pull harmonically-matched raw notes back into bars
    # that the confidence/isolation filters left under-populated.
    if PIANO_CHORD_RECOVERY_ENABLED:
        stem_path = os.path.join(get_instrument_dir("piano"), "01_stem.wav")
        window_s  = round(max(0.25, min(1.0, 60.0 / bpm)), 3) if bpm else PIANO_CHORD_RECOVERY_WINDOW_S
        before = len(notes)
        notes = _chord_guided_recovery(
            cleaned_notes=notes,
            raw_notes=raw_notes,
            stem_path=stem_path,
            window_s=window_s,
            chroma_top_n=PIANO_CHORD_RECOVERY_CHROMA_TOP_N,
            chroma_min_energy=PIANO_CHORD_RECOVERY_CHROMA_MIN_ENERGY,
            max_notes_trigger=PIANO_CHORD_RECOVERY_MAX_NOTES_TRIGGER,
            max_per_window=PIANO_CHORD_RECOVERY_MAX_PER_WINDOW,
            min_raw_conf=PIANO_CHORD_RECOVERY_MIN_RAW_CONF,
        )
        added = len(notes) - before
        if added:
            # Re-apply polyphony limit on the expanded set
            notes = _limit_polyphony_piano(notes, max_poly=max_poly)
            print(f"[Stage 3P] After chord recovery     : {len(notes):4d}  (net +{added})")

    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 3P] Final: {len(notes)} notes")

    if save:
        out_path = os.path.join(get_instrument_dir("piano"), "03_cleaned_notes.json")
        with open(out_path, "w") as f:
            json.dump(notes, f, indent=2)
        print(f"[Stage 3P] Saved -> {out_path}")

        meta_path = os.path.join(get_instrument_dir("piano"), "03_clean_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"piano_mode": piano_mode, "conf_thresh": conf_thresh,
                       "max_poly": max_poly}, f, indent=2)

    return notes


def apply_key_confidence_filter_piano(
    notes: list[dict],
    key_info: dict,
    conf_cutoff: float = PIANO_KEY_CONFIDENCE_CUTOFF,
) -> list[dict]:
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
        print(f"[Stage 5bP] Key-confidence filter: removed {removed} off-key notes")
    return kept


def apply_key_octave_correction_piano(notes: list[dict], key_info: dict) -> list[dict]:
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
        if note["confidence"] >= PIANO_OCTAVE_CONF_GATE:
            continue

        si = orig_to_sorted[orig_i]
        t  = note["start"]
        lo = bisect.bisect_left( sorted_starts, t - PIANO_OCTAVE_WINDOW_S)
        hi = bisect.bisect_right(sorted_starts, t + PIANO_OCTAVE_WINDOW_S)

        neighbour_pitches = [sorted_pitches[j] for j in range(lo, hi) if j != si]
        if len(neighbour_pitches) < PIANO_OCTAVE_MIN_NEIGHBORS:
            lo2 = bisect.bisect_left( sorted_starts, t - PIANO_OCTAVE_WIDE_WINDOW_S)
            hi2 = bisect.bisect_right(sorted_starts, t + PIANO_OCTAVE_WIDE_WINDOW_S)
            neighbour_pitches = [sorted_pitches[j] for j in range(lo2, hi2) if j != si]
            if len(neighbour_pitches) < PIANO_OCTAVE_MIN_NEIGHBORS:
                continue

        median = sorted(neighbour_pitches)[len(neighbour_pitches) // 2]
        p      = note["pitch"]
        dist   = abs(p - median)

        if dist < PIANO_OCTAVE_SHIFT_THRESHOLD:
            continue

        best_shift = best_dist = None
        best_dist  = dist
        for delta in (12, -12):
            candidate = p + delta
            if PIANO_MIDI_MIN <= candidate <= PIANO_MIDI_MAX:
                d = abs(candidate - median)
                if d < best_dist:
                    best_dist  = d
                    best_shift = delta

        if best_shift is not None:
            result[orig_i] = dict(note, pitch=p + best_shift)
            corrected += 1

    if corrected:
        print(f"[Stage 5bP] Octave correction: shifted {corrected} notes")
    return result


def _max_concurrent_notes_piano(notes: list[dict]) -> int:
    """Return the maximum number of simultaneously active notes in the list."""
    if not notes:
        return 0
    events = []
    for n in notes:
        events.append((n["start"], 0))
        events.append((n["start"] + n["duration"], 1))
    events.sort(key=lambda e: (e[0], e[1]))
    current = peak = 0
    for _, et in events:
        if et == 0:
            current += 1
            peak = max(peak, current)
        else:
            current -= 1
    return peak


def _merge_nearby_piano(notes, gap_threshold, merge_ratio=0.0):
    if not notes:
        return notes
    by_pitch: dict[int, list] = {}
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


def _local_pitch_filter_piano(notes, window_s, max_deviation, min_context):
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


def _filter_isolated_notes_piano(
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


def _limit_polyphony_piano(notes, max_poly):
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
                worst = min(active, key=lambda i: (active[i]["confidence"], -active[i]["pitch"]))
                to_remove.add(worst)
                del active[worst]
        else:
            active.pop(idx, None)
    return [n for i, n in enumerate(notes) if i not in to_remove]


def _chord_guided_recovery(
    cleaned_notes: list[dict],
    raw_notes: list[dict],
    stem_path: str,
    window_s: float,
    chroma_top_n: int,
    chroma_min_energy: float,
    max_notes_trigger: int,
    max_per_window: int,
    min_raw_conf: float,
) -> list[dict]:
    """
    Chromagram-guided note recovery.

    For each time window, compute the dominant pitch classes from the stem
    chroma.  If the cleaned note list has fewer than max_notes_trigger notes
    in that window AND there is meaningful harmonic energy, search the raw
    notes for high-confidence candidates whose pitch class matches the
    detected chord and add them back.

    Only adds notes — never removes.  A final polyphony pass is run by the
    caller after this returns.
    """
    import bisect
    import numpy as np

    try:
        import librosa
    except ImportError:
        print("[Stage 3P] Chord recovery skipped (librosa not available)")
        return cleaned_notes

    if not os.path.isfile(stem_path):
        print(f"[Stage 3P] Chord recovery skipped (stem not found: {stem_path})")
        return cleaned_notes

    if not raw_notes:
        return cleaned_notes

    y, sr = librosa.load(stem_path, sr=22050, mono=True)
    hop_length = 512
    hop_s      = hop_length / sr

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    duration_s = len(y) / sr

    # Index raw notes by start time for fast lookup
    raw_sorted = sorted(raw_notes, key=lambda n: n["start"])
    raw_starts  = [n["start"] for n in raw_sorted]

    # Index cleaned notes for fast dedup check (pitch × start bucket)
    cleaned_index = {}
    for n in cleaned_notes:
        cleaned_index.setdefault(n["pitch"] % 12, []).append(n["start"])

    result = list(cleaned_notes)
    total_added = 0

    w_start = 0.0
    while w_start < duration_s:
        w_end = w_start + window_s

        # --- count cleaned notes that overlap this window ---
        window_cleaned = [
            n for n in cleaned_notes
            if n["start"] < w_end and (n["start"] + n["duration"]) > w_start
        ]
        if len(window_cleaned) >= max_notes_trigger:
            w_start = w_end
            continue

        # --- chroma analysis for this window ---
        f0 = int(w_start / hop_s)
        f1 = min(chroma.shape[1], int(w_end / hop_s))
        if f0 >= f1:
            w_start = w_end
            continue

        window_chroma = chroma[:, f0:f1].mean(axis=1)
        if window_chroma.max() < chroma_min_energy:
            w_start = w_end
            continue

        # Top-N pitch classes = chord tones
        chord_pcs = set(int(p) for p in np.argsort(window_chroma)[-chroma_top_n:])

        # --- find raw candidates in this window ---
        lo = bisect.bisect_left(raw_starts,  w_start)
        hi = bisect.bisect_right(raw_starts, w_end)

        already_pitches_starts = {(n["pitch"], round(n["start"], 2))
                                   for n in window_cleaned}

        candidates = []
        for raw in raw_sorted[lo:hi]:
            if raw["pitch"] % 12 not in chord_pcs:
                continue
            if raw["confidence"] < min_raw_conf:
                continue
            key = (raw["pitch"], round(raw["start"], 2))
            if key in already_pitches_starts:
                continue
            candidates.append(raw)

        if not candidates:
            w_start = w_end
            continue

        candidates.sort(key=lambda n: -n["confidence"])
        to_add = candidates[:max_per_window]
        result.extend(to_add)
        total_added += len(to_add)

        w_start = w_end

    if total_added:
        result.sort(key=lambda n: n["start"])
        print(f"[Stage 3P] Chord recovery: pulled back {total_added} notes from raw")

    return result


def _load_bpm():
    path = os.path.join(get_shared_dir(), "04_tempo.json")
    try:
        with open(path) as f:
            return json.load(f).get("bpm")
    except FileNotFoundError:
        return None


def load_cleaned_notes_piano() -> list[dict]:
    path = os.path.join(get_instrument_dir("piano"), "03_cleaned_notes.json")
    with open(path) as f:
        return json.load(f)
