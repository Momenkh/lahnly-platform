"""
Stage 3 (Vocals): Note Cleaning
Output: cleaned notes saved to outputs/03_vocals_cleaned_notes.json
"""
import json, os
from pipeline.config import get_instrument_dir, get_shared_dir
from pipeline.settings import (
    VOCALS_MIDI_MIN, VOCALS_MIDI_MAX,
    VOCALS_CLEANING_TYPE_PARAMS, VOCALS_DEFAULT_MERGE_GAP_S,
    VOCALS_MERGE_BEAT_DIVISOR, VOCALS_MERGE_GAP_FLOOR_S, VOCALS_MERGE_GAP_CEILING_S,
    VOCALS_BEND_MAX_SEMITONES, VOCALS_BEND_MAX_GAP_S,
    VOCALS_KEY_CONFIDENCE_CUTOFF, VOCALS_BPM_MIN_DUR_CLAMP_MIN_S, VOCALS_BPM_MIN_DUR_CLAMP_MAX_S,
    VOCALS_LOCAL_PITCH_WINDOW_S, VOCALS_LOCAL_PITCH_MIN_CONTEXT, VOCALS_LOCAL_PITCH_MAX_DEV,
    VOCALS_OCTAVE_CONF_GATE, VOCALS_OCTAVE_WINDOW_S, VOCALS_OCTAVE_WIDE_WINDOW_S,
    VOCALS_OCTAVE_MIN_NEIGHBORS, VOCALS_OCTAVE_SHIFT_THRESHOLD,
    VOCALS_STEM_GATE_ENABLED, VOCALS_STEM_GATE_WINDOW, VOCALS_STEM_GATE_THRESH,
    VOCALS_CONF_THRESH_MIN,
    VOCALS_ISOLATION_ENABLED,
    VOCALS_ISOLATION_WINDOW_S,
    VOCALS_ISOLATION_MIN_TOTAL,
    VOCALS_ISOLATION_PITCH_AWARE,
    VOCALS_ISOLATION_MAX_INTERVAL_ST,
    VOCALS_ISOLATION_MIN_NEIGHBORS,
    VOCALS_HIGH_PITCH_FLOOR_1,
    VOCALS_HIGH_PITCH_REDUCTION_1,
    VOCALS_HIGH_PITCH_FLOOR_2,
    VOCALS_HIGH_PITCH_REDUCTION_2,
    VOCALS_HIGH_PITCH_MIN_DUR_SCALE,
    VOCALS_POLY_ROOT_PROTECTION,
    VOCALS_POLY_HEIGHT_PENALTY_ST,
    VOCALS_POLY_HEIGHT_PENALTY_COEF,
    VOCALS_BPM_FAST_THRESHOLD_BPM,
    VOCALS_BPM_FAST_SUBDIV,
    HARMONIC_COHERENCE_ENABLED,
    HARMONIC_CONF_OVERRIDE,
    SPECTRAL_PRESENCE_ENABLED,
    SPECTRAL_PRESENCE_MIN_ENERGY,
    SPECTRAL_PRESENCE_CONF_OVERRIDE,
    ATTACK_GATE_ENABLED,
    ATTACK_GATE_PRE_WINDOW_S,
    ATTACK_GATE_POST_WINDOW_S,
    ATTACK_GATE_MIN_RATIO,
    ATTACK_GATE_CONF_OVERRIDE,
)


def clean_notes_vocals(raw_notes, vocals_mode="vocals_lead", save=True):
    from pipeline.shared.separation import load_stem_meta_for, gate_notes_by_stem_energy_for

    if vocals_mode not in VOCALS_CLEANING_TYPE_PARAMS:
        vocals_mode = "vocals_lead"

    min_dur_fb, conf_floor, conf_scale, max_poly, bpm_subdiv, merge_ratio = (
        VOCALS_CLEANING_TYPE_PARAMS[vocals_mode]
    )
    stem_conf   = load_stem_meta_for("vocals").get("stem_confidence", 0.5)
    bpm         = _load_bpm()
    conf_thresh = round(max(VOCALS_CONF_THRESH_MIN,
                            conf_floor + (1.0 - stem_conf) * conf_scale), 3)
    if bpm:
        beat_s    = 60.0 / bpm
        merge_gap = round(max(VOCALS_MERGE_GAP_FLOOR_S,
                              min(VOCALS_MERGE_GAP_CEILING_S, beat_s / VOCALS_MERGE_BEAT_DIVISOR)), 4)
    else:
        merge_gap = VOCALS_DEFAULT_MERGE_GAP_S
    if bpm:
        eff_subdiv = VOCALS_BPM_FAST_SUBDIV if bpm >= VOCALS_BPM_FAST_THRESHOLD_BPM else bpm_subdiv
        min_dur = round(
            max(VOCALS_BPM_MIN_DUR_CLAMP_MIN_S,
                min(VOCALS_BPM_MIN_DUR_CLAMP_MAX_S, 60.0 / (bpm * eff_subdiv))),
            4,
        )
    else:
        min_dur = min_dur_fb

    print(f"[Stage 3V] Cleaning {len(raw_notes)} raw notes  (mode: {vocals_mode})")
    print(f"[Stage 3V] stem_conf={stem_conf:.2f}  conf_thresh={conf_thresh}  "
          f"merge_gap={merge_gap*1000:.0f}ms  max_poly={max_poly}")

    notes = list(raw_notes)

    if VOCALS_STEM_GATE_ENABLED:
        before = len(notes)
        notes  = gate_notes_by_stem_energy_for(notes, "vocals",
                                                window_s=VOCALS_STEM_GATE_WINDOW,
                                                thresh=VOCALS_STEM_GATE_THRESH)
        print(f"[Stage 3V] After stem energy gate   : {len(notes):4d}  (removed {before - len(notes)})")

    def _eff_min_dur_v(pitch):
        if pitch >= VOCALS_HIGH_PITCH_FLOOR_1:
            return min_dur * VOCALS_HIGH_PITCH_MIN_DUR_SCALE
        return min_dur
    before = len(notes); notes = [n for n in notes if n["duration"] >= _eff_min_dur_v(n["pitch"])]
    print(f"[Stage 3V] After duration filter    : {len(notes):4d}  (removed {before - len(notes)})")

    def _eff_conf_thresh_v(pitch):
        if pitch >= VOCALS_HIGH_PITCH_FLOOR_2:
            return max(0.0, conf_thresh - VOCALS_HIGH_PITCH_REDUCTION_2)
        if pitch >= VOCALS_HIGH_PITCH_FLOOR_1:
            return max(0.0, conf_thresh - VOCALS_HIGH_PITCH_REDUCTION_1)
        return conf_thresh
    before = len(notes); notes = [n for n in notes if n["confidence"] >= _eff_conf_thresh_v(n["pitch"])]
    print(f"[Stage 3V] After confidence filter  : {len(notes):4d}  (removed {before - len(notes)})"
          f"  [gate: {conf_thresh} / tier1: {_eff_conf_thresh_v(VOCALS_HIGH_PITCH_FLOOR_1):.3f}"
          f" / tier2: {_eff_conf_thresh_v(VOCALS_HIGH_PITCH_FLOOR_2):.3f}]")

    # Vibrato / portamento merge
    before = len(notes); notes = _merge_bends(notes, VOCALS_BEND_MAX_SEMITONES, VOCALS_BEND_MAX_GAP_S)
    print(f"[Stage 3V] After vibrato merge      : {len(notes):4d}  (merged  {before - len(notes)})")

    before = len(notes); notes = _merge_nearby(notes, merge_gap, merge_ratio)
    print(f"[Stage 3V] After merging nearby     : {len(notes):4d}  (merged  {before - len(notes)})")

    local_max_dev = VOCALS_LOCAL_PITCH_MAX_DEV.get(vocals_mode)
    if local_max_dev:
        before = len(notes); notes = _local_pitch_filter(notes, VOCALS_LOCAL_PITCH_WINDOW_S, local_max_dev, VOCALS_LOCAL_PITCH_MIN_CONTEXT)
        if before != len(notes):
            print(f"[Stage 3V] After local pitch filter : {len(notes):4d}  (removed {before - len(notes)})")

    if VOCALS_ISOLATION_ENABLED:
        before = len(notes)
        if VOCALS_ISOLATION_PITCH_AWARE:
            from pipeline.shared.spectral import filter_isolated_notes_pitch_aware
            notes = filter_isolated_notes_pitch_aware(
                notes,
                window_s=VOCALS_ISOLATION_WINDOW_S,
                min_neighbors=VOCALS_ISOLATION_MIN_NEIGHBORS,
                max_interval_st=VOCALS_ISOLATION_MAX_INTERVAL_ST,
            )
        else:
            notes = _filter_isolated_notes_vocals(notes, VOCALS_ISOLATION_WINDOW_S, VOCALS_ISOLATION_MIN_TOTAL)
        removed = before - len(notes)
        if removed:
            print(f"[Stage 3V] After isolation filter   : {len(notes):4d}  (removed {removed})")

    # Spectral gates — pitch-specific ghost note removal
    from pipeline.shared.separation import get_stem_path_for
    _stem = get_stem_path_for("vocals")
    if os.path.isfile(_stem):
        from pipeline.shared.spectral import (
            compute_stem_cqt,
            check_harmonic_coherence,
            spectral_presence_gate,
            attack_envelope_gate,
        )
        import librosa as _librosa
        _cqt, _sr, _hop = compute_stem_cqt(_stem)
        _stem_audio, _sr2 = _librosa.load(_stem, sr=22050, mono=True)

        if HARMONIC_COHERENCE_ENABLED:
            before = len(notes)
            notes  = check_harmonic_coherence(notes, _cqt, _sr, _hop,
                                              dominance_ratio=2.5,
                                              conf_override=HARMONIC_CONF_OVERRIDE)
            removed = before - len(notes)
            if removed:
                print(f"[Stage 3V] After harmonic coherence : {len(notes):4d}  (removed {removed} overtones)")

        if SPECTRAL_PRESENCE_ENABLED:
            before = len(notes)
            notes  = spectral_presence_gate(notes, _cqt, _sr, _hop,
                                            min_energy=SPECTRAL_PRESENCE_MIN_ENERGY,
                                            conf_override=SPECTRAL_PRESENCE_CONF_OVERRIDE)
            removed = before - len(notes)
            if removed:
                print(f"[Stage 3V] After spectral presence  : {len(notes):4d}  (removed {removed} absent pitches)")

        if ATTACK_GATE_ENABLED:
            before = len(notes)
            notes  = attack_envelope_gate(notes, _stem_audio, _sr2,
                                          pre_window_s=ATTACK_GATE_PRE_WINDOW_S,
                                          post_window_s=ATTACK_GATE_POST_WINDOW_S,
                                          min_ratio=ATTACK_GATE_MIN_RATIO,
                                          conf_override=ATTACK_GATE_CONF_OVERRIDE)
            removed = before - len(notes)
            if removed:
                print(f"[Stage 3V] After attack gate        : {len(notes):4d}  (removed {removed} no-ramp notes)")

    before = len(notes); notes = _limit_polyphony(notes, max_poly)
    print(f"[Stage 3V] After polyphony limit    : {len(notes):4d}  (removed {before - len(notes)})")

    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 3V] Final: {len(notes)} notes")

    if save:
        path = os.path.join(get_instrument_dir("vocals"), "03_cleaned_notes.json")
        with open(path, "w") as f: json.dump(notes, f, indent=2)
        print(f"[Stage 3V] Saved -> {path}")
        meta = os.path.join(get_instrument_dir("vocals"), "03_clean_meta.json")
        with open(meta, "w") as f: json.dump({"vocals_mode": vocals_mode, "conf_thresh": conf_thresh}, f, indent=2)
    return notes


def apply_key_confidence_filter_vocals(notes, key_info, conf_cutoff=VOCALS_KEY_CONFIDENCE_CUTOFF):
    if not key_info: return notes
    scale_pcs = set(key_info.get("scale_pcs", []))
    if not scale_pcs: return notes
    kept, removed = [], 0
    for note in notes:
        if note["pitch"] % 12 not in scale_pcs and note["confidence"] < conf_cutoff:
            removed += 1
        else:
            kept.append(note)
    if removed:
        print(f"[Stage 5bV] Key-confidence filter: removed {removed} off-key notes")
    return kept


def apply_key_octave_correction_vocals(notes, key_info):
    import bisect
    if not notes: return notes
    so = sorted(range(len(notes)), key=lambda i: notes[i]["start"])
    ss = [notes[i]["start"] for i in so]; sp = [notes[i]["pitch"] for i in so]
    o2s = {orig: si for si, orig in enumerate(so)}
    result = list(notes); corrected = 0
    for orig_i, note in enumerate(notes):
        if note["confidence"] >= VOCALS_OCTAVE_CONF_GATE: continue
        si = o2s[orig_i]; t = note["start"]
        lo = bisect.bisect_left(ss, t - VOCALS_OCTAVE_WINDOW_S)
        hi = bisect.bisect_right(ss, t + VOCALS_OCTAVE_WINDOW_S)
        nbrs = [sp[j] for j in range(lo, hi) if j != si]
        if len(nbrs) < VOCALS_OCTAVE_MIN_NEIGHBORS:
            lo2 = bisect.bisect_left(ss, t - VOCALS_OCTAVE_WIDE_WINDOW_S)
            hi2 = bisect.bisect_right(ss, t + VOCALS_OCTAVE_WIDE_WINDOW_S)
            nbrs = [sp[j] for j in range(lo2, hi2) if j != si]
            if len(nbrs) < VOCALS_OCTAVE_MIN_NEIGHBORS: continue
        median = sorted(nbrs)[len(nbrs) // 2]; p = note["pitch"]; dist = abs(p - median)
        if dist < VOCALS_OCTAVE_SHIFT_THRESHOLD: continue
        best_shift = best_dist = None; best_dist = dist
        for delta in (12, -12):
            c = p + delta
            if VOCALS_MIDI_MIN <= c <= VOCALS_MIDI_MAX and abs(c - median) < best_dist:
                best_dist = abs(c - median); best_shift = delta
        if best_shift:
            result[orig_i] = dict(note, pitch=p + best_shift); corrected += 1
    if corrected: print(f"[Stage 5bV] Octave correction: shifted {corrected} notes")
    return result


def _merge_bends(notes, max_st, max_gap):
    if len(notes) < 2: return notes
    notes = sorted(notes, key=lambda n: n["start"]); merged = [dict(notes[0])]
    for nxt in notes[1:]:
        cur = merged[-1]; gap = nxt["start"] - (cur["start"] + cur["duration"])
        if gap <= max_gap and 1 <= abs(nxt["pitch"] - cur["pitch"]) <= max_st:
            new_end = max(cur["start"] + cur["duration"], nxt["start"] + nxt["duration"])
            merged[-1] = {**cur, "pitch": min(cur["pitch"], nxt["pitch"]),
                          "duration": round(new_end - cur["start"], 4),
                          "confidence": round(max(cur["confidence"], nxt["confidence"]), 4)}
        else:
            merged.append(dict(nxt))
    return merged


def _merge_nearby(notes, gap_threshold, merge_ratio=0.0):
    if not notes: return notes
    by_pitch = {}
    for n in notes: by_pitch.setdefault(n["pitch"], []).append(n)
    merged = []
    for _, group in by_pitch.items():
        group.sort(key=lambda n: n["start"]); cur = dict(group[0])
        for nxt in group[1:]:
            cur_end = cur["start"] + cur["duration"]; gap = nxt["start"] - cur_end
            ratio   = merge_ratio * min(cur["duration"], nxt["duration"])
            if gap <= gap_threshold or (gap > 0 and gap <= ratio):
                new_end = max(cur_end, nxt["start"] + nxt["duration"])
                cur["duration"] = round(new_end - cur["start"], 4)
                cur["confidence"] = round(max(cur["confidence"], nxt["confidence"]), 4)
            else:
                merged.append(cur); cur = dict(nxt)
        merged.append(cur)
    return merged


def _local_pitch_filter(notes, window_s, max_dev, min_ctx):
    import bisect
    if not notes: return notes
    ns = sorted(notes, key=lambda n: n["start"])
    starts = [n["start"] for n in ns]; pitches = [n["pitch"] for n in ns]; kept = []
    for i, note in enumerate(ns):
        t = note["start"]
        lo = bisect.bisect_left(starts, t - window_s); hi = bisect.bisect_right(starts, t + window_s)
        ctx = [pitches[j] for j in range(lo, hi) if j != i]
        if len(ctx) < min_ctx: kept.append(note); continue
        ctx.sort(); median = ctx[len(ctx) // 2]
        if abs(note["pitch"] - median) <= max_dev: kept.append(note)
    return kept


def _filter_isolated_notes_vocals(
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


def _limit_polyphony(notes, max_poly):
    if not notes: return notes
    events = []
    for i, n in enumerate(notes):
        events += [(n["start"], 0, i), (n["start"] + n["duration"], 1, i)]
    events.sort(key=lambda e: (e[0], e[1]))
    active = {}; to_remove = set()

    def _eviction_score(idx):
        note    = active[idx]
        conf    = float(note.get("confidence", 0.0))
        pitches = [active[j]["pitch"] for j in active]
        is_lowest  = note["pitch"] == min(pitches)
        root_bonus = 0.5 if (VOCALS_POLY_ROOT_PROTECTION and is_lowest) else 0.0
        median_p   = sorted(pitches)[len(pitches) // 2]
        over       = max(0, note["pitch"] - median_p - VOCALS_POLY_HEIGHT_PENALTY_ST)
        return (1.0 - conf) + over * VOCALS_POLY_HEIGHT_PENALTY_COEF - root_bonus

    for _, et, idx in events:
        if idx in to_remove: continue
        if et == 0:
            active[idx] = notes[idx]
            while len(active) > max_poly:
                worst = max(active, key=_eviction_score)
                to_remove.add(worst); del active[worst]
        else:
            active.pop(idx, None)
    return [n for i, n in enumerate(notes) if i not in to_remove]


def _load_bpm():
    path = os.path.join(get_shared_dir(), "04_tempo.json")
    try:
        with open(path) as f: return json.load(f).get("bpm")
    except FileNotFoundError: return None


def load_cleaned_notes_vocals():
    path = os.path.join(get_instrument_dir("vocals"), "03_cleaned_notes.json")
    with open(path) as f: return json.load(f)
