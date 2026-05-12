"""
Stage 3: Note Cleaning
Input:  raw notes [{pitch, start, duration, confidence}]
Output: cleaned notes, saved to outputs/03_cleaned_notes.json

Filters applied in order:
  1. Confidence filter    — drop notes below adaptive threshold (scales with stem quality
                            AND guitar type — lead solos rate lower confidence than chords)
  2. Bass filter          — stricter confidence gate for notes below C2 (MIDI 36)
                            (relaxed when htdemucs_6s dedicated guitar stem was used)
  3. Bend detection       — consecutive notes within 2 semitones over a short gap are
                            merged into one note at the lower pitch (guitar bends/vibrato)
  4. Merge nearby         — merge same-pitch notes with gaps smaller than merge_gap
                            (merge_gap scales with tempo when available)
  5. Duration filter      — drop notes shorter than min_duration (type-dependent).
                            Runs AFTER merge so short fragments combined into a full
                            note survive (e.g. 3×50ms → 150ms passes a 77ms gate).
  6. Polyphony limit      — keep at most N simultaneous notes (type-dependent):
                              lead = 3, acoustic = 5, rhythm = 6
                            evicts by lowest confidence AND highest pitch (overtones)

Post-key feedback (called from main.py after Stage 5):
  apply_key_confidence_filter — removes notes that are BOTH off-key AND low confidence
                                 (retroactive cleanup using key context)

Guitar types
------------
  lead     Electric lead / solo. Fast single-note runs, bends, hammer-ons.
           Lower confidence floor — the model rates bent/vibrato notes poorly.
           Tight polyphony limit — solos are rarely >2-3 simultaneous pitches.
  acoustic Fingerpicked or strummed acoustic. Middle-ground thresholds.
  rhythm   Electric or acoustic rhythm / full chords. Default. Strictest
           confidence (chords have clear strong frame detections).
"""

import json
import os

from pipeline.config import get_instrument_dir, get_shared_dir
from pipeline.settings import (
    GUITAR_MIDI_MIN,
    GUITAR_MIDI_MAX,
    CLEANING_BASS_CUTOFF_MIDI,
    CLEANING_BASS_CONF_DEDICATED_STEM,
    CLEANING_BASS_CONF_NONDEDICATED_STEM,
    CLEANING_LOW_STRING_CUTOFF_MIDI,
    CLEANING_LOW_STRING_CONF_GATE,
    CLEANING_DEFAULT_MERGE_GAP_S,
    CLEANING_MERGE_BEAT_DIVISOR,
    CLEANING_MERGE_GAP_FLOOR_S,
    CLEANING_MERGE_GAP_CEILING_S,
    CLEANING_BEND_MAX_SEMITONES,
    CLEANING_BEND_MAX_GAP_S,
    CLEANING_BEND_MIN_FIRST_NOTE_S,
    CLEANING_TYPE_PARAMS,
    CLEANING_KEY_CONFIDENCE_CUTOFF,
    CLEANING_BPM_MIN_DUR_CLAMP_MIN_S,
    CLEANING_BPM_MIN_DUR_CLAMP_MAX_S,
    CLEANING_LOCAL_PITCH_WINDOW_S,
    CLEANING_LOCAL_PITCH_MIN_CONTEXT,
    CLEANING_LOCAL_PITCH_MAX_DEV,
    CLEANING_OCTAVE_CONF_GATE,
    CLEANING_OCTAVE_WINDOW_S,
    CLEANING_OCTAVE_WIDE_WINDOW_S,
    CLEANING_OCTAVE_MIN_NEIGHBORS,
    CLEANING_OCTAVE_SHIFT_THRESHOLD,
    CLEANING_HIGH_PITCH_FLOOR_1,
    CLEANING_HIGH_PITCH_REDUCTION_1,
    CLEANING_HIGH_PITCH_FLOOR_2,
    CLEANING_HIGH_PITCH_REDUCTION_2,
    CLEANING_HIGH_PITCH_MIN_DUR_SCALE,
    CLEANING_STEM_GATE_ENABLED,
    CLEANING_STEM_GATE_WINDOW,
    CLEANING_STEM_GATE_THRESH,
    CLEANING_ISOLATION_ENABLED,
    CLEANING_ISOLATION_WINDOW_S,
    CLEANING_ISOLATION_MIN_TOTAL,
    CLEANING_ISOLATION_PITCH_AWARE,
    CLEANING_ISOLATION_MAX_INTERVAL_ST,
    CLEANING_POLY_ROOT_PROTECTION,
    CLEANING_POLY_HEIGHT_PENALTY_ST,
    CLEANING_POLY_HEIGHT_PENALTY_COEF,
    CLEANING_BPM_FAST_THRESHOLD_BPM,
    CLEANING_BPM_FAST_LEAD_SUBDIV,
    CLEANING_BPM_FAST_RHYTHM_SUBDIV,
    HARMONIC_COHERENCE_ENABLED,
    HARMONIC_DOMINANCE_RATIO,
    HARMONIC_CONF_OVERRIDE,
    SPECTRAL_PRESENCE_ENABLED,
    SPECTRAL_PRESENCE_MIN_ENERGY,
    SPECTRAL_PRESENCE_CONF_OVERRIDE,
    GUITAR_ATTACK_GATE_ENABLED,
    ATTACK_GATE_PRE_WINDOW_S,
    ATTACK_GATE_POST_WINDOW_S,
    ATTACK_GATE_MIN_RATIO,
    ATTACK_GATE_CONF_OVERRIDE,
)

VALID_GUITAR_TYPES = list(CLEANING_TYPE_PARAMS.keys())


def clean_notes(
    raw_notes: list[dict],
    guitar_type: str = "clean",
    guitar_role: str = "rhythm",
    save: bool = True,
) -> list[dict]:
    """
    Clean raw pitch-detection output into stable, distinct notes.
    Loads stem_confidence and tempo_info from prior stages to tune thresholds.

    guitar_type: "acoustic" | "clean" | "distorted"
    guitar_role: "lead" | "rhythm"
    """
    from pipeline.shared.separation import load_stem_meta

    mode_key = f"{guitar_type}_{guitar_role}"
    if mode_key not in CLEANING_TYPE_PARAMS:
        print(f"[Stage 3] Unknown mode '{mode_key}' — defaulting to 'clean_rhythm'")
        mode_key = "clean_rhythm"

    min_dur_fallback, conf_floor, conf_stem_scale, max_poly, bpm_subdiv, merge_ratio = CLEANING_TYPE_PARAMS[mode_key]

    stem_meta  = load_stem_meta()
    stem_conf  = stem_meta.get("stem_confidence", 0.5)
    used_model = stem_meta.get("model", "unknown")
    bpm        = _load_bpm()

    # Adaptive confidence threshold: clean stem allows softer notes through
    conf_thresh = round(conf_floor + (1.0 - stem_conf) * conf_stem_scale, 3)

    # Adaptive bass threshold: if dedicated guitar stem, bass is already gone
    if "htdemucs_6s" in used_model:
        bass_conf_thresh = CLEANING_BASS_CONF_DEDICATED_STEM
    else:
        bass_conf_thresh = CLEANING_BASS_CONF_NONDEDICATED_STEM

    # Adaptive merge gap: beat / DIVISOR, clamped to [FLOOR, CEILING]
    if bpm:
        beat_s    = 60.0 / bpm
        merge_gap = round(max(CLEANING_MERGE_GAP_FLOOR_S,
                              min(CLEANING_MERGE_GAP_CEILING_S, beat_s / CLEANING_MERGE_BEAT_DIVISOR)), 4)
    else:
        merge_gap = CLEANING_DEFAULT_MERGE_GAP_S

    # Adaptive min duration: beat / bpm_subdiv, clamped to a sensible range.
    # At fast BPM (≥ FAST_THRESHOLD), use a finer subdivision so fast runs of
    # 16th/32nd notes are not deleted.  Falls back to fixed value when BPM unknown.
    if bpm:
        is_lead      = guitar_role == "lead"
        eff_subdiv   = bpm_subdiv
        if bpm >= CLEANING_BPM_FAST_THRESHOLD_BPM:
            eff_subdiv = CLEANING_BPM_FAST_LEAD_SUBDIV if is_lead else CLEANING_BPM_FAST_RHYTHM_SUBDIV
        bpm_min_dur = 60.0 / (bpm * eff_subdiv)
        min_dur = round(
            max(CLEANING_BPM_MIN_DUR_CLAMP_MIN_S,
                min(CLEANING_BPM_MIN_DUR_CLAMP_MAX_S, bpm_min_dur)),
            4,
        )
    else:
        min_dur = min_dur_fallback

    print(f"[Stage 3] Cleaning {len(raw_notes)} raw notes  (mode: {mode_key})")
    print(f"[Stage 3] stem_conf={stem_conf:.2f}  conf_thresh={conf_thresh}  "
          f"bass_conf={bass_conf_thresh}  merge_gap={merge_gap*1000:.0f}ms  "
          f"max_poly={max_poly}  min_dur={min_dur*1000:.0f}ms"
          + (f"  bpm={bpm:.1f}" if bpm else "  bpm=unknown"))

    notes = list(raw_notes)

    # 0. Stem energy gate — drop notes in provably silent stem regions
    #    (also applied in Stage 2; repeated here to cover --from-stage 3 runs)
    if CLEANING_STEM_GATE_ENABLED:
        from pipeline.shared.separation import gate_notes_by_stem_energy
        before = len(notes)
        notes  = gate_notes_by_stem_energy(notes,
                                           window_s=CLEANING_STEM_GATE_WINDOW,
                                           thresh=CLEANING_STEM_GATE_THRESH)
        print(f"[Stage 3] After stem energy gate   : {len(notes):4d}  (removed {before - len(notes)})")

    # 1. Adaptive confidence filter (two-tier: Tier1 MIDI>=69 -0.040, Tier2 MIDI>=84 -0.075)
    #    Run BEFORE duration filter so merged fragments aren't pre-killed as too short.
    def _eff_conf_thresh(pitch):
        if pitch >= CLEANING_HIGH_PITCH_FLOOR_2:
            return max(0.0, conf_thresh - CLEANING_HIGH_PITCH_REDUCTION_2)
        if pitch >= CLEANING_HIGH_PITCH_FLOOR_1:
            return max(0.0, conf_thresh - CLEANING_HIGH_PITCH_REDUCTION_1)
        return conf_thresh
    before = len(notes)
    notes  = [n for n in notes if n["confidence"] >= _eff_conf_thresh(n["pitch"])]
    print(f"[Stage 3] After confidence filter  : {len(notes):4d}  (removed {before - len(notes)})"
          f"  [gate: {conf_thresh} / tier1: {_eff_conf_thresh(CLEANING_HIGH_PITCH_FLOOR_1):.3f}"
          f" / tier2: {_eff_conf_thresh(CLEANING_HIGH_PITCH_FLOOR_2):.3f}]")

    # 2. Bass filter (sub-C2 bleed)
    before = len(notes)
    notes  = [
        n for n in notes
        if n["pitch"] >= CLEANING_BASS_CUTOFF_MIDI or n["confidence"] >= bass_conf_thresh
    ]
    print(f"[Stage 3] After bass filter        : {len(notes):4d}  (removed {before - len(notes)})")

    # 2b. Low-string gate — A/D/E string range (MIDI < G3) requires higher confidence.
    #     Body resonance and sympathetic vibration in this range score conf 0.20-0.49;
    #     intentional low bass lines score ≥ 0.55.
    before = len(notes)
    notes  = [
        n for n in notes
        if n["pitch"] >= CLEANING_LOW_STRING_CUTOFF_MIDI
        or n["confidence"] >= CLEANING_LOW_STRING_CONF_GATE
    ]
    removed = before - len(notes)
    if removed:
        print(f"[Stage 3] After low-string gate    : {len(notes):4d}  (removed {removed})")

    # 3. Bend detection — merge semitone-adjacent consecutive notes.
    #    Only for electric lead (clean/distorted): bends are common on electric and
    #    basic-pitch spreads them into 2 adjacent-pitch detections.
    #    Acoustic is EXCLUDED: bends are rare on acoustic, and its legato
    #    hammer-on runs (1-2 semitone steps, 0ms gap) are indistinguishable from
    #    bends — enabling this for acoustic destroys entire melodic phrases.
    if guitar_role == "lead" and guitar_type in ("clean", "distorted"):
        before = len(notes)
        notes  = _merge_bends(notes,
                               max_semitones=CLEANING_BEND_MAX_SEMITONES,
                               max_gap_s=CLEANING_BEND_MAX_GAP_S,
                               min_first_note_s=CLEANING_BEND_MIN_FIRST_NOTE_S)
        print(f"[Stage 3] After bend merge         : {len(notes):4d}  (merged  {before - len(notes)})")

    # 4. Merge nearby same-pitch notes
    before = len(notes)
    notes  = _merge_nearby(notes, gap_threshold=merge_gap, merge_ratio=merge_ratio)
    print(f"[Stage 3] After merging nearby     : {len(notes):4d}  (merged  {before - len(notes)})")

    # 5. Duration filter — applied AFTER merge so fragments combined into full notes
    #    survive. A 3×50ms fragment that merges into 150ms is now judged as 150ms.
    def _eff_min_dur(pitch):
        if pitch >= CLEANING_HIGH_PITCH_FLOOR_1:
            return min_dur * CLEANING_HIGH_PITCH_MIN_DUR_SCALE
        return min_dur
    before = len(notes)
    notes  = [n for n in notes if n["duration"] >= _eff_min_dur(n["pitch"])]
    print(f"[Stage 3] After duration filter    : {len(notes):4d}  (removed {before - len(notes)})")

    # 5b. Local pitch context filter
    local_max_dev = CLEANING_LOCAL_PITCH_MAX_DEV.get(mode_key)
    if local_max_dev is not None:
        before = len(notes)
        notes  = _local_pitch_filter(
            notes,
            window_s=CLEANING_LOCAL_PITCH_WINDOW_S,
            max_deviation=local_max_dev,
            min_context=CLEANING_LOCAL_PITCH_MIN_CONTEXT,
        )
        removed = before - len(notes)
        if removed:
            print(f"[Stage 3] After local pitch filter : {len(notes):4d}  (removed {removed})")

    # 6. Ghost note isolation filter (pitch-aware upgrade)
    if CLEANING_ISOLATION_ENABLED:
        before = len(notes)
        if CLEANING_ISOLATION_PITCH_AWARE:
            from pipeline.shared.spectral import filter_isolated_notes_pitch_aware
            notes = filter_isolated_notes_pitch_aware(
                notes,
                window_s=CLEANING_ISOLATION_WINDOW_S,
                min_neighbors=2,
                max_interval_st=CLEANING_ISOLATION_MAX_INTERVAL_ST,
            )
        else:
            notes = _filter_isolated_notes(notes, CLEANING_ISOLATION_WINDOW_S, CLEANING_ISOLATION_MIN_TOTAL)
        removed = before - len(notes)
        if removed:
            print(f"[Stage 3] After isolation filter   : {len(notes):4d}  (removed {removed})")

    # 6b. Spectral gates — pitch-specific ghost note removal using stem CQT
    from pipeline.shared.separation import get_stem_path
    _stem = get_stem_path()
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
                                              dominance_ratio=HARMONIC_DOMINANCE_RATIO,
                                              conf_override=HARMONIC_CONF_OVERRIDE)
            removed = before - len(notes)
            if removed:
                print(f"[Stage 3] After harmonic coherence : {len(notes):4d}  (removed {removed} overtones)")

        if SPECTRAL_PRESENCE_ENABLED:
            before = len(notes)
            notes  = spectral_presence_gate(notes, _cqt, _sr, _hop,
                                            min_energy=SPECTRAL_PRESENCE_MIN_ENERGY,
                                            conf_override=SPECTRAL_PRESENCE_CONF_OVERRIDE)
            removed = before - len(notes)
            if removed:
                print(f"[Stage 3] After spectral presence  : {len(notes):4d}  (removed {removed} absent pitches)")

        if GUITAR_ATTACK_GATE_ENABLED:
            before = len(notes)
            notes  = attack_envelope_gate(notes, _stem_audio, _sr2,
                                          pre_window_s=ATTACK_GATE_PRE_WINDOW_S,
                                          post_window_s=ATTACK_GATE_POST_WINDOW_S,
                                          min_ratio=ATTACK_GATE_MIN_RATIO,
                                          conf_override=ATTACK_GATE_CONF_OVERRIDE)
            removed = before - len(notes)
            if removed:
                print(f"[Stage 3] After attack gate        : {len(notes):4d}  (removed {removed} no-ramp notes)")

    # 7. Polyphony limit (smart eviction — protects chord root)
    before = len(notes)
    notes  = _limit_polyphony(notes, max_poly=max_poly)
    print(f"[Stage 3] After polyphony limit    : {len(notes):4d}  (removed {before - len(notes)})")

    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 3] Final: {len(notes)} notes")

    if save:
        out_path = os.path.join(get_instrument_dir("guitar"), "03_cleaned_notes.json")
        with open(out_path, "w") as f:
            json.dump(notes, f, indent=2)
        print(f"[Stage 3] Saved -> {out_path}")

        # Save mode used so re-runs can detect mismatches
        meta_path = os.path.join(get_instrument_dir("guitar"), "03_clean_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"guitar_type": guitar_type, "guitar_role": guitar_role,
                       "mode": mode_key, "conf_thresh": conf_thresh,
                       "max_poly": max_poly}, f, indent=2)

    return notes


# ── Post-key feedback filter (run after Stage 5) ─────────────────────────────

def apply_key_confidence_filter(
    notes: list[dict],
    key_info: dict,
    conf_cutoff: float = CLEANING_KEY_CONFIDENCE_CUTOFF,
    protected_pcs: set[int] | None = None,
) -> list[dict]:
    """
    Retroactive cleanup using key context.

    Removes notes that are ALL THREE of:
      - Not in the detected key's scale (off-key pitch class)
      - Below conf_cutoff confidence
      - Not a chord tone (pitch class absent from protected_pcs)

    Notes that are in-key, have high confidence (chromatic passing tones),
    or belong to a detected chord voicing are kept.

    protected_pcs: set of pitch classes (0-11) found in any chord group
    during the pre-chord pass (get_chord_tone_pcs). Pass None to skip
    chord-tone protection (legacy behaviour).
    """
    if not key_info:
        return notes

    scale_pcs = set(key_info.get("scale_pcs", []))
    if not scale_pcs:
        return notes

    chord_protected = protected_pcs or set()
    kept, removed = [], 0
    for note in notes:
        pc     = note["pitch"] % 12
        in_key = pc in scale_pcs
        if not in_key and note["confidence"] < conf_cutoff and pc not in chord_protected:
            removed += 1
        else:
            kept.append(note)

    if removed:
        print(f"[Stage 5b] Key-confidence filter: removed {removed} off-key "
              f"low-confidence notes  (conf < {conf_cutoff})")
    if protected_pcs:
        n_protected = sum(
            1 for n in notes
            if n["pitch"] % 12 not in scale_pcs
            and n["confidence"] < conf_cutoff
            and n["pitch"] % 12 in protected_pcs
        )
        if n_protected:
            print(f"[Stage 5b] Chord-tone protection: kept {n_protected} off-key "
                  f"notes that are chord tones")
    return kept


def apply_key_confidence_filter_segmented(
    notes: list[dict],
    segments: list[dict],
    global_key_info: dict,
    conf_cutoff: float = CLEANING_KEY_CONFIDENCE_CUTOFF,
    protected_pcs: set[int] | None = None,
) -> list[dict]:
    """
    Per-segment variant of apply_key_confidence_filter.

    Each note is checked against the scale of the segment it falls in, so notes
    that are off-key for the global key but diatonic in their local section are
    preserved.  Falls back to global_key_info for notes outside any segment.
    """
    if not segments:
        return apply_key_confidence_filter(notes, global_key_info, conf_cutoff, protected_pcs)

    import bisect
    seg_starts = [s["seg_start"] for s in segments]
    chord_protected = protected_pcs or set()

    kept, removed = [], 0
    for note in notes:
        t  = note["start"]
        pc = note["pitch"] % 12

        # Find the segment this note belongs to
        idx = bisect.bisect_right(seg_starts, t) - 1
        if 0 <= idx < len(segments) and t <= segments[idx]["seg_end"]:
            scale_pcs = set(segments[idx].get("scale_pcs", []))
        else:
            scale_pcs = set(global_key_info.get("scale_pcs", []))

        if not scale_pcs:
            kept.append(note)
            continue

        in_key = pc in scale_pcs
        if not in_key and note["confidence"] < conf_cutoff and pc not in chord_protected:
            removed += 1
        else:
            kept.append(note)

    if removed:
        print(f"[Stage 5b] Segmented key-confidence filter: removed {removed} off-key "
              f"low-confidence notes  (conf < {conf_cutoff})")
    return kept


# ── Key-context octave correction (run after Stage 5) ─────────────────────────

def apply_key_octave_correction(notes: list[dict], key_info: dict) -> list[dict]:
    """
    Second-pass octave error correction using local pitch context.

    For each low-confidence note, look at all other notes within
    ±CLEANING_OCTAVE_WINDOW_S seconds and compute their median pitch.
    If the note is ≥CLEANING_OCTAVE_SHIFT_THRESHOLD semitones from that
    median, and shifting by ±12 semitones would bring it significantly
    closer to the median, apply the shift.

    Only notes below CLEANING_OCTAVE_CONF_GATE confidence are candidates —
    high-confidence notes are trusted as-is.  This preserves intentional
    wide-interval leaps while fixing the ML model's common octave mistakes.
    """
    import bisect

    if not notes:
        return notes

    # Build a sorted index for fast neighbour lookup — don't modify input order
    sorted_order = sorted(range(len(notes)), key=lambda i: notes[i]["start"])
    sorted_starts  = [notes[i]["start"]  for i in sorted_order]
    sorted_pitches = [notes[i]["pitch"]  for i in sorted_order]
    orig_to_sorted = {orig: si for si, orig in enumerate(sorted_order)}

    result    = list(notes)   # shallow copy — items replaced if corrected
    corrected = 0

    for orig_i, note in enumerate(notes):
        # High-confidence notes are trusted — skip
        if note["confidence"] >= CLEANING_OCTAVE_CONF_GATE:
            continue

        si = orig_to_sorted[orig_i]
        t  = note["start"]

        lo = bisect.bisect_left( sorted_starts, t - CLEANING_OCTAVE_WINDOW_S)
        hi = bisect.bisect_right(sorted_starts, t + CLEANING_OCTAVE_WINDOW_S)

        neighbour_pitches = [
            sorted_pitches[j] for j in range(lo, hi) if j != si
        ]
        if len(neighbour_pitches) < CLEANING_OCTAVE_MIN_NEIGHBORS:
            lo2 = bisect.bisect_left( sorted_starts, t - CLEANING_OCTAVE_WIDE_WINDOW_S)
            hi2 = bisect.bisect_right(sorted_starts, t + CLEANING_OCTAVE_WIDE_WINDOW_S)
            neighbour_pitches = [sorted_pitches[j] for j in range(lo2, hi2) if j != si]
            if len(neighbour_pitches) < CLEANING_OCTAVE_MIN_NEIGHBORS:
                continue   # still not enough context — skip

        median = sorted(neighbour_pitches)[len(neighbour_pitches) // 2]
        p      = note["pitch"]
        dist   = abs(p - median)

        if dist < CLEANING_OCTAVE_SHIFT_THRESHOLD:
            continue   # note is already close enough to the local median

        # Try ±12 shift — pick the one that reduces distance most
        best_shift = None
        best_dist  = dist
        for delta in (12, -12):
            candidate = p + delta
            if GUITAR_MIDI_MIN <= candidate <= GUITAR_MIDI_MAX:
                shifted_dist = abs(candidate - median)
                if shifted_dist < best_dist:
                    best_dist  = shifted_dist
                    best_shift = delta

        if best_shift is not None:
            result[orig_i] = dict(note, pitch=p + best_shift)
            corrected += 1

    if corrected:
        print(f"[Stage 5b] Octave correction: shifted {corrected} low-confidence notes "
              f"(>={CLEANING_OCTAVE_SHIFT_THRESHOLD} semitones from local context)")

    return result


def load_clean_meta() -> dict:
    path = os.path.join(get_instrument_dir("guitar"), "03_clean_meta.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ── Filter implementations ────────────────────────────────────────────────────

def _merge_bends(
    notes: list[dict],
    max_semitones: int = CLEANING_BEND_MAX_SEMITONES,
    max_gap_s: float = CLEANING_BEND_MAX_GAP_S,
    min_first_note_s: float = CLEANING_BEND_MIN_FIRST_NOTE_S,
) -> list[dict]:
    """
    Detect and merge guitar bends / vibrato.

    A bend shows up as two nearly-adjacent notes close in pitch (the model
    detects the pre-bend pitch then the bent pitch).  Conditions to merge:
      - gap between the two notes <= max_gap_s  (tight — true bends are ~0ms apart)
      - pitch distance == 1 or 2 semitones
      - first note duration >= min_first_note_s  (a 30ms note can't be a bend)

    Result:
      - pitch  = lower of the two (the fretted note before the bend)
      - start  = earlier start
      - duration = span from first start to last end
      - confidence = max of the two

    Notes are sorted by start time before processing.
    Only consecutive note pairs are tested (not all pairs).
    """
    if len(notes) < 2:
        return notes

    notes = sorted(notes, key=lambda n: n["start"])
    merged = [dict(notes[0])]

    for nxt in notes[1:]:
        cur = merged[-1]
        cur_end = cur["start"] + cur["duration"]
        gap     = nxt["start"] - cur_end
        semitone_dist = abs(nxt["pitch"] - cur["pitch"])

        if (gap <= max_gap_s
                and 1 <= semitone_dist <= max_semitones
                and cur["duration"] >= min_first_note_s):
            # Merge: keep lower pitch, extend duration, take max confidence
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


def _filter_isolated_notes(
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


def _limit_polyphony(notes: list[dict], max_poly: int) -> list[dict]:
    """
    Sweep through time. Whenever >max_poly notes are simultaneously active,
    evict the note with the highest eviction score.

    Smart eviction (CLEANING_POLY_ROOT_PROTECTION=True):
      - Protects the lowest-pitch note in any cluster (likely the chord root).
      - Penalises notes far above the cluster median (likely overtones).
    Legacy fallback: lowest confidence first, then highest pitch.
    """
    if not notes:
        return notes

    events = []
    for i, note in enumerate(notes):
        events.append((note["start"],                    0, i))
        events.append((note["start"] + note["duration"], 1, i))
    events.sort(key=lambda e: (e[0], e[1]))

    active    = {}
    to_remove = set()

    def _eviction_score(idx):
        note   = active[idx]
        conf   = float(note.get("confidence", 0.0))
        pitches = [active[j]["pitch"] for j in active]
        is_lowest = note["pitch"] == min(pitches)
        root_bonus = 0.5 if (CLEANING_POLY_ROOT_PROTECTION and is_lowest) else 0.0
        median_p   = sorted(pitches)[len(pitches) // 2]
        over       = max(0, note["pitch"] - median_p - CLEANING_POLY_HEIGHT_PENALTY_ST)
        height_pen = over * CLEANING_POLY_HEIGHT_PENALTY_COEF
        return (1.0 - conf) + height_pen - root_bonus

    for _time, event_type, idx in events:
        if idx in to_remove:
            continue
        if event_type == 0:
            active[idx] = notes[idx]
            while len(active) > max_poly:
                worst = max(active, key=_eviction_score)
                to_remove.add(worst)
                del active[worst]
        else:
            active.pop(idx, None)

    return [n for i, n in enumerate(notes) if i not in to_remove]


def _local_pitch_filter(
    notes: list[dict],
    window_s: float,
    max_deviation: int,
    min_context: int = 6,
) -> list[dict]:
    """
    Remove notes whose pitch is far from the local median pitch.

    For each note at time t, collect all other notes within [t-window_s, t+window_s]
    and compute their median pitch. If this note deviates more than max_deviation
    semitones from that median, it is removed.

    Skipped when fewer than min_context notes are in the window (sparse passages
    where a "local median" is not meaningful).
    """
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


def _merge_nearby(notes: list[dict], gap_threshold: float, merge_ratio: float = 0.0) -> list[dict]:
    """
    Merge consecutive notes of the same pitch separated by a small gap.

    Two conditions (either triggers a merge):
      1. gap <= gap_threshold  — absolute time-based catch for tiny gaps
      2. gap <= merge_ratio * min(dur_current, dur_next)  — ratio-based catch for
         long sustained notes with small detection dips.

    The ratio condition deliberately ignores short repeated notes: a 50ms gap
    between two 60ms notes has a ratio of 83% and won't merge at ratio=0.25,
    while the same 50ms gap inside a 500ms sustain has a ratio of 10% and will.
    """
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
            gap = nxt["start"] - current_end
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


def _load_bpm() -> float | None:
    """Try to load BPM from a previous quantization run."""
    path = os.path.join(get_shared_dir(), "04_tempo.json")
    try:
        with open(path) as f:
            return json.load(f).get("bpm")
    except FileNotFoundError:
        return None


def load_cleaned_notes() -> list[dict]:
    path = os.path.join(get_instrument_dir("guitar"), "03_cleaned_notes.json")
    with open(path) as f:
        return json.load(f)
