"""
Stage 2: Pitch Extraction
Input:  audio file (guitar stem or raw mix)
Output: raw notes [{pitch, start, duration, confidence}]
        saved to outputs/02_raw_notes.json

Backend: basic-pitch (Spotify) — polyphonic ML model, CUDA-capable.
Fallback: librosa pyin — monophonic, used if basic-pitch unavailable.

Quality improvements:
  - onset/frame thresholds scale with stem_confidence from Stage 1
    (weak stem → stricter thresholds to reject noise)
  - confidence = mean frame-level detection probability from model_output["note"]
    instead of velocity/127 (a much more meaningful signal)
  - Octave error correction: notes that are out-of-context octave-wise are shifted
"""

import json
import os
import numpy as np

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    GUITAR_MIDI_MIN,
    GUITAR_MIDI_MAX,
    GUITAR_MIDI_MAX_LEAD,
    GUITAR_HZ_MIN,
    GUITAR_HZ_MAX,
    GUITAR_HZ_MAX_LEAD,
    BP_SAMPLE_RATE,
    BP_HOP_LENGTH,
    BP_MIDI_OFFSET,
    PITCH_THRESHOLDS,
    PITCH_ACOUSTIC_ONSET_BASE,
    PITCH_ACOUSTIC_ONSET_SCALE,
    PITCH_ACOUSTIC_FRAME_BASE,
    PITCH_ACOUSTIC_FRAME_SCALE,
    PITCH_RHYTHM_ONSET_BASE,
    PITCH_RHYTHM_ONSET_SCALE,
    PITCH_RHYTHM_FRAME_BASE,
    PITCH_RHYTHM_FRAME_SCALE,
    PITCH_PYIN_MIN_NOTE_DURATION_S,
    PITCH_MULTI_PASS_CONFIGS,
    PITCH_MERGE_PROXIMITY_S,
    PITCH_CONF_WEIGHT_BASE,
    PITCH_CONF_WEIGHT_CONFIRMED,
    PITCH_CONF_WEIGHT_STRONG,
    CREPE_ENABLED,
    CREPE_MODEL,
    CREPE_FMAX,
    CREPE_CONF_THRESHOLDS,
    CREPE_MIN_NOTE_S,
    CREPE_MAX_GAP_S,
    CREPE_PITCH_TOLERANCE,
    CREPE_REPET_ENABLED,
    CREPE_REPET_AUTO_THRESHOLD,
    CREPE_HOP_LENGTH,
    CREPE_FMAX_PER_TYPE,
    PITCH_OCTAVE_ERROR_THRESHOLD_ST,
    PITCH_SUSTAINED_BOOST_MIN_S,
    PITCH_SUSTAINED_BOOST_AMOUNT,
    PITCH_SUSTAINED_BOOST_CAP,
    PITCH_HIGH_NOTE_RECOVERY_MIDI,
    PITCH_HIGH_NOTE_RECOVERY_HZ,
    PITCH_HIGH_NOTE_RECOVERY_ONSET,
    PITCH_HIGH_NOTE_RECOVERY_FRAME,
    PITCH_HIGH_NOTE_RECOVERY_MIN_MS,
    PITCH_HIGH_NOTE_RECOVERY_MIN_CONF,
    PITCH_HIGH_NOTE_RECOVERY_ONSET_FLOOR,
    PITCH_HIGH_NOTE_RECOVERY_FRAME_FLOOR,
    PITCH_HIGH_NOTE_RECOVERY_PITCH_ZERO,
    PITCH_HIGH_NOTE_RECOVERY_PITCH_FULL,
    PITCH_HIGH_NOTE_RECOVERY_ONSET_GATE,
)

BP_FRAMES_PER_SEC = BP_SAMPLE_RATE / BP_HOP_LENGTH   # ~86.1 fps


def extract_pitches(
    audio_path: str,
    guitar_type: str = "clean",
    guitar_role: str = "rhythm",
    save: bool = True,
) -> list[dict]:
    """
    Extract notes from audio.
    Loads stem_confidence from Stage 1 metadata to tune thresholds.
    guitar_type: "acoustic" | "clean" | "distorted"
    guitar_role: "lead" | "rhythm"
    Returns list of {pitch, start, duration, confidence}.
    """
    from pipeline.shared.presence import check_and_update_stem_presence
    if not check_and_update_stem_presence("guitar"):
        if save:
            _save([])
        return []

    try:
        return _extract_basic_pitch(audio_path, guitar_type=guitar_type, guitar_role=guitar_role, save=save)
    except ImportError as e:
        print(f"[Stage 2] basic-pitch not available ({e}), falling back to pyin")
        return _extract_pyin(audio_path, save=save)


# ── basic-pitch backend ───────────────────────────────────────────────────────

def _extract_basic_pitch(audio_path: str, guitar_type: str = "clean", guitar_role: str = "rhythm", save: bool = True) -> list[dict]:
    from basic_pitch.inference import run_inference, AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.inference import infer as bp_infer
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from pipeline.shared.separation import load_stem_meta, gate_notes_by_stem_energy
    from pipeline.settings import (
        PITCH_DISTORTED_HPSS_MARGIN,
        PITCH_DISTORTED_HPSS_MARGIN_WEAK,
        PITCH_DISTORTED_HPSS_MARGIN_STRONG,
        PITCH_DISTORTED_HPSS_STEM_THRESH_WEAK,
        PITCH_DISTORTED_HPSS_STEM_THRESH_STRONG,
    )

    stem_conf = load_stem_meta().get("stem_confidence", 0.5)
    mode_key  = f"{guitar_type}_{guitar_role}"
    print(f"[Stage 2] Loading audio: {audio_path}")
    print(f"[Stage 2] stem_confidence={stem_conf:.2f}  mode={mode_key}")

    # HPSS pre-processing for distorted guitar — isolate harmonic component
    # before basic-pitch so boosted distortion harmonics don't flood detections.
    # Margin is adaptive: weak stems get more aggressive harmonic isolation.
    _hpss_tmp = None
    if guitar_type == "distorted":
        import librosa, tempfile, soundfile as sf
        if stem_conf < PITCH_DISTORTED_HPSS_STEM_THRESH_WEAK:
            hpss_margin = PITCH_DISTORTED_HPSS_MARGIN_WEAK
        elif stem_conf > PITCH_DISTORTED_HPSS_STEM_THRESH_STRONG:
            hpss_margin = PITCH_DISTORTED_HPSS_MARGIN_STRONG
        else:
            # Linear interpolation between weak and strong
            t = ((stem_conf - PITCH_DISTORTED_HPSS_STEM_THRESH_WEAK)
                 / (PITCH_DISTORTED_HPSS_STEM_THRESH_STRONG - PITCH_DISTORTED_HPSS_STEM_THRESH_WEAK))
            hpss_margin = PITCH_DISTORTED_HPSS_MARGIN_WEAK + t * (
                PITCH_DISTORTED_HPSS_MARGIN_STRONG - PITCH_DISTORTED_HPSS_MARGIN_WEAK)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        y_harmonic, _ = librosa.effects.hpss(y, margin=hpss_margin)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y_harmonic, sr)
        audio_path_for_bp = tmp.name
        _hpss_tmp = tmp.name
        print(f"[Stage 2] Distorted guitar: HPSS applied (margin={hpss_margin:.2f}, stem_conf={stem_conf:.2f})")
    else:
        audio_path_for_bp = audio_path

    try:
        # Lead role uses a higher ceiling to capture bends at the highest frets
        midi_max = GUITAR_MIDI_MAX_LEAD if guitar_role == "lead" else GUITAR_MIDI_MAX
        hz_max   = GUITAR_HZ_MAX_LEAD   if guitar_role == "lead" else GUITAR_HZ_MAX

        # Determine pass configurations
        pass_configs = PITCH_MULTI_PASS_CONFIGS.get(mode_key)
        if pass_configs is None:
            # Single adaptive pass fallback
            onset, frame, min_ms = _resolve_thresholds(guitar_type, guitar_role, stem_conf)
            pass_configs = [(onset, frame, min_ms)]

        n_passes = len(pass_configs)

        # Run model inference ONCE — all passes share the same neural network output.
        # Previously predict() was called N times, re-running the full network each time.
        print(f"[Stage 2] Running model inference...")
        model_output = run_inference(audio_path_for_bp, ICASSP_2022_MODEL_PATH)
        note_arr = model_output.get("note")

        all_pass_notes = []    # list[list[dict]]

        for pass_idx, (onset_t, frame_t, min_ms) in enumerate(pass_configs):
            print(f"[Stage 2] Pass {pass_idx + 1}/{n_passes}:  "
                  f"onset={onset_t}  frame={frame_t}  min_note={min_ms}ms")

            min_note_len = int(np.round(min_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
            # melodia_trick suppresses notes it considers harmonically overshadowed by a
            # stronger neighbour. Designed for monophonic melodies; for guitar chords it
            # kills chord tones (e.g. G and B in an Em chord are suppressed by the root E).
            # Disabled for all guitar types — polyphony is handled downstream by the
            # multi-pass confidence merge and the cleaning polyphony limit.
            midi_data, _ = bp_infer.model_output_to_notes(
                model_output,
                onset_thresh=onset_t,
                frame_thresh=frame_t,
                min_note_len=min_note_len,
                min_freq=GUITAR_HZ_MIN,
                max_freq=hz_max,
                melodia_trick=False,
            )

            pass_notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    pitch = int(note.pitch)
                    if not (GUITAR_MIDI_MIN <= pitch <= midi_max):
                        continue
                    confidence = _frame_confidence(note_arr, note.start, note.end, pitch)
                    pass_notes.append({
                        "pitch":      pitch,
                        "start":      round(float(note.start), 4),
                        "duration":   round(float(note.end - note.start), 4),
                        "confidence": round(confidence, 4),
                    })

            pass_notes.sort(key=lambda n: n["start"])
            pass_notes = _correct_octave_errors(pass_notes)
            all_pass_notes.append(pass_notes)
            print(f"[Stage 2] Pass {pass_idx + 1}: {len(pass_notes)} notes")
            _save_pass_preview(pass_notes, pass_idx + 1)

        # Merge all passes into a single deduplicated note list
        notes = _merge_passes(all_pass_notes, note_arr)
        notes.sort(key=lambda n: n["start"])

        total_raw = sum(len(p) for p in all_pass_notes)
        print(f"[Stage 2] Merged {total_raw} notes across {n_passes} pass(es) "
              f"-> {len(notes)} unique notes")

        # Stem energy gate: remove ghost notes from silent stem sections
        before = len(notes)
        notes  = gate_notes_by_stem_energy(notes)
        if before != len(notes):
            print(f"[Stage 2] Stem energy gate: removed {before - len(notes)} ghost notes "
                  f"from silent stem regions  ({len(notes)} remaining)")

        # High-note recovery: re-run model_output_to_notes at low thresholds
        # for pitches >= PITCH_HIGH_NOTE_RECOVERY_MIDI to catch notes the main
        # passes missed due to confidence bias in the upper register.
        notes = _recover_high_notes(notes, model_output, hz_max)

        # CREPE overlay for lead role — monophonic CNN on top of basic-pitch.
        # Distorted guitar uses the HPSS harmonic component (same as basic-pitch)
        # so CREPE gets a cleaner signal; clean/acoustic use the original stem.
        if guitar_role == "lead" and CREPE_ENABLED:
            crepe_input = audio_path_for_bp if guitar_type == "distorted" else audio_path
            auto_repet  = stem_conf < CREPE_REPET_AUTO_THRESHOLD
            use_repet   = CREPE_REPET_ENABLED or auto_repet
            if auto_repet and not CREPE_REPET_ENABLED:
                print(f"[Stage 2] Auto-enabling REPET-SIM (stem_confidence={stem_conf:.2f} < {CREPE_REPET_AUTO_THRESHOLD})")
            try:
                crepe_notes = _extract_crepe(crepe_input, midi_max, guitar_type, use_repet=use_repet)
                notes = _merge_crepe_with_basic_pitch(crepe_notes, notes)
            except ImportError:
                print("[Stage 2] torchcrepe not installed — skipping CREPE overlay")
            except Exception as exc:
                print(f"[Stage 2] CREPE failed ({exc}) — using basic-pitch only")

        _save_pass_preview(notes, "merged_raw")

        _apply_sustained_boost(notes, PITCH_SUSTAINED_BOOST_MIN_S,
                               PITCH_SUSTAINED_BOOST_AMOUNT, PITCH_SUSTAINED_BOOST_CAP)

        if save:
            _save(notes)
        return notes
    finally:
        if _hpss_tmp is not None:
            try:
                os.unlink(_hpss_tmp)
            except OSError:
                pass


def _resolve_thresholds(guitar_type: str, guitar_role: str, stem_conf: float) -> tuple[float, float, int]:
    """Return (onset_threshold, frame_threshold, min_note_ms) for the given type+role."""
    mode_key = f"{guitar_type}_{guitar_role}"
    row = PITCH_THRESHOLDS.get(mode_key, PITCH_THRESHOLDS["clean_rhythm"])
    onset, frame, min_ms = row

    if onset is None:
        # Compute from stem_confidence scaling
        if guitar_type == "acoustic":
            onset = round(PITCH_ACOUSTIC_ONSET_BASE - stem_conf * PITCH_ACOUSTIC_ONSET_SCALE, 2)
            frame = round(PITCH_ACOUSTIC_FRAME_BASE - stem_conf * PITCH_ACOUSTIC_FRAME_SCALE, 2)
        else:  # clean / distorted rhythm (and any unknown type)
            onset = round(PITCH_RHYTHM_ONSET_BASE - stem_conf * PITCH_RHYTHM_ONSET_SCALE, 2)
            frame = round(PITCH_RHYTHM_FRAME_BASE - stem_conf * PITCH_RHYTHM_FRAME_SCALE, 2)

    return onset, frame, min_ms


def _frame_confidence(note_arr, start_s: float, end_s: float, midi_pitch: int) -> float:
    """
    Extract mean frame-level detection probability for a note.
    Falls back to 0.5 if model_output is unavailable or pitch out of range.
    """
    if note_arr is None:
        return 0.5

    start_f   = int(start_s * BP_FRAMES_PER_SEC)
    end_f     = max(start_f + 1, int(end_s * BP_FRAMES_PER_SEC))
    pitch_idx = midi_pitch - BP_MIDI_OFFSET

    if not (0 <= pitch_idx < note_arr.shape[1]):
        return 0.5

    frames = note_arr[start_f:end_f, pitch_idx]
    return float(frames.mean()) if len(frames) > 0 else 0.5


def _correct_octave_errors(notes: list[dict]) -> list[dict]:
    """
    Basic-pitch occasionally places a note one octave too high or low.
    For each note, if its pitch is >=12 semitones away from all neighbours
    within +/-1 second, but the octave-shifted version is within 2 semitones
    of at least one neighbour, shift it.

    Uses bisect for O(n log n) neighbour lookup instead of O(n^2).
    """
    import bisect

    if len(notes) < 3:
        return notes

    corrected = list(notes)
    starts = [n["start"] for n in corrected]   # already sorted (notes are sorted by start)

    for i, note in enumerate(corrected):
        t = note["start"]
        lo = bisect.bisect_left(starts, t - 1.0)
        hi = bisect.bisect_right(starts, t + 1.0)

        neighbours = [
            corrected[j]["pitch"]
            for j in range(lo, hi)
            if j != i
        ]
        if not neighbours:
            continue

        p = note["pitch"]
        min_dist = min(abs(p - nb) for nb in neighbours)

        if min_dist > PITCH_OCTAVE_ERROR_THRESHOLD_ST:
            for shift in (12, -12):
                shifted = p + shift
                if not (GUITAR_MIDI_MIN <= shifted <= GUITAR_MIDI_MAX):
                    continue
                shifted_dist = min(abs(shifted - nb) for nb in neighbours)
                if shifted_dist < min_dist:
                    corrected[i] = dict(note, pitch=shifted)
                    break

    return corrected


# ── Multi-pass merge ─────────────────────────────────────────────────────────

def _merge_passes(all_pass_notes: list[list[dict]], note_arr) -> list[dict]:
    """
    Confidence-boost merge: passes run from base (least strict) to ultra-strict.

    Only base-pass notes are kept — stricter passes NEVER add new notes.
    If a base note is also detected by 1 or more stricter passes, its
    confidence weight is boosted, making it more likely to survive cleaning.

    Confidence = frame_prob x weight:
      - base only         -> PITCH_CONF_WEIGHT_BASE      (0.80)
      - confirmed by 1    -> PITCH_CONF_WEIGHT_CONFIRMED  (0.92)
      - confirmed by all  -> PITCH_CONF_WEIGHT_STRONG     (1.00)

    Pass order in PITCH_MULTI_PASS_CONFIGS: [base, very-strict, ultra-strict, ...]
    Falls back to flat merge for single-pass configs.
    """
    import bisect

    if len(all_pass_notes) == 1:
        return _flat_merge(all_pass_notes, note_arr)

    base_notes      = all_pass_notes[0]   # least strict — all notes come from here
    stricter_passes = all_pass_notes[1:]  # each is a subset of the previous

    # Build sorted (pitch, start) lookup for each stricter pass
    def _build_lookup(notes: list[dict]) -> list[tuple]:
        return sorted((n["pitch"], n["start"]) for n in notes)

    stricter_lookups = [_build_lookup(p) for p in stricter_passes]

    def _is_confirmed(note: dict, lookup: list[tuple]) -> bool:
        """True if lookup contains the same pitch within PITCH_MERGE_PROXIMITY_S."""
        pitch = note["pitch"]
        t     = note["start"]
        lo = bisect.bisect_left( lookup, (pitch, t - PITCH_MERGE_PROXIMITY_S))
        hi = bisect.bisect_right(lookup, (pitch, t + PITCH_MERGE_PROXIMITY_S))
        return any(lookup[i][0] == pitch for i in range(lo, hi))

    confirmed_1   = 0
    confirmed_all = 0
    result = []

    for note in sorted(base_notes, key=lambda n: n["start"]):
        confirmations = sum(
            1 for lk in stricter_lookups if _is_confirmed(note, lk)
        )

        if confirmations >= len(stricter_passes):
            weight = PITCH_CONF_WEIGHT_STRONG
            confirmed_all += 1
        elif confirmations >= 1:
            weight = PITCH_CONF_WEIGHT_CONFIRMED
            confirmed_1 += 1
        else:
            weight = PITCH_CONF_WEIGHT_BASE

        ns = note["start"]
        ne = ns + note["duration"]
        frame_conf = _frame_confidence(note_arr, ns, ne, note["pitch"])
        result.append({
            "pitch":      note["pitch"],
            "start":      note["start"],
            "duration":   note["duration"],
            "confidence": round(frame_conf * weight, 4),
        })

    base_only = len(base_notes) - confirmed_1 - confirmed_all
    print(f"[Stage 2] Confidence merge: {confirmed_all} strong "
          f"+ {confirmed_1} confirmed + {base_only} base-only "
          f"= {len(result)} notes")

    return result


def _flat_merge(all_pass_notes: list[list[dict]], note_arr) -> list[dict]:
    """Simple deduplicating merge used for single-pass configs."""
    import bisect

    pool = []
    for pass_notes in all_pass_notes:
        pool.extend(pass_notes)
    pool.sort(key=lambda n: (n["pitch"], n["start"]))

    if not pool:
        return []

    merged = [dict(pool[0])]
    for note in pool[1:]:
        prev     = merged[-1]
        prev_end = prev["start"] + prev["duration"]
        if note["pitch"] == prev["pitch"] and note["start"] <= prev_end + PITCH_MERGE_PROXIMITY_S:
            new_end = max(prev_end, note["start"] + note["duration"])
            merged[-1]["duration"] = round(new_end - prev["start"], 4)
        else:
            merged.append(dict(note))

    result = []
    for note in merged:
        ns = note["start"]
        ne = ns + note["duration"]
        frame_conf = _frame_confidence(note_arr, ns, ne, note["pitch"])
        note["confidence"] = round(frame_conf, 4)
        result.append(note)
    return result


# ── pyin fallback backend ─────────────────────────────────────────────────────

def _extract_pyin(audio_path: str, save: bool = True) -> list[dict]:
    import librosa

    print(f"[Stage 2] Loading audio: {audio_path}")
    y, sr = load_audio(audio_path, target_sr=22050)
    print(f"[Stage 2] Audio loaded — {len(y)/sr:.1f}s @ {sr}Hz")
    print("[Stage 2] Running pyin (monophonic fallback)...")

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, sr=sr, fmin=GUITAR_HZ_MIN, fmax=GUITAR_HZ_MAX,
        frame_length=2048, hop_length=512,
    )

    notes = _frames_to_notes(f0, voiced_flag, voiced_prob, hop_s=512/sr)
    print(f"[Stage 2] pyin extracted {len(notes)} notes in guitar range")

    if save:
        _save(notes)
    return notes


# ── CREPE monophonic overlay (lead role only) ─────────────────────────────────

def _extract_crepe(audio_path: str, midi_max: int, guitar_type: str = "clean", use_repet: bool = False) -> list[dict]:
    """
    Run torchcrepe monophonic pitch tracking on the stem.

    Only frames with periodicity >= conf_thresh produce notes — polyphonic
    and noisy frames naturally score low, acting as a chord suppressor.
    fmax is capped at CREPE_FMAX (1760 Hz) because the CREPE model's
    internal frequency grid tops out near 1975 Hz; values above that cause
    periodicity to collapse to ~0 (all frames appear unvoiced).

    Optional: set CREPE_REPET_ENABLED=True to run REPET-SIM first.
    """
    import torchcrepe
    import librosa
    import torch

    conf_thresh = CREPE_CONF_THRESHOLDS.get(guitar_type, CREPE_CONF_THRESHOLDS["clean"])
    fmax        = CREPE_FMAX_PER_TYPE.get(guitar_type, CREPE_FMAX)

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    if use_repet:
        S_full   = np.abs(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine")
        S_filter = np.minimum(S_full, S_filter)
        mask     = librosa.util.softmask(S_full - S_filter, 2.0 * S_filter, power=2)
        phase    = np.exp(1j * np.angle(librosa.stft(y)))
        y        = np.real(librosa.istft(mask * S_full * phase))
        print("[Stage 2] CREPE: REPET-SIM foreground separation applied")

    hop_length_samples = CREPE_HOP_LENGTH
    hop_s = hop_length_samples / sr

    _crepe_device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tensor = torch.tensor(y[None], dtype=torch.float32)
    frequency, confidence = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=hop_length_samples,
        fmin=GUITAR_HZ_MIN,
        fmax=fmax,
        model=CREPE_MODEL,
        return_periodicity=True,
        decoder=torchcrepe.decode.viterbi,
        device=_crepe_device,
    )

    freq_arr = frequency.squeeze(0).numpy()
    conf_arr = confidence.squeeze(0).numpy()

    notes         = []
    current_pitch = None
    current_start = None
    current_probs = []
    last_voiced_t = None

    for i, (freq, conf) in enumerate(zip(freq_arr, conf_arr)):
        t = i * hop_s
        is_voiced = (
            conf >= conf_thresh
            and freq > 0
            and not np.isnan(freq)
            and GUITAR_HZ_MIN <= freq <= GUITAR_HZ_MAX_LEAD
        )
        if is_voiced:
            midi = _hz_to_midi(freq)
            if not (GUITAR_MIDI_MIN <= midi <= midi_max):
                is_voiced = False

        if is_voiced:
            midi = _hz_to_midi(freq)
            gap  = (t - last_voiced_t) if last_voiced_t is not None else 0.0
            if midi == current_pitch and gap <= CREPE_MAX_GAP_S:
                current_probs.append(float(conf))
            else:
                if current_pitch is not None:
                    _finish_note(notes, current_pitch, current_start,
                                 last_voiced_t + hop_s, current_probs)
                current_pitch = midi
                current_start = t
                current_probs = [float(conf)]
            last_voiced_t = t
        else:
            if current_pitch is not None:
                gap = (t - last_voiced_t) if last_voiced_t is not None else 0.0
                if gap > CREPE_MAX_GAP_S:
                    _finish_note(notes, current_pitch, current_start,
                                 last_voiced_t + hop_s, current_probs)
                    current_pitch = None
                    current_start = None
                    current_probs = []
                    last_voiced_t = None

    if current_pitch is not None and current_start is not None:
        _finish_note(notes, current_pitch, current_start,
                     len(freq_arr) * hop_s, current_probs)

    notes = [n for n in notes if n["duration"] >= CREPE_MIN_NOTE_S]
    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 2] CREPE: {len(notes)} notes (conf>={conf_thresh}, type={guitar_type})")
    return notes


def _merge_crepe_with_basic_pitch(crepe_notes: list[dict], bp_notes: list[dict]) -> list[dict]:
    """
    Merge CREPE notes (high-confidence monophonic) with basic-pitch notes.

    Strategy — additive-only, never suppressive:
    1. For each bp note that overlaps a CREPE note at the same pitch (±2 semitones),
       boost its confidence to max(bp_conf, crepe_conf).  bp notes are NEVER removed.
    2. CREPE notes that have no matching bp note in their interval are added as new
       notes (catches pitches basic-pitch missed entirely).

    This means CREPE can only improve or add — it never removes bp notes.
    Stage 3 polyphony / confidence filtering handles any remaining noise.
    """
    import bisect

    if not crepe_notes:
        return bp_notes

    # Index bp notes by start time for fast lookup
    bp_sorted = sorted(bp_notes, key=lambda n: n["start"])
    bp_starts  = [n["start"] for n in bp_sorted]

    boosted    = 0
    added      = 0
    bp_out     = [dict(n) for n in bp_sorted]   # mutable copy

    for cn in crepe_notes:
        c_s    = cn["start"]
        c_e    = c_s + cn["duration"]
        c_p    = cn["pitch"]
        c_conf = cn["confidence"]

        # Find ALL bp notes that overlap this CREPE note's span.
        # A CREPE note typically spans 1–2 seconds; many bp notes may fall inside it.
        lo = bisect.bisect_left(bp_starts,  c_s - 0.10)
        hi = bisect.bisect_right(bp_starts, c_e + 0.10)

        # Find the single highest-confidence bp note at the CREPE pitch.
        # Boosting ALL same-pitch bp notes in a long CREPE span promotes sustain/ring
        # artifacts that are correctly pitched but should stay low-confidence.
        best_idx  = None
        best_conf = -1.0
        for idx in range(lo, hi):
            bp   = bp_out[idx]
            bp_s = bp["start"]
            bp_e = bp_s + bp["duration"]
            overlap = max(0.0, min(c_e, bp_e) - max(c_s, bp_s))
            if overlap <= 0:
                continue
            # Pitch match within tolerance; basic-pitch must have some evidence (conf >= 0.10).
            # ±1 semitone catches bends/vibrato where the two models disagree by one bin.
            if abs(bp["pitch"] - c_p) <= CREPE_PITCH_TOLERANCE and bp["confidence"] >= 0.10:
                if bp["confidence"] > best_conf:
                    best_conf = bp["confidence"]
                    best_idx  = idx

        if best_idx is not None:
            if c_conf > bp_out[best_idx]["confidence"]:
                bp_out[best_idx] = dict(bp_out[best_idx], confidence=round(c_conf, 4))
                boosted += 1
        else:
            # CREPE detected a pitch basic-pitch never found in this span — add it.
            bp_out.append(dict(cn))
            bp_starts.append(c_s)
            added += 1

    bp_out.sort(key=lambda n: n["start"])
    print(f"[Stage 2] CREPE merge: boosted {boosted} bp notes, added {added} new "
          f"= {len(bp_out)} total")
    return bp_out


# ── Audio loading (public — reused by quantization) ──────────────────────────

def load_audio(audio_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    return _load_audio(audio_path, target_sr)


def _load_audio(audio_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    ext = os.path.splitext(audio_path)[1].lower()

    if ext in (".wav", ".flac", ".ogg"):
        import librosa
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return y, sr

    import av
    import librosa

    container = av.open(audio_path)
    stream    = next(s for s in container.streams if s.type == "audio")
    native_sr = stream.codec_context.sample_rate
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=native_sr)

    samples = []
    for frame in container.decode(stream):
        for out_frame in resampler.resample(frame):
            arr = out_frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            samples.append(arr.astype(np.float32))
    for out_frame in resampler.resample(None):
        arr = out_frame.to_ndarray()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        samples.append(arr.astype(np.float32))
    container.close()

    y_native = np.concatenate(samples)
    if native_sr != target_sr:
        import librosa
        y = librosa.resample(y_native, orig_sr=native_sr, target_sr=target_sr)
    else:
        y = y_native

    return y, target_sr


# ── pyin frame segmentation ───────────────────────────────────────────────────

def _hz_to_midi(freq_hz: float) -> int:
    return int(round(69 + 12 * np.log2(freq_hz / 440.0)))

def _frames_to_notes(f0, voiced_flag, voiced_prob, hop_s) -> list[dict]:
    notes = []
    current_pitch = None
    current_start = None
    current_probs = []

    for i, (freq, is_voiced, prob) in enumerate(zip(f0, voiced_flag, voiced_prob)):
        t = i * hop_s
        if is_voiced and freq is not None and not np.isnan(freq):
            midi = _hz_to_midi(freq)
            if not (GUITAR_MIDI_MIN <= midi <= GUITAR_MIDI_MAX):
                is_voiced = False

        if is_voiced and freq is not None and not np.isnan(freq):
            midi = _hz_to_midi(freq)
            if midi == current_pitch:
                current_probs.append(float(prob))
            else:
                if current_pitch is not None:
                    _finish_note(notes, current_pitch, current_start, t, current_probs)
                current_pitch = midi
                current_start = t
                current_probs = [float(prob)]
        else:
            if current_pitch is not None:
                _finish_note(notes, current_pitch, current_start, t, current_probs)
                current_pitch = None
                current_start = None
                current_probs = []

    if current_pitch is not None and current_start is not None:
        _finish_note(notes, current_pitch, current_start, len(f0) * hop_s, current_probs)

    notes = [n for n in notes if n["duration"] >= PITCH_PYIN_MIN_NOTE_DURATION_S]
    notes.sort(key=lambda n: n["start"])
    return notes

def _finish_note(notes, pitch, start, end, probs):
    notes.append({
        "pitch":      pitch,
        "start":      round(start, 4),
        "duration":   round(end - start, 4),
        "confidence": round(float(np.mean(probs)) if probs else 0.0, 4),
    })


# ── High-note recovery pass ──────────────────────────────────────────────────

def _adaptive_high_note_thresh(
    midi_pitch: int,
    base_onset: float,
    base_frame: float,
    onset_floor: float,
    frame_floor: float,
    pitch_zero: int,
    pitch_full: int,
) -> tuple[float, float]:
    """Scale detection thresholds linearly from base (at pitch_zero) to floor (at pitch_full)."""
    excess = max(0, midi_pitch - pitch_zero)
    scale  = max(0.0, 1.0 - excess / max(1, pitch_full - pitch_zero))
    return max(onset_floor, base_onset * scale), max(frame_floor, base_frame * scale)


def _recover_high_notes(notes: list[dict], model_output, hz_max: float) -> list[dict]:
    """
    Re-run model_output_to_notes at floor thresholds restricted to the high register
    (>= PITCH_HIGH_NOTE_RECOVERY_HZ), then apply adaptive per-pitch filtering and an
    onset gate to suppress overtone false positives.

    Only ADDS notes not already present — never removes or modifies existing ones.
    """
    from basic_pitch.inference import infer as bp_infer
    import bisect

    min_note_len = int(round(
        PITCH_HIGH_NOTE_RECOVERY_MIN_MS / 1000
        * (22050 / 512)   # AUDIO_SAMPLE_RATE / FFT_HOP
    ))

    try:
        # Use floor thresholds so all candidates reach the per-note filter below.
        midi_data, _ = bp_infer.model_output_to_notes(
            model_output,
            onset_thresh=PITCH_HIGH_NOTE_RECOVERY_ONSET_FLOOR,
            frame_thresh=PITCH_HIGH_NOTE_RECOVERY_FRAME_FLOOR,
            min_note_len=min_note_len,
            min_freq=PITCH_HIGH_NOTE_RECOVERY_HZ,
            max_freq=hz_max,
            melodia_trick=False,
        )
    except Exception as exc:
        print(f"[Stage 2] High-note recovery skipped ({exc})")
        return notes

    note_arr  = model_output.get("note")
    onset_arr = model_output.get("onset")
    existing  = sorted((n["pitch"], n["start"]) for n in notes)

    added = 0
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitch = int(note.pitch)
            if not (PITCH_HIGH_NOTE_RECOVERY_MIDI <= pitch <= GUITAR_MIDI_MAX_LEAD):
                continue
            start = round(float(note.start), 4)
            # Skip if an existing note at same pitch within 80ms already covers this
            lo = bisect.bisect_left(existing, (pitch, start - 0.080))
            hi = bisect.bisect_right(existing, (pitch, start + 0.080))
            if any(existing[i][0] == pitch for i in range(lo, hi)):
                continue
            conf = _frame_confidence(note_arr, note.start, note.end, pitch)
            # Global floor
            if conf < PITCH_HIGH_NOTE_RECOVERY_MIN_CONF:
                continue
            # Onset gate: require an onset spike in model_output["onset"] within ±2 frames
            if onset_arr is not None:
                pitch_idx   = pitch - BP_MIDI_OFFSET
                onset_frame = int(start * BP_FRAMES_PER_SEC)
                f_lo = max(0, onset_frame - 2)
                f_hi = min(onset_arr.shape[0], onset_frame + 3)
                if 0 <= pitch_idx < onset_arr.shape[1] and f_lo < f_hi:
                    peak_onset = float(onset_arr[f_lo:f_hi, pitch_idx].max())
                    if peak_onset < PITCH_HIGH_NOTE_RECOVERY_ONSET_GATE and conf < 0.70:
                        continue  # no onset spike and not very high confidence → likely overtone
            # Adaptive confidence gate: progressively lower threshold for higher notes
            _, adap_frame = _adaptive_high_note_thresh(
                pitch,
                PITCH_HIGH_NOTE_RECOVERY_ONSET,
                PITCH_HIGH_NOTE_RECOVERY_FRAME,
                PITCH_HIGH_NOTE_RECOVERY_ONSET_FLOOR,
                PITCH_HIGH_NOTE_RECOVERY_FRAME_FLOOR,
                PITCH_HIGH_NOTE_RECOVERY_PITCH_ZERO,
                PITCH_HIGH_NOTE_RECOVERY_PITCH_FULL,
            )
            if conf < adap_frame:
                continue
            notes.append({
                "pitch":      pitch,
                "start":      start,
                "duration":   round(float(note.end - note.start), 4),
                "confidence": round(conf, 4),
            })
            existing.append((pitch, start))
            existing.sort()
            added += 1

    if added:
        notes.sort(key=lambda n: n["start"])
        print(f"[Stage 2] High-note recovery: added {added} notes (MIDI>={PITCH_HIGH_NOTE_RECOVERY_MIDI})")
    return notes


# ── Post-processing helpers ───────────────────────────────────────────────────

def _apply_sustained_boost(
    notes: list[dict],
    min_s: float,
    amount: float,
    cap: float,
) -> None:
    boosted = 0
    for note in notes:
        if note["duration"] >= min_s:
            note["confidence"] = min(cap, note["confidence"] + amount)
            boosted += 1
    if boosted:
        print(f"[Stage 2] Sustained boost applied to {boosted} notes")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _save_pass_preview(notes: list[dict], pass_num: int | str) -> None:
    """Synthesize and save a per-pass preview WAV to the outputs directory."""
    import soundfile as sf
    from pipeline.shared.audio import synthesize_notes
    from pipeline.config import get_instrument_dir

    if isinstance(pass_num, int):
        filename = f"02_pass{pass_num}_preview.wav"
        tag = f"pass {pass_num}"
    else:
        filename = f"02_{pass_num}_preview.wav"
        tag = pass_num.replace("_", " ")

    if not notes:
        return

    buffer = synthesize_notes(notes)
    out_path = os.path.join(get_instrument_dir("guitar"), filename)
    sf.write(out_path, buffer, 44100, subtype="PCM_16")
    print(f"[Stage 2] Preview ({tag}: {len(notes)} notes) -> {out_path}")


def _save(notes: list[dict]) -> None:
    out_path = os.path.join(get_instrument_dir("guitar"), "02_raw_notes.json")
    with open(out_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"[Stage 2] Saved -> {out_path}")

def load_raw_notes() -> list[dict]:
    path = os.path.join(get_instrument_dir("guitar"), "02_raw_notes.json")
    with open(path) as f:
        return json.load(f)
