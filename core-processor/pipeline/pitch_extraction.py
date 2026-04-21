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

from pipeline.config import get_outputs_dir
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
)

BP_FRAMES_PER_SEC = BP_SAMPLE_RATE / BP_HOP_LENGTH   # ~86.1 fps


def extract_pitches(
    audio_path: str,
    guitar_type: str = "rhythm",
    save: bool = True,
) -> list[dict]:
    """
    Extract notes from audio.
    Loads stem_confidence from Stage 1 metadata to tune thresholds.
    guitar_type: "lead" | "acoustic" | "rhythm"
    Returns list of {pitch, start, duration, confidence}.
    """
    try:
        return _extract_basic_pitch(audio_path, guitar_type=guitar_type, save=save)
    except ImportError as e:
        print(f"[Stage 2] basic-pitch not available ({e}), falling back to pyin")
        return _extract_pyin(audio_path, save=save)


# ── basic-pitch backend ───────────────────────────────────────────────────────

def _extract_basic_pitch(audio_path: str, guitar_type: str = "rhythm", save: bool = True) -> list[dict]:
    from basic_pitch.inference import run_inference, AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.inference import infer as bp_infer
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from pipeline.separation import load_stem_meta

    stem_conf = load_stem_meta().get("stem_confidence", 0.5)
    print(f"[Stage 2] Loading audio: {audio_path}")
    print(f"[Stage 2] stem_confidence={stem_conf:.2f}  type={guitar_type}")

    # Lead guitar uses a higher ceiling to capture bends at the highest frets
    midi_max = GUITAR_MIDI_MAX_LEAD if guitar_type == "lead" else GUITAR_MIDI_MAX
    hz_max   = GUITAR_HZ_MAX_LEAD   if guitar_type == "lead" else GUITAR_HZ_MAX

    # Determine pass configurations
    pass_configs = PITCH_MULTI_PASS_CONFIGS.get(guitar_type)
    if pass_configs is None:
        # Single adaptive pass (rhythm)
        onset, frame, min_ms = _resolve_thresholds(guitar_type, stem_conf)
        pass_configs = [(onset, frame, min_ms)]

    n_passes = len(pass_configs)

    # Run model inference ONCE — all passes share the same neural network output.
    # Previously predict() was called N times, re-running the full network each time.
    print(f"[Stage 2] Running model inference...")
    model_output = run_inference(audio_path, ICASSP_2022_MODEL_PATH)
    note_arr = model_output.get("note")

    all_pass_notes = []    # list[list[dict]]

    for pass_idx, (onset_t, frame_t, min_ms) in enumerate(pass_configs):
        print(f"[Stage 2] Pass {pass_idx + 1}/{n_passes}:  "
              f"onset={onset_t}  frame={frame_t}  min_note={min_ms}ms")

        min_note_len = int(np.round(min_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
        # melodia_trick suppresses notes it considers harmonically overshadowed by a
        # stronger neighbour. For lead guitar this kills adjacent-semitone bend notes
        # (e.g. MIDI 84 and 86 next to a dominant MIDI 85). Disable it for lead so
        # every detected pitch survives; the multi-pass confidence merge handles quality.
        melodia = guitar_type != "lead"
        midi_data, _ = bp_infer.model_output_to_notes(
            model_output,
            onset_thresh=onset_t,
            frame_thresh=frame_t,
            min_note_len=min_note_len,
            min_freq=GUITAR_HZ_MIN,
            max_freq=hz_max,
            melodia_trick=melodia,
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
    _save_pass_preview(notes, "merged_raw")

    if save:
        _save(notes)
    return notes


def _resolve_thresholds(guitar_type: str, stem_conf: float) -> tuple[float, float, int]:
    """Return (onset_threshold, frame_threshold, min_note_ms) for the given type."""
    row = PITCH_THRESHOLDS.get(guitar_type, PITCH_THRESHOLDS["rhythm"])
    onset, frame, min_ms = row

    if onset is None:
        # Compute from stem_confidence scaling
        if guitar_type == "acoustic":
            onset = round(PITCH_ACOUSTIC_ONSET_BASE - stem_conf * PITCH_ACOUSTIC_ONSET_SCALE, 2)
            frame = round(PITCH_ACOUSTIC_FRAME_BASE - stem_conf * PITCH_ACOUSTIC_FRAME_SCALE, 2)
        else:  # rhythm (and any unknown type)
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
        min_dist = min(abs(p - nb) % 12 for nb in neighbours)

        if min_dist > 5:
            for shift in (12, -12):
                shifted = p + shift
                if not (GUITAR_MIDI_MIN <= shifted <= GUITAR_MIDI_MAX):
                    continue
                shifted_dist = min(abs(shifted - nb) % 12 for nb in neighbours)
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


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _save_pass_preview(notes: list[dict], pass_num: int | str) -> None:
    """Synthesize and save a per-pass preview WAV to the outputs directory."""
    import soundfile as sf
    from pipeline.audio_playback import synthesize_notes
    from pipeline.config import get_outputs_dir

    if isinstance(pass_num, int):
        filename = f"02_pass{pass_num}_preview.wav"
        tag = f"pass {pass_num}"
    else:
        filename = f"02_{pass_num}_preview.wav"
        tag = pass_num.replace("_", " ")

    if not notes:
        return

    buffer = synthesize_notes(notes)
    os.makedirs(get_outputs_dir(), exist_ok=True)
    out_path = os.path.join(get_outputs_dir(), filename)
    sf.write(out_path, buffer, 44100, subtype="PCM_16")
    print(f"[Stage 2] Preview ({tag}: {len(notes)} notes) -> {out_path}")


def _save(notes: list[dict]) -> None:
    os.makedirs(get_outputs_dir(), exist_ok=True)
    out_path = os.path.join(get_outputs_dir(), "02_raw_notes.json")
    with open(out_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"[Stage 2] Saved -> {out_path}")

def load_raw_notes() -> list[dict]:
    path = os.path.join(get_outputs_dir(), "02_raw_notes.json")
    with open(path) as f:
        return json.load(f)
