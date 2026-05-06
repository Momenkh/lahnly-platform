"""
Stage 2 (Piano): Pitch Extraction
Input:  piano stem WAV
Output: raw notes [{pitch, start, duration, confidence}]
        saved to outputs/02_piano_raw_notes.json
"""

import json
import os
import numpy as np

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    PIANO_MIDI_MIN,
    PIANO_MIDI_MAX,
    PIANO_HZ_MIN,
    PIANO_HZ_MAX,
    BP_SAMPLE_RATE,
    BP_HOP_LENGTH,
    BP_MIDI_OFFSET,
    PIANO_MULTI_PASS_CONFIGS,
    PIANO_PITCH_THRESHOLDS,
    PIANO_PITCH_ONSET_BASE,
    PIANO_PITCH_ONSET_SCALE,
    PIANO_PITCH_FRAME_BASE,
    PIANO_PITCH_FRAME_SCALE,
    PIANO_PITCH_MERGE_PROXIMITY_S,
    PIANO_CONF_WEIGHT_BASE,
    PIANO_CONF_WEIGHT_CONFIRMED,
    PIANO_CONF_WEIGHT_STRONG,
    PIANO_PYIN_MIN_NOTE_DURATION_S,
    PIANO_CREPE_ENABLED,
    PIANO_CREPE_MODEL,
    PIANO_CREPE_FMAX,
    PIANO_CREPE_CONF_THRESHOLDS,
    PIANO_CREPE_MIN_NOTE_S,
    PIANO_CREPE_MAX_GAP_S,
    PIANO_CREPE_PITCH_TOLERANCE,
    PIANO_CREPE_REPET_ENABLED,
    PIANO_SUSTAINED_BOOST_MIN_S,
    PIANO_SUSTAINED_BOOST_AMOUNT,
    PIANO_SUSTAINED_BOOST_CAP,
    PIANO_HIGH_NOTE_RECOVERY_MIDI,
    PIANO_HIGH_NOTE_RECOVERY_HZ,
    PIANO_HIGH_NOTE_RECOVERY_ONSET,
    PIANO_HIGH_NOTE_RECOVERY_FRAME,
    PIANO_HIGH_NOTE_RECOVERY_MIN_MS,
    PIANO_HIGH_NOTE_RECOVERY_MIN_CONF,
)

BP_FRAMES_PER_SEC = BP_SAMPLE_RATE / BP_HOP_LENGTH


def extract_pitches_piano(
    audio_path: str,
    piano_mode: str = "piano_chord",
    save: bool = True,
) -> list[dict]:
    """
    Extract notes from a piano stem.
    piano_mode: "piano_melody" | "piano_chord"
    Returns list of {pitch, start, duration, confidence}.
    """
    from pipeline.shared.presence import check_and_update_stem_presence
    if not check_and_update_stem_presence("piano"):
        if save:
            _save_piano([])
        return []

    try:
        return _extract_basic_pitch_piano(audio_path, piano_mode=piano_mode, save=save)
    except ImportError as e:
        print(f"[Stage 2P] basic-pitch not available ({e}), falling back to pyin")
        return _extract_pyin_piano(audio_path, save=save)


def _extract_basic_pitch_piano(
    audio_path: str,
    piano_mode: str = "piano_chord",
    save: bool = True,
) -> list[dict]:
    from basic_pitch.inference import run_inference, AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.inference import infer as bp_infer
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from pipeline.shared.separation import load_stem_meta_for, gate_notes_by_stem_energy_for

    stem_conf = load_stem_meta_for("piano").get("stem_confidence", 0.5)
    print(f"[Stage 2P] Loading audio: {audio_path}")
    print(f"[Stage 2P] stem_confidence={stem_conf:.2f}  mode={piano_mode}")

    pass_configs = PIANO_MULTI_PASS_CONFIGS.get(piano_mode)
    if pass_configs is None:
        onset, frame, min_ms = _resolve_thresholds_piano(piano_mode, stem_conf)
        pass_configs = [(onset, frame, min_ms)]

    n_passes = len(pass_configs)
    print(f"[Stage 2P] Running model inference...")
    model_output = run_inference(audio_path, ICASSP_2022_MODEL_PATH)
    note_arr     = model_output.get("note")

    all_pass_notes = []

    for pass_idx, (onset_t, frame_t, min_ms) in enumerate(pass_configs):
        print(f"[Stage 2P] Pass {pass_idx + 1}/{n_passes}:  "
              f"onset={onset_t}  frame={frame_t}  min_note={min_ms}ms")

        min_note_len = int(np.round(min_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
        midi_data, _ = bp_infer.model_output_to_notes(
            model_output,
            onset_thresh=onset_t,
            frame_thresh=frame_t,
            min_note_len=min_note_len,
            min_freq=PIANO_HZ_MIN,
            max_freq=PIANO_HZ_MAX,
            melodia_trick=False,
        )

        pass_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                pitch = int(note.pitch)
                if not (PIANO_MIDI_MIN <= pitch <= PIANO_MIDI_MAX):
                    continue
                confidence = _frame_confidence(note_arr, note.start, note.end, pitch)
                pass_notes.append({
                    "pitch":      pitch,
                    "start":      round(float(note.start), 4),
                    "duration":   round(float(note.end - note.start), 4),
                    "confidence": round(confidence, 4),
                })

        pass_notes.sort(key=lambda n: n["start"])
        all_pass_notes.append(pass_notes)
        print(f"[Stage 2P] Pass {pass_idx + 1}: {len(pass_notes)} notes")

    notes = _merge_passes_piano(all_pass_notes, note_arr)
    notes.sort(key=lambda n: n["start"])

    total_raw = sum(len(p) for p in all_pass_notes)
    print(f"[Stage 2P] Merged {total_raw} notes across {n_passes} pass(es) "
          f"-> {len(notes)} unique notes")

    before = len(notes)
    notes  = gate_notes_by_stem_energy_for(notes, "piano")
    if before != len(notes):
        print(f"[Stage 2P] Stem energy gate: removed {before - len(notes)} ghost notes "
              f"({len(notes)} remaining)")

    notes = _recover_high_notes_piano(notes, model_output)

    # CREPE overlay only for melody mode (monophonic lines)
    if piano_mode == "piano_melody" and PIANO_CREPE_ENABLED:
        try:
            crepe_notes = _extract_crepe_piano(audio_path, piano_mode)
            notes = _merge_crepe_with_basic_pitch_piano(crepe_notes, notes)
        except ImportError:
            print("[Stage 2P] torchcrepe not installed — skipping CREPE overlay")
        except Exception as exc:
            print(f"[Stage 2P] CREPE failed ({exc}) — using basic-pitch only")

    _apply_sustained_boost_piano(notes, PIANO_SUSTAINED_BOOST_MIN_S,
                                 PIANO_SUSTAINED_BOOST_AMOUNT, PIANO_SUSTAINED_BOOST_CAP)

    if save:
        _save_piano(notes)
    return notes


def _resolve_thresholds_piano(piano_mode: str, stem_conf: float) -> tuple[float, float, int]:
    row = PIANO_PITCH_THRESHOLDS.get(piano_mode, PIANO_PITCH_THRESHOLDS["piano_chord"])
    onset, frame, min_ms = row
    if onset is None:
        onset = round(PIANO_PITCH_ONSET_BASE - stem_conf * PIANO_PITCH_ONSET_SCALE, 2)
        frame = round(PIANO_PITCH_FRAME_BASE - stem_conf * PIANO_PITCH_FRAME_SCALE, 2)
    return onset, frame, min_ms


def _frame_confidence(note_arr, start_s: float, end_s: float, midi_pitch: int) -> float:
    if note_arr is None:
        return 0.5
    start_f   = int(start_s * BP_FRAMES_PER_SEC)
    end_f     = max(start_f + 1, int(end_s * BP_FRAMES_PER_SEC))
    pitch_idx = midi_pitch - BP_MIDI_OFFSET
    if not (0 <= pitch_idx < note_arr.shape[1]):
        return 0.5
    frames = note_arr[start_f:end_f, pitch_idx]
    return float(frames.mean()) if len(frames) > 0 else 0.5


def _merge_passes_piano(all_pass_notes: list[list[dict]], note_arr) -> list[dict]:
    import bisect

    if len(all_pass_notes) == 1:
        return _flat_merge_piano(all_pass_notes, note_arr)

    base_notes      = all_pass_notes[0]
    stricter_passes = all_pass_notes[1:]

    def _build_lookup(notes):
        return sorted((n["pitch"], n["start"]) for n in notes)

    stricter_lookups = [_build_lookup(p) for p in stricter_passes]

    def _is_confirmed(note, lookup):
        pitch = note["pitch"]
        t     = note["start"]
        lo = bisect.bisect_left( lookup, (pitch, t - PIANO_PITCH_MERGE_PROXIMITY_S))
        hi = bisect.bisect_right(lookup, (pitch, t + PIANO_PITCH_MERGE_PROXIMITY_S))
        return any(lookup[i][0] == pitch for i in range(lo, hi))

    confirmed_1 = confirmed_all = 0
    result = []

    for note in sorted(base_notes, key=lambda n: n["start"]):
        confirmations = sum(1 for lk in stricter_lookups if _is_confirmed(note, lk))
        if confirmations >= len(stricter_passes):
            weight = PIANO_CONF_WEIGHT_STRONG
            confirmed_all += 1
        elif confirmations >= 1:
            weight = PIANO_CONF_WEIGHT_CONFIRMED
            confirmed_1 += 1
        else:
            weight = PIANO_CONF_WEIGHT_BASE

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
    print(f"[Stage 2P] Confidence merge: {confirmed_all} strong "
          f"+ {confirmed_1} confirmed + {base_only} base-only "
          f"= {len(result)} notes")
    return result


def _flat_merge_piano(all_pass_notes: list[list[dict]], note_arr) -> list[dict]:
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
        if note["pitch"] == prev["pitch"] and note["start"] <= prev_end + PIANO_PITCH_MERGE_PROXIMITY_S:
            new_end = max(prev_end, note["start"] + note["duration"])
            merged[-1]["duration"] = round(new_end - prev["start"], 4)
        else:
            merged.append(dict(note))

    result = []
    for note in merged:
        ns = note["start"]
        ne = ns + note["duration"]
        note["confidence"] = round(_frame_confidence(note_arr, ns, ne, note["pitch"]), 4)
        result.append(note)
    return result


def _extract_pyin_piano(audio_path: str, save: bool = True) -> list[dict]:
    import librosa

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, sr=sr, fmin=PIANO_HZ_MIN, fmax=min(PIANO_HZ_MAX, 2093.0),
        frame_length=2048, hop_length=512,
    )
    hop_s = 512 / sr
    notes = []
    current_pitch = None
    current_start = None
    current_probs = []

    for i, (freq, is_voiced, prob) in enumerate(zip(f0, voiced_flag, voiced_prob)):
        t = i * hop_s
        if is_voiced and freq is not None and not np.isnan(freq):
            midi = int(round(69 + 12 * np.log2(freq / 440.0)))
            if not (PIANO_MIDI_MIN <= midi <= PIANO_MIDI_MAX):
                is_voiced = False

        if is_voiced and freq is not None and not np.isnan(freq):
            midi = int(round(69 + 12 * np.log2(freq / 440.0)))
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

    if current_pitch is not None:
        _finish_note(notes, current_pitch, current_start, len(f0) * hop_s, current_probs)

    notes = [n for n in notes if n["duration"] >= PIANO_PYIN_MIN_NOTE_DURATION_S]
    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 2P] pyin: {len(notes)} notes")

    if save:
        _save_piano(notes)
    return notes


def _extract_crepe_piano(audio_path: str, piano_mode: str) -> list[dict]:
    import torchcrepe
    import librosa
    import torch

    conf_thresh = PIANO_CREPE_CONF_THRESHOLDS.get(piano_mode, 0.72)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    if PIANO_CREPE_REPET_ENABLED:
        S_full   = np.abs(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine")
        S_filter = np.minimum(S_full, S_filter)
        mask     = librosa.util.softmask(S_full - S_filter, 2.0 * S_filter, power=2)
        phase    = np.exp(1j * np.angle(librosa.stft(y)))
        y        = np.real(librosa.istft(mask * S_full * phase))

    hop_length_samples = 256
    hop_s = hop_length_samples / sr
    audio_tensor = torch.tensor(y[None], dtype=torch.float32)
    frequency, confidence = torchcrepe.predict(
        audio_tensor, sr,
        hop_length=hop_length_samples,
        fmin=PIANO_HZ_MIN,
        fmax=PIANO_CREPE_FMAX,
        model=PIANO_CREPE_MODEL,
        return_periodicity=True,
        decoder=torchcrepe.decode.viterbi,
        device="cpu",
    )

    freq_arr = frequency.squeeze(0).numpy()
    conf_arr = confidence.squeeze(0).numpy()
    notes = []
    current_pitch = None
    current_start = None
    current_probs = []
    last_voiced_t = None

    for i, (freq, conf) in enumerate(zip(freq_arr, conf_arr)):
        t = i * hop_s
        is_voiced = (conf >= conf_thresh and freq > 0 and not np.isnan(freq)
                     and PIANO_HZ_MIN <= freq <= PIANO_CREPE_FMAX)
        if is_voiced:
            midi = int(round(69 + 12 * np.log2(freq / 440.0)))
            if not (PIANO_MIDI_MIN <= midi <= PIANO_MIDI_MAX):
                is_voiced = False

        if is_voiced:
            midi = int(round(69 + 12 * np.log2(freq / 440.0)))
            gap  = (t - last_voiced_t) if last_voiced_t is not None else 0.0
            if midi == current_pitch and gap <= PIANO_CREPE_MAX_GAP_S:
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
                if gap > PIANO_CREPE_MAX_GAP_S:
                    _finish_note(notes, current_pitch, current_start,
                                 last_voiced_t + hop_s, current_probs)
                    current_pitch = None
                    current_start = None
                    current_probs = []
                    last_voiced_t = None

    if current_pitch is not None and current_start is not None:
        _finish_note(notes, current_pitch, current_start,
                     len(freq_arr) * hop_s, current_probs)

    notes = [n for n in notes if n["duration"] >= PIANO_CREPE_MIN_NOTE_S]
    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 2P] CREPE: {len(notes)} notes (conf>={conf_thresh})")
    return notes


def _merge_crepe_with_basic_pitch_piano(crepe_notes, bp_notes):
    import bisect

    if not crepe_notes:
        return bp_notes

    bp_sorted = sorted(bp_notes, key=lambda n: n["start"])
    bp_starts  = [n["start"] for n in bp_sorted]
    boosted = added = 0
    bp_out  = [dict(n) for n in bp_sorted]

    for cn in crepe_notes:
        c_s, c_e, c_p, c_conf = cn["start"], cn["start"] + cn["duration"], cn["pitch"], cn["confidence"]
        lo = bisect.bisect_left(bp_starts, c_s - 0.10)
        hi = bisect.bisect_right(bp_starts, c_e + 0.10)
        best_idx = best_conf = None
        for idx in range(lo, hi):
            bp = bp_out[idx]
            overlap = max(0.0, min(c_e, bp["start"] + bp["duration"]) - max(c_s, bp["start"]))
            if overlap > 0 and abs(bp["pitch"] - c_p) <= PIANO_CREPE_PITCH_TOLERANCE and bp["confidence"] >= 0.10:
                if best_conf is None or bp["confidence"] > best_conf:
                    best_conf = bp["confidence"]
                    best_idx  = idx
        if best_idx is not None:
            if c_conf > bp_out[best_idx]["confidence"]:
                bp_out[best_idx] = dict(bp_out[best_idx], confidence=round(c_conf, 4))
                boosted += 1
        else:
            bp_out.append(dict(cn))
            bp_starts.append(c_s)
            added += 1

    bp_out.sort(key=lambda n: n["start"])
    print(f"[Stage 2P] CREPE merge: boosted {boosted}, added {added} = {len(bp_out)} total")
    return bp_out


def _finish_note(notes, pitch, start, end, probs):
    notes.append({
        "pitch":      pitch,
        "start":      round(start, 4),
        "duration":   round(end - start, 4),
        "confidence": round(float(np.mean(probs)) if probs else 0.0, 4),
    })


def _recover_high_notes_piano(notes: list[dict], model_output) -> list[dict]:
    """Recovery pass for high piano notes (MIDI >= PIANO_HIGH_NOTE_RECOVERY_MIDI)."""
    from basic_pitch.inference import infer as bp_infer
    import bisect

    min_note_len = int(round(PIANO_HIGH_NOTE_RECOVERY_MIN_MS / 1000 * (22050 / 512)))

    try:
        midi_data, _ = bp_infer.model_output_to_notes(
            model_output,
            onset_thresh=PIANO_HIGH_NOTE_RECOVERY_ONSET,
            frame_thresh=PIANO_HIGH_NOTE_RECOVERY_FRAME,
            min_note_len=min_note_len,
            min_freq=PIANO_HIGH_NOTE_RECOVERY_HZ,
            max_freq=PIANO_HZ_MAX,
            melodia_trick=False,
        )
    except Exception as exc:
        print(f"[Stage 2P] High-note recovery skipped ({exc})")
        return notes

    note_arr = model_output.get("note")
    existing = sorted((n["pitch"], n["start"]) for n in notes)
    added = 0

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitch = int(note.pitch)
            if not (PIANO_HIGH_NOTE_RECOVERY_MIDI <= pitch <= PIANO_MIDI_MAX):
                continue
            start = round(float(note.start), 4)
            lo = bisect.bisect_left(existing, (pitch, start - 0.080))
            hi = bisect.bisect_right(existing, (pitch, start + 0.080))
            if any(existing[i][0] == pitch for i in range(lo, hi)):
                continue
            conf = _frame_confidence(note_arr, note.start, note.end, pitch)
            if conf < PIANO_HIGH_NOTE_RECOVERY_MIN_CONF:
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
        print(f"[Stage 2P] High-note recovery: added {added} notes (MIDI>={PIANO_HIGH_NOTE_RECOVERY_MIDI})")
    return notes


def _apply_sustained_boost_piano(
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
        print(f"[Stage 2P] Sustained boost applied to {boosted} notes")


def _save_piano(notes: list[dict]) -> None:
    out_path = os.path.join(get_instrument_dir("piano"), "02_raw_notes.json")
    with open(out_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"[Stage 2P] Saved -> {out_path}")


def load_raw_notes_piano() -> list[dict]:
    path = os.path.join(get_instrument_dir("piano"), "02_raw_notes.json")
    with open(path) as f:
        return json.load(f)
