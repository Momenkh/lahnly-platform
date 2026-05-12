"""
Stage 2 (Vocals): Pitch Extraction
CREPE is the primary extractor for vocals (monophonic, harmonic, F0 very clear).
basic-pitch runs as a secondary verification pass.
Output: raw notes [{pitch, start, duration, confidence}]
        saved to outputs/02_vocals_raw_notes.json
"""

import json
import os
import numpy as np

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    VOCALS_MIDI_MIN, VOCALS_MIDI_MAX, VOCALS_HZ_MIN, VOCALS_HZ_MAX,
    BP_SAMPLE_RATE, BP_HOP_LENGTH, BP_MIDI_OFFSET,
    VOCALS_MULTI_PASS_CONFIGS, VOCALS_PITCH_THRESHOLDS,
    VOCALS_PITCH_ONSET_BASE, VOCALS_PITCH_ONSET_SCALE,
    VOCALS_PITCH_FRAME_BASE, VOCALS_PITCH_FRAME_SCALE,
    VOCALS_PITCH_MERGE_PROXIMITY_S,
    VOCALS_CONF_WEIGHT_BASE, VOCALS_CONF_WEIGHT_CONFIRMED, VOCALS_CONF_WEIGHT_STRONG,
    VOCALS_PYIN_MIN_NOTE_DURATION_S,
    VOCALS_CREPE_ENABLED, VOCALS_CREPE_MODEL, VOCALS_CREPE_FMAX,
    VOCALS_CREPE_FMAX_PER_MODE, VOCALS_HARMONY_BP_PRIMARY,
    VOCALS_CREPE_CONF_THRESHOLDS, VOCALS_CREPE_MIN_NOTE_S, VOCALS_CREPE_MAX_GAP_S,
    VOCALS_CREPE_PITCH_TOLERANCE, VOCALS_CREPE_REPET_ENABLED, VOCALS_CREPE_REPET_AUTO_THRESHOLD,
    VOCALS_SUSTAINED_BOOST_MIN_S,
    VOCALS_SUSTAINED_BOOST_AMOUNT,
    VOCALS_SUSTAINED_BOOST_CAP,
)

BP_FRAMES_PER_SEC = BP_SAMPLE_RATE / BP_HOP_LENGTH


def extract_pitches_vocals(
    audio_path: str,
    vocals_mode: str = "vocals_lead",
    save: bool = True,
    use_repet: bool = False,
) -> list[dict]:
    """
    Extract vocal pitch track. CREPE first, basic-pitch as fallback/secondary.
    vocals_mode: "vocals_lead" | "vocals_harmony"
    use_repet: override for VOCALS_CREPE_REPET_ENABLED (also enabled by --vocals-repet CLI flag)
    """
    from pipeline.shared.presence import check_and_update_stem_presence
    if not check_and_update_stem_presence("vocals"):
        if save:
            _save_vocals([])
        return []

    # For vocals, CREPE is the primary — run it first and always
    notes = []

    # Auto-enable REPET-SIM for weak stems
    from pipeline.shared.separation import load_stem_meta_for
    stem_conf_v  = load_stem_meta_for("vocals").get("stem_confidence", 0.5)
    auto_repet_v = stem_conf_v < VOCALS_CREPE_REPET_AUTO_THRESHOLD
    if auto_repet_v and not (VOCALS_CREPE_REPET_ENABLED or use_repet):
        print(f"[Stage 2V] Auto-enabling REPET-SIM (stem_confidence={stem_conf_v:.2f} < {VOCALS_CREPE_REPET_AUTO_THRESHOLD})")
    use_repet_final = VOCALS_CREPE_REPET_ENABLED or use_repet or auto_repet_v

    harmony_bp_primary = (vocals_mode == "vocals_harmony" and VOCALS_HARMONY_BP_PRIMARY)

    if harmony_bp_primary:
        # Harmony mode: basic-pitch is primary (polyphonic → captures harmony lines)
        print(f"[Stage 2V] Harmony mode: basic-pitch primary, CREPE booster")
        try:
            notes = _extract_basic_pitch_vocals(audio_path, vocals_mode)
        except ImportError:
            notes = _extract_pyin_vocals(audio_path)
        if notes and VOCALS_CREPE_ENABLED:
            try:
                crepe_notes = _extract_crepe_vocals(audio_path, vocals_mode, repet=use_repet_final)
                notes = _merge_bp_primary_crepe_booster(notes, crepe_notes)
            except Exception:
                pass   # CREPE boost is best-effort; bp result stands
    else:
        # Lead mode (and harmony fallback): CREPE is primary
        if VOCALS_CREPE_ENABLED:
            try:
                notes = _extract_crepe_vocals(audio_path, vocals_mode,
                                              repet=use_repet_final)
            except ImportError:
                print("[Stage 2V] torchcrepe not available — falling back to basic-pitch only")
            except Exception as exc:
                print(f"[Stage 2V] CREPE failed ({exc}) — falling back to basic-pitch only")

        if not notes:
            # CREPE unavailable or produced nothing — use basic-pitch as primary
            try:
                notes = _extract_basic_pitch_vocals(audio_path, vocals_mode)
            except ImportError:
                notes = _extract_pyin_vocals(audio_path)
        elif VOCALS_CREPE_ENABLED:
            # CREPE succeeded — use basic-pitch as secondary verification
            try:
                bp_notes = _extract_basic_pitch_vocals(audio_path, vocals_mode, save=False)
                notes    = _merge_crepe_primary_bp_secondary(notes, bp_notes)
            except Exception:
                pass   # basic-pitch verification is best-effort; CREPE result stands

    _apply_sustained_boost_vocals(notes, VOCALS_SUSTAINED_BOOST_MIN_S,
                                   VOCALS_SUSTAINED_BOOST_AMOUNT, VOCALS_SUSTAINED_BOOST_CAP)

    if save:
        _save_vocals(notes)
    return notes


def _extract_crepe_vocals(audio_path: str, vocals_mode: str, repet: bool = False) -> list[dict]:
    import torchcrepe
    import librosa
    import torch

    conf_thresh = VOCALS_CREPE_CONF_THRESHOLDS.get(vocals_mode, 0.65)
    fmax        = VOCALS_CREPE_FMAX_PER_MODE.get(vocals_mode, VOCALS_CREPE_FMAX)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    if repet:
        S_full   = np.abs(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine")
        S_filter = np.minimum(S_full, S_filter)
        mask     = librosa.util.softmask(S_full - S_filter, 2.0 * S_filter, power=2)
        phase    = np.exp(1j * np.angle(librosa.stft(y)))
        y        = np.real(librosa.istft(mask * S_full * phase))
        print("[Stage 2V] CREPE: REPET-SIM foreground separation applied")

    hop_length_samples = 256
    hop_s = hop_length_samples / sr
    _crepe_device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tensor = torch.tensor(y[None], dtype=torch.float32)

    frequency, confidence = torchcrepe.predict(
        audio_tensor, sr,
        hop_length=hop_length_samples,
        fmin=VOCALS_HZ_MIN,
        fmax=fmax,
        model=VOCALS_CREPE_MODEL,
        return_periodicity=True,
        decoder=torchcrepe.decode.viterbi,
        device=_crepe_device,
    )

    freq_arr = frequency.squeeze(0).numpy()
    conf_arr = confidence.squeeze(0).numpy()

    notes = []
    current_pitch = current_start = None
    current_probs = []
    last_voiced_t = None

    for i, (freq, conf) in enumerate(zip(freq_arr, conf_arr)):
        t = i * hop_s
        is_voiced = (conf >= conf_thresh and freq > 0 and not np.isnan(freq)
                     and VOCALS_HZ_MIN <= freq <= VOCALS_CREPE_FMAX)
        if is_voiced:
            midi = _hz_to_midi(freq)
            if not (VOCALS_MIDI_MIN <= midi <= VOCALS_MIDI_MAX):
                is_voiced = False

        if is_voiced:
            midi = _hz_to_midi(freq)
            gap  = (t - last_voiced_t) if last_voiced_t is not None else 0.0
            if midi == current_pitch and gap <= VOCALS_CREPE_MAX_GAP_S:
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
                if gap > VOCALS_CREPE_MAX_GAP_S:
                    _finish_note(notes, current_pitch, current_start,
                                 last_voiced_t + hop_s, current_probs)
                    current_pitch = current_start = last_voiced_t = None
                    current_probs = []

    if current_pitch is not None and current_start is not None:
        _finish_note(notes, current_pitch, current_start, len(freq_arr) * hop_s, current_probs)

    notes = [n for n in notes if n["duration"] >= VOCALS_CREPE_MIN_NOTE_S]
    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 2V] CREPE: {len(notes)} notes (conf>={conf_thresh}, mode={vocals_mode})")
    return notes


def _extract_basic_pitch_vocals(
    audio_path: str, vocals_mode: str, save: bool = False
) -> list[dict]:
    from basic_pitch.inference import run_inference, AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.inference import infer as bp_infer
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from pipeline.shared.separation import load_stem_meta_for

    stem_conf    = load_stem_meta_for("vocals").get("stem_confidence", 0.5)
    pass_configs = VOCALS_MULTI_PASS_CONFIGS.get(vocals_mode)
    if pass_configs is None:
        onset, frame, min_ms = _resolve_thresholds(vocals_mode, stem_conf)
        pass_configs = [(onset, frame, min_ms)]

    print(f"[Stage 2V] Running basic-pitch inference (secondary)...")
    model_output = run_inference(audio_path, ICASSP_2022_MODEL_PATH)
    note_arr     = model_output.get("note")

    all_pass_notes = []
    for onset_t, frame_t, min_ms in pass_configs:
        min_note_len = int(np.round(min_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
        midi_data, _ = bp_infer.model_output_to_notes(
            model_output,
            onset_thresh=onset_t, frame_thresh=frame_t,
            min_note_len=min_note_len,
            min_freq=VOCALS_HZ_MIN, max_freq=VOCALS_HZ_MAX,
            melodia_trick=True,   # vocal melody: use melodia (monophonic)
        )
        pass_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                pitch = int(note.pitch)
                if not (VOCALS_MIDI_MIN <= pitch <= VOCALS_MIDI_MAX):
                    continue
                pass_notes.append({
                    "pitch":      pitch,
                    "start":      round(float(note.start), 4),
                    "duration":   round(float(note.end - note.start), 4),
                    "confidence": round(_frame_conf(note_arr, note.start, note.end, pitch), 4),
                })
        pass_notes.sort(key=lambda n: n["start"])
        all_pass_notes.append(pass_notes)

    notes = _merge_passes(all_pass_notes, note_arr)
    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 2V] basic-pitch: {len(notes)} notes")
    return notes


def _extract_pyin_vocals(audio_path: str) -> list[dict]:
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, sr=sr, fmin=VOCALS_HZ_MIN, fmax=VOCALS_HZ_MAX,
        frame_length=2048, hop_length=512,
    )
    hop_s = 512 / sr
    notes = []
    cp = cs = None
    cprobs = []
    for i, (freq, is_v, prob) in enumerate(zip(f0, voiced_flag, voiced_prob)):
        t = i * hop_s
        if is_v and freq and not np.isnan(freq):
            midi = _hz_to_midi(freq)
            if VOCALS_MIDI_MIN <= midi <= VOCALS_MIDI_MAX:
                if midi == cp:
                    cprobs.append(float(prob))
                else:
                    if cp is not None:
                        _finish_note(notes, cp, cs, t, cprobs)
                    cp, cs, cprobs = midi, t, [float(prob)]
            else:
                if cp is not None:
                    _finish_note(notes, cp, cs, t, cprobs)
                cp = cs = None; cprobs = []
        else:
            if cp is not None:
                _finish_note(notes, cp, cs, t, cprobs)
            cp = cs = None; cprobs = []
    if cp is not None:
        _finish_note(notes, cp, cs, len(f0) * hop_s, cprobs)
    notes = [n for n in notes if n["duration"] >= VOCALS_PYIN_MIN_NOTE_DURATION_S]
    notes.sort(key=lambda n: n["start"])
    print(f"[Stage 2V] pyin: {len(notes)} notes")
    return notes


def _merge_bp_primary_crepe_booster(
    bp_notes: list[dict], crepe_notes: list[dict]
) -> list[dict]:
    """
    Harmony mode merge: basic-pitch is the source of truth (polyphonic).
    CREPE is monophonic — it only boosts confidence of bp notes it agrees with.
    CREPE-only notes are NOT added (they would be the dominant pitch, already in bp).
    """
    if not crepe_notes:
        return bp_notes

    out = [dict(n) for n in bp_notes]
    boosted = 0

    for cn in crepe_notes:
        c_s, c_e, c_p = cn["start"], cn["start"] + cn["duration"], cn["pitch"]
        for note in out:
            n_e = note["start"] + note["duration"]
            overlap = max(0.0, min(c_e, n_e) - max(c_s, note["start"]))
            if overlap > 0 and abs(note["pitch"] - c_p) <= VOCALS_CREPE_PITCH_TOLERANCE:
                if cn["confidence"] > note["confidence"]:
                    note["confidence"] = round(cn["confidence"], 4)
                    boosted += 1
                break

    out.sort(key=lambda n: n["start"])
    print(f"[Stage 2V] Merge: bp primary + CREPE booster: {boosted} notes boosted = {len(out)} total")
    return out


def _merge_crepe_primary_bp_secondary(
    crepe_notes: list[dict], bp_notes: list[dict]
) -> list[dict]:
    """
    CREPE is trusted as ground truth for timing/pitch.
    basic-pitch boosts confidence where it agrees, and adds notes CREPE missed.
    """
    import bisect
    if not bp_notes:
        return crepe_notes

    bp_sorted = sorted(bp_notes, key=lambda n: n["start"])
    bp_starts  = [n["start"] for n in bp_sorted]
    out = [dict(n) for n in crepe_notes]
    added = boosted = 0

    for bp in bp_sorted:
        b_s, b_e, b_p = bp["start"], bp["start"] + bp["duration"], bp["pitch"]
        # Find CREPE notes that overlap this bp note
        lo = bisect.bisect_left(bp_starts, b_s - 0.15)
        matched = False
        for cn in out:
            c_e = cn["start"] + cn["duration"]
            if cn["start"] > b_e + 0.10:
                break
            overlap = max(0.0, min(b_e, c_e) - max(b_s, cn["start"]))
            if overlap > 0 and abs(cn["pitch"] - b_p) <= VOCALS_CREPE_PITCH_TOLERANCE:
                if bp["confidence"] > cn["confidence"]:
                    cn["confidence"] = round(bp["confidence"], 4)
                    boosted += 1
                matched = True
                break
        if not matched:
            out.append(dict(bp))
            added += 1

    out.sort(key=lambda n: n["start"])
    print(f"[Stage 2V] Merge: CREPE primary + bp secondary: "
          f"boosted {boosted}, added {added} = {len(out)} total")
    return out


# ── helpers ───────────────────────────────────────────────────────────────────

def _hz_to_midi(freq: float) -> int:
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def _frame_conf(note_arr, start_s, end_s, midi_pitch):
    if note_arr is None:
        return 0.5
    start_f   = int(start_s * BP_FRAMES_PER_SEC)
    end_f     = max(start_f + 1, int(end_s * BP_FRAMES_PER_SEC))
    pitch_idx = midi_pitch - BP_MIDI_OFFSET
    if not (0 <= pitch_idx < note_arr.shape[1]):
        return 0.5
    frames = note_arr[start_f:end_f, pitch_idx]
    return float(frames.mean()) if len(frames) > 0 else 0.5


def _resolve_thresholds(vocals_mode, stem_conf):
    row = VOCALS_PITCH_THRESHOLDS.get(vocals_mode, (None, None, 60))
    onset, frame, min_ms = row
    if onset is None:
        onset = round(VOCALS_PITCH_ONSET_BASE - stem_conf * VOCALS_PITCH_ONSET_SCALE, 2)
        frame = round(VOCALS_PITCH_FRAME_BASE - stem_conf * VOCALS_PITCH_FRAME_SCALE, 2)
    return onset, frame, min_ms


def _merge_passes(all_pass_notes, note_arr):
    import bisect
    if len(all_pass_notes) == 1:
        result = []
        for note in all_pass_notes[0]:
            ns, ne = note["start"], note["start"] + note["duration"]
            note["confidence"] = round(_frame_conf(note_arr, ns, ne, note["pitch"]), 4)
            result.append(note)
        return result

    base = all_pass_notes[0]
    stricter = all_pass_notes[1:]
    lookups  = [sorted((n["pitch"], n["start"]) for n in p) for p in stricter]

    def confirmed(note, lk):
        p, t = note["pitch"], note["start"]
        lo = bisect.bisect_left(lk,  (p, t - VOCALS_PITCH_MERGE_PROXIMITY_S))
        hi = bisect.bisect_right(lk, (p, t + VOCALS_PITCH_MERGE_PROXIMITY_S))
        return any(lk[i][0] == p for i in range(lo, hi))

    result = []
    for note in sorted(base, key=lambda n: n["start"]):
        confs = sum(1 for lk in lookups if confirmed(note, lk))
        weight = (VOCALS_CONF_WEIGHT_STRONG if confs >= len(stricter)
                  else VOCALS_CONF_WEIGHT_CONFIRMED if confs >= 1
                  else VOCALS_CONF_WEIGHT_BASE)
        ns, ne = note["start"], note["start"] + note["duration"]
        result.append(dict(note, confidence=round(_frame_conf(note_arr, ns, ne, note["pitch"]) * weight, 4)))
    return result


def _finish_note(notes, pitch, start, end, probs):
    notes.append({
        "pitch":      pitch,
        "start":      round(start, 4),
        "duration":   round(end - start, 4),
        "confidence": round(float(np.mean(probs)) if probs else 0.0, 4),
    })


def _apply_sustained_boost_vocals(
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
        print(f"[Stage 2V] Sustained boost applied to {boosted} notes")


def _save_vocals(notes):
    out_path = os.path.join(get_instrument_dir("vocals"), "02_raw_notes.json")
    with open(out_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"[Stage 2V] Saved -> {out_path}")


def load_raw_notes_vocals() -> list[dict]:
    path = os.path.join(get_instrument_dir("vocals"), "02_raw_notes.json")
    with open(path) as f:
        return json.load(f)
