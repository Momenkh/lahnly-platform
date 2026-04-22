"""
Stage 4: Tempo Detection & Note Quantization
Input:  cleaned notes + audio path
Output: quantized notes -> outputs/04_quantized_notes.json
        tempo info      -> outputs/04_tempo.json

Quality improvements:
  - BPM-relative snap tolerance (35% of a subdivision) instead of fixed 50ms
    (fixed 50ms is catastrophic at fast tempos: 50ms / 83ms = 60% of a 16th note)
  - Half/double-time BPM correction: tests bpm, bpm*2, bpm/2 and picks whichever
    causes the most notes to snap — catches the most common librosa error
  - Onset-dense window: finds the 60s window with the most note onsets and uses
    that for beat tracking (avoids silent intros/outros corrupting BPM)
  - Tempo instability warning: flags songs where quantization may be unreliable
"""

import json
import os
import bisect
import numpy as np

from pipeline.config import get_outputs_dir
from pipeline.settings import (
    QUANTIZATION_SUBDIVISION,
    QUANTIZATION_SNAP_TOLERANCE_FRAC,
    QUANTIZATION_DENSE_WINDOW_S,
    QUANTIZATION_BPM_MIN,
    QUANTIZATION_BPM_MAX,
    QUANTIZATION_INSTABILITY_WARN_MS,
)


def quantize_notes(
    notes: list[dict],
    audio_path: str,
    bpm_override: float | None = None,
    reuse_tempo: dict | None = None,
    save: bool = True,
) -> tuple[list[dict], dict]:
    """
    Detect BPM and snap note start times to the nearest 16th-note grid point.
    Returns (quantized_notes, tempo_info).

    bpm_override: force a specific BPM instead of detecting from audio
    reuse_tempo:  skip detection entirely and use this saved tempo_info dict
                  (prevents BPM flip-flopping when re-cleaning with a different
                  guitar type but the same audio)
    """
    if bpm_override is not None:
        bpm      = max(QUANTIZATION_BPM_MIN, min(QUANTIZATION_BPM_MAX, bpm_override))
        beat_s   = 60.0 / bpm
        subdiv_s = beat_s / QUANTIZATION_SUBDIVISION
        tempo_info = {
            "bpm":            round(bpm, 2),
            "beat_s":         round(beat_s, 6),
            "subdivision_s":  round(subdiv_s, 6),
            "instability_ms": 0.0,
            "window_start_s": 0.0,
            "source":         "override",
        }
        print(f"[Stage 4] BPM override: {bpm:.1f}")

    elif reuse_tempo is not None:
        tempo_info = reuse_tempo
        print(f"[Stage 4] Reusing saved BPM: {tempo_info['bpm']:.1f}  "
              f"(pass --force-tempo to re-detect)")

    else:
        print(f"[Stage 4] Detecting tempo from audio...")
        note_starts = sorted(n["start"] for n in notes)
        tempo_info  = _detect_tempo(audio_path, note_starts)

    bpm       = tempo_info["bpm"]
    subdiv_s  = tempo_info["subdivision_s"]
    tolerance = subdiv_s * QUANTIZATION_SNAP_TOLERANCE_FRAC

    print(f"[Stage 4] BPM: {bpm:.1f}  |  "
          f"16th note = {subdiv_s*1000:.1f}ms  |  "
          f"snap tolerance = {tolerance*1000:.1f}ms")

    if tempo_info.get("instability_ms", 0) > QUANTIZATION_INSTABILITY_WARN_MS:
        print(f"[Stage 4] WARNING: tempo unstable "
              f"(std={tempo_info['instability_ms']:.1f}ms) — quantization may drift")

    snapped = 0
    quantized = []
    for note in notes:
        n = dict(note)
        nearest = round(n["start"] / subdiv_s) * subdiv_s
        if abs(nearest - n["start"]) <= tolerance:
            n["start"] = round(nearest, 4)
            snapped += 1
        quantized.append(n)

    print(f"[Stage 4] Snapped {snapped}/{len(notes)} note starts to grid")

    # Snap durations to the nearest subdivision using the same tolerance.
    # Very short articulations (< 0.5 × subdivision) are left untouched —
    # hammer-ons, pull-offs, and ghost notes below this threshold should not
    # be elongated to the nearest grid point.
    dur_snapped = 0
    for note in quantized:
        raw_dur = note["duration"]
        if raw_dur < 0.5 * subdiv_s:
            continue   # leave very short articulations at their original duration
        nearest_dur = max(subdiv_s, round(raw_dur / subdiv_s) * subdiv_s)
        if abs(nearest_dur - raw_dur) <= tolerance:
            note["duration"] = round(nearest_dur, 4)
            dur_snapped += 1

    print(f"[Stage 4] Snapped {dur_snapped}/{len(quantized)} note durations to grid")

    if save:
        _save(quantized, tempo_info)

    return quantized, tempo_info


def load_quantization() -> tuple[list[dict], dict]:
    base = get_outputs_dir()
    with open(os.path.join(base, "04_quantized_notes.json")) as f:
        notes = json.load(f)
    with open(os.path.join(base, "04_tempo.json")) as f:
        tempo_info = json.load(f)
    return notes, tempo_info


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_tempo(audio_path: str, note_starts: list[float]) -> dict:
    import librosa
    from pipeline.pitch_extraction import load_audio

    # Find the densest window of note activity
    window_start = _dense_window(note_starts, QUANTIZATION_DENSE_WINDOW_S)
    print(f"[Stage 4] Using audio window {window_start:.1f}s – "
          f"{window_start + QUANTIZATION_DENSE_WINDOW_S:.1f}s for BPM detection")

    y, sr = load_audio(audio_path, target_sr=22050)

    start_sample = int(window_start * sr)
    end_sample   = min(len(y), int((window_start + QUANTIZATION_DENSE_WINDOW_S) * sr))
    y_window     = y[start_sample:end_sample]

    bpm_raw, beats = librosa.beat.beat_track(y=y_window, sr=sr, units="time")
    if hasattr(bpm_raw, "__len__"):
        bpm_raw = float(bpm_raw[0])
    else:
        bpm_raw = float(bpm_raw)

    # Correct beats back to global time
    beats_global = beats + window_start

    # Compute inter-beat interval instability
    if len(beats_global) > 2:
        ibi            = np.diff(beats_global) * 1000   # ms
        instability_ms = float(np.std(ibi))
    else:
        instability_ms = 0.0

    # Half / double-time correction
    bpm = _best_bpm_candidate(bpm_raw, note_starts)
    bpm = max(QUANTIZATION_BPM_MIN, min(QUANTIZATION_BPM_MAX, bpm))

    beat_s   = 60.0 / bpm
    subdiv_s = beat_s / QUANTIZATION_SUBDIVISION

    return {
        "bpm":            round(bpm, 2),
        "beat_s":         round(beat_s, 6),
        "subdivision_s":  round(subdiv_s, 6),
        "instability_ms": round(instability_ms, 2),
        "window_start_s": round(window_start, 2),
    }


def _dense_window(starts: list[float], window: float) -> float:
    """Return the start time of the window with the most note onsets."""
    if not starts:
        return 0.0
    best_t, best_count = 0.0, 0
    for t in starts:
        lo = bisect.bisect_left(starts, t)
        hi = bisect.bisect_left(starts, t + window)
        count = hi - lo
        if count > best_count:
            best_count, best_t = count, t
    return best_t


def _best_bpm_candidate(bpm_raw: float, note_starts: list[float]) -> float:
    """
    Test bpm, bpm/2, bpm*2. Pick the candidate that causes the most note onsets
    to fall within the snap tolerance of a grid point.
    """
    candidates = [bpm_raw / 2, bpm_raw, bpm_raw * 2]
    candidates = [b for b in candidates if QUANTIZATION_BPM_MIN <= b <= QUANTIZATION_BPM_MAX]

    best_bpm, best_count = bpm_raw, -1

    for bpm in candidates:
        subdiv  = 60.0 / bpm / QUANTIZATION_SUBDIVISION
        tol     = subdiv * QUANTIZATION_SNAP_TOLERANCE_FRAC
        count   = sum(
            1 for t in note_starts
            if abs(t - round(t / subdiv) * subdiv) <= tol
        )
        if count > best_count:
            best_count, best_bpm = count, bpm

    if best_bpm != bpm_raw:
        print(f"[Stage 4] BPM corrected: {bpm_raw:.1f} -> {best_bpm:.1f} "
              f"(half/double-time — {best_count} notes snap)")

    return best_bpm


def _save(notes: list[dict], tempo_info: dict) -> None:
    os.makedirs(get_outputs_dir(), exist_ok=True)
    base = get_outputs_dir()
    with open(os.path.join(base, "04_quantized_notes.json"), "w") as f:
        json.dump(notes, f, indent=2)
    with open(os.path.join(base, "04_tempo.json"), "w") as f:
        json.dump(tempo_info, f, indent=2)
    print(f"[Stage 4] Saved -> 04_quantized_notes.json + 04_tempo.json")
