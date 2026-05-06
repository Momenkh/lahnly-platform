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

from pipeline.config import get_shared_dir
from pipeline.settings import (
    QUANTIZATION_SUBDIVISION,
    QUANTIZATION_SNAP_TOLERANCE_FRAC,
    QUANTIZATION_DENSE_WINDOW_S,
    QUANTIZATION_BPM_MIN,
    QUANTIZATION_BPM_MAX,
    QUANTIZATION_INSTABILITY_WARN_MS,
    TIME_SIG_CANDIDATES,
    TIME_SIG_4_4_BIAS,
    TIME_SIG_6_8_RATIO,
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
        note_starts = sorted(n["start"] for n in notes)
        ts_num, ts_den = infer_time_signature(note_starts, beat_s)
        tempo_info = {
            "bpm":            round(bpm, 2),
            "beat_s":         round(beat_s, 6),
            "subdivision_s":  round(subdiv_s, 6),
            "instability_ms": 0.0,
            "window_start_s": 0.0,
            "source":         "override",
            "time_sig_num":   ts_num,
            "time_sig_den":   ts_den,
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

    ts_num = tempo_info.get("time_sig_num", 4)
    ts_den = tempo_info.get("time_sig_den", 4)
    print(f"[Stage 4] BPM: {bpm:.1f}  |  Time: {ts_num}/{ts_den}  |  "
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
    base = get_shared_dir()
    with open(os.path.join(base, "04_quantized_notes.json")) as f:
        notes = json.load(f)
    with open(os.path.join(base, "04_tempo.json")) as f:
        tempo_info = json.load(f)
    return notes, tempo_info


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_tempo(audio_path: str, note_starts: list[float]) -> dict:
    import librosa
    from pipeline.instruments.guitar.pitch import load_audio

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

    time_sig_num, time_sig_den = infer_time_signature(note_starts, beat_s)

    return {
        "bpm":            round(bpm, 2),
        "beat_s":         round(beat_s, 6),
        "subdivision_s":  round(subdiv_s, 6),
        "instability_ms": round(instability_ms, 2),
        "window_start_s": round(window_start, 2),
        "time_sig_num":   time_sig_num,
        "time_sig_den":   time_sig_den,
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


def infer_time_signature(
    note_starts: list[float],
    beat_s: float,
) -> tuple[int, int]:
    """
    Infer time signature from note onset autocorrelation.

    Bins note onsets onto a fine grid (beat_s/16 resolution), computes
    autocorrelation via FFT, then samples it at lags corresponding to
    2, 3, and 4 beats per bar.  The candidate whose lag has the highest
    autocorrelation wins, with a small bias toward 4/4 to avoid flipping
    common-time songs on sparse onset patterns.

    Compound duple (6/8) is distinguished from simple triple (3/4) by
    checking whether the dotted-quarter lag (1.5×beat) is also strong:
    in 6/8 the subdivisions naturally cluster in threes within two main
    pulses, raising the 1.5-beat autocorrelation relative to 3/4.

    Returns (numerator, denominator), e.g. (4, 4), (3, 4), (6, 8), (2, 4).
    Falls back to (4, 4) when there are too few onsets to be reliable.
    """
    MIN_ONSETS = 8
    if len(note_starts) < MIN_ONSETS or beat_s <= 0:
        return (4, 4)

    # ── Build onset vector ────────────────────────────────────────────────────
    # Resolution: 1/16th of a beat (= 64th note at quarter-note beat).
    # Fine enough to resolve 1/16th-note timing differences; coarse enough to
    # pool nearby onsets (rubato, expressive timing) into the same bin.
    bin_s   = beat_s / 16.0
    max_t   = max(note_starts)
    n_bins  = int(max_t / bin_s) + 2
    onset_v = np.zeros(n_bins)
    for t in note_starts:
        idx = int(round(t / bin_s))
        if 0 <= idx < n_bins:
            onset_v[idx] += 1.0

    # ── Autocorrelation via FFT (O(n log n) vs O(n²) for direct) ─────────────
    fft_size = 1
    while fft_size < 2 * n_bins:
        fft_size <<= 1
    F  = np.fft.rfft(onset_v, n=fft_size)
    ac = np.fft.irfft(F * np.conj(F))[:n_bins]   # keep non-negative lags only
    ac_norm = ac[0] if ac[0] > 0 else 1.0         # normalise so peak = 1.0
    ac = ac / ac_norm

    def _ac_at(n_beats: float) -> float:
        """Autocorrelation value at a lag of n_beats × beat_s, averaged ±1 bin."""
        lag_bin = int(round(n_beats * beat_s / bin_s))
        lo = max(0, lag_bin - 1)
        hi = min(len(ac), lag_bin + 2)
        return float(np.mean(ac[lo:hi]))

    # ── Score each candidate time signature ───────────────────────────────────
    scores: dict[int, float] = {}
    for num in TIME_SIG_CANDIDATES:
        # Primary lag: one full bar = num beats
        s = _ac_at(num)
        # Secondary confirmation: half a bar (exists for 4/4 and 2/4, not 3/4)
        if num % 2 == 0:
            s += 0.5 * _ac_at(num / 2)
        scores[num] = s

    # Bias toward 4/4 for ambiguous patterns
    if 4 in scores:
        scores[4] *= (1.0 + TIME_SIG_4_4_BIAS)

    best_num = max(scores, key=lambda n: scores[n])

    # ── 6/8 vs 3/4 discrimination ─────────────────────────────────────────────
    # In 6/8 the dotted-quarter (1.5 beats on a quarter-note grid) is the
    # primary pulse.  If the 3-beat winner also shows strong periodicity at
    # 1.5 beats, report 6/8 rather than 3/4.
    if best_num == 3:
        dotted_q_score = _ac_at(1.5)
        bar_3_score    = _ac_at(3.0)
        if bar_3_score > 0 and dotted_q_score / bar_3_score >= TIME_SIG_6_8_RATIO:
            return (6, 8)
        return (3, 4)

    return (best_num, 4)


def _save(notes: list[dict], tempo_info: dict) -> None:
    base = get_shared_dir()
    with open(os.path.join(base, "04_quantized_notes.json"), "w") as f:
        json.dump(notes, f, indent=2)
    with open(os.path.join(base, "04_tempo.json"), "w") as f:
        json.dump(tempo_info, f, indent=2)
    print(f"[Stage 4] Saved -> shared/04_quantized_notes.json + shared/04_tempo.json")
