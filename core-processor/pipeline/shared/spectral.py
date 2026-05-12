"""
Shared spectral analysis utilities.

Functions
---------
compute_velocity(notes, stem_path)
    Estimate MIDI velocity (1–127) for each note from stem RMS at the attack point.
    Adds a "velocity" field to each note dict (in-place) and returns the list.

compute_stem_cqt(stem_path)
    Load a stem WAV and compute its CQT magnitude array.
    Returns (cqt_array, sample_rate, hop_length).
    Cache-friendly — callers should call once and pass cqt downstream.

check_harmonic_coherence(notes, stem_cqt, sr, hop, dominance_ratio, conf_override)
    Mark notes that appear to be overtones of a lower note.
    Mutates each note dict with "likely_overtone": bool.
    Caller decides whether to remove them.

spectral_presence_gate(notes, stem_cqt, sr, hop, min_energy, conf_override)
    Remove notes whose specific pitch is absent from the stem CQT at note time.
    Returns filtered list.

attack_envelope_gate(notes, stem_audio, sr, pre_s, post_s, min_ratio, conf_override)
    Remove notes with no amplitude ramp at their onset (noise/bleed).
    Returns filtered list.

filter_isolated_notes_pitch_aware(notes, window_s, min_neighbors, max_interval_st)
    Isolation filter that only counts pitch-close neighbors.
    Returns filtered list.
"""

import numpy as np
import librosa

from pipeline.settings import (
    VELOCITY_ATTACK_WEIGHT,
    VELOCITY_CONF_WEIGHT,
    VELOCITY_NORM_PERCENTILE,
    VELOCITY_ATTACK_WINDOW_S,
    VELOCITY_MIN,
    VELOCITY_MAX,
    HARMONIC_CQT_HOP,
    HARMONIC_CQT_N_BINS,
    HARMONIC_CQT_BINS_PER_OCT,
    SPECTRAL_PRESENCE_PITCH_TOL,
)

# MIDI pitch of the lowest CQT bin (C1).
_CQT_MIDI_MIN = 24


# ── CQT helper ────────────────────────────────────────────────────────────────

def compute_stem_cqt(stem_path: str) -> tuple[np.ndarray, int, int]:
    """
    Load stem and compute CQT magnitude array.
    Returns (cqt_mag, sample_rate, hop_length).
    """
    y, sr = librosa.load(stem_path, sr=22050, mono=True)
    cqt   = np.abs(librosa.cqt(
        y, sr=sr,
        hop_length=HARMONIC_CQT_HOP,
        n_bins=HARMONIC_CQT_N_BINS,
        bins_per_octave=HARMONIC_CQT_BINS_PER_OCT,
    ))
    return cqt, sr, HARMONIC_CQT_HOP


def _midi_to_bin(midi: int) -> int:
    return midi - _CQT_MIDI_MIN


def _note_cqt_energy(cqt: np.ndarray, midi: int, t_start_f: int, t_end_f: int,
                     pitch_tol: int = 0) -> float:
    """Mean CQT magnitude for a given MIDI pitch over a time range."""
    n_bins = cqt.shape[0]
    p_bin  = _midi_to_bin(midi)
    if p_bin < 0 or p_bin >= n_bins:
        return 0.0
    b_lo   = max(0, p_bin - pitch_tol)
    b_hi   = min(n_bins, p_bin + pitch_tol + 1)
    T      = cqt.shape[1]
    ts     = min(max(t_start_f, 0), T - 1)
    te     = min(max(t_end_f, ts + 1), T)
    return float(np.mean(cqt[b_lo:b_hi, ts:te]))


# ── Velocity estimation ───────────────────────────────────────────────────────

def compute_velocity(notes: list[dict], stem_path: str) -> list[dict]:
    """
    Estimate MIDI velocity (1–127) for each note from stem RMS at the attack point.

    Blends:
      - RMS in the first VELOCITY_ATTACK_WINDOW_S of the note (primary)
      - Existing confidence score (secondary)

    RMS values are normalized to the VELOCITY_NORM_PERCENTILE-th percentile
    across all notes so soft songs don't produce universally low velocities.

    Adds a "velocity" int field to each note dict in-place.
    Returns the (same) list of notes.
    """
    if not notes:
        return notes

    y, sr = librosa.load(stem_path, sr=22050, mono=True)
    win_samples = max(1, int(VELOCITY_ATTACK_WINDOW_S * sr))

    rms_raw = np.zeros(len(notes), dtype=np.float32)
    for i, note in enumerate(notes):
        onset_i  = int(note["start"] * sr)
        end_i    = min(len(y), onset_i + win_samples)
        if end_i > onset_i:
            rms_raw[i] = float(np.sqrt(np.mean(y[onset_i:end_i] ** 2)))

    # Normalize against the song's loudest notes
    p_val  = float(np.percentile(rms_raw, VELOCITY_NORM_PERCENTILE)) if len(rms_raw) > 0 else 1.0
    norm   = rms_raw / max(p_val, 1e-8)
    norm   = np.clip(norm, 0.0, 1.0)

    for note, rms_n in zip(notes, norm):
        conf    = float(note.get("confidence", 0.5))
        vel_f   = VELOCITY_ATTACK_WEIGHT * float(rms_n) + VELOCITY_CONF_WEIGHT * conf
        note["velocity"]    = int(np.clip(vel_f * 127, VELOCITY_MIN, VELOCITY_MAX))
        note["attack_rms"]  = round(float(rms_n), 4)

    return notes


# ── Harmonic coherence check ──────────────────────────────────────────────────

def check_harmonic_coherence(
    notes: list[dict],
    stem_cqt: np.ndarray,
    sr: int,
    hop: int,
    dominance_ratio: float = 2.5,
    conf_override: float = 0.65,
) -> list[dict]:
    """
    Mark notes that are likely overtones of a lower note.

    For each note at MIDI pitch P, check if the energy one octave below (P-12)
    is `dominance_ratio` times stronger in the stem CQT.  If so, P is likely the
    2nd harmonic of P-12, not a separate note.

    Mutates each note dict with "likely_overtone": bool.
    Removes overtone-marked notes unless confidence >= conf_override.
    Returns filtered list.
    """
    kept = []
    for note in notes:
        conf    = float(note.get("confidence", 0.0))
        midi    = int(note["pitch"])
        t_s     = int(note["start"] * sr / hop)
        t_e     = max(t_s + 1, int((note["start"] + note["duration"]) * sr / hop))

        energy_self = _note_cqt_energy(stem_cqt, midi,     t_s, t_e)
        energy_sub  = _note_cqt_energy(stem_cqt, midi - 12, t_s, t_e)

        is_overtone = (energy_sub > energy_self * dominance_ratio and energy_self > 0)
        note["likely_overtone"] = is_overtone

        if is_overtone and conf < conf_override:
            continue
        kept.append(note)

    return kept


# ── Spectral presence verification ───────────────────────────────────────────

def spectral_presence_gate(
    notes: list[dict],
    stem_cqt: np.ndarray,
    sr: int,
    hop: int,
    min_energy: float = 0.05,
    conf_override: float = 0.60,
    pitch_tol: int = SPECTRAL_PRESENCE_PITCH_TOL,
) -> list[dict]:
    """
    Remove notes whose specific pitch is absent from the stem CQT at note time.

    Unlike the broadband RMS gate (which removes notes in totally silent sections),
    this is pitch-specific: even if the stem has energy (e.g. from another instrument),
    if there is no energy at *this* pitch, the note is a false detection.

    Notes with confidence >= conf_override are kept unconditionally.
    """
    kept = []
    for note in notes:
        conf = float(note.get("confidence", 0.0))
        if conf >= conf_override:
            kept.append(note)
            continue
        midi  = int(note["pitch"])
        t_s   = int(note["start"] * sr / hop)
        t_e   = max(t_s + 1, int((note["start"] + note["duration"]) * sr / hop))
        energy = _note_cqt_energy(stem_cqt, midi, t_s, t_e, pitch_tol=pitch_tol)
        if energy >= min_energy:
            kept.append(note)

    return kept


# ── Attack envelope gate ──────────────────────────────────────────────────────

def attack_envelope_gate(
    notes: list[dict],
    stem_audio: np.ndarray,
    sr: int,
    pre_window_s: float = 0.050,
    post_window_s: float = 0.020,
    min_ratio: float = 1.8,
    conf_override: float = 0.70,
) -> list[dict]:
    """
    Remove notes with no amplitude ramp at their onset.

    Real notes have a pick/key attack: amplitude rises sharply at the onset.
    Noise and bleed start at full amplitude with no ramp (ratio ≈ 1.0).

    Measures: attack_rms / pre_noise_rms.  If ratio < min_ratio → likely noise.
    Notes with confidence >= conf_override are kept unconditionally.
    """
    pre_samples  = max(1, int(pre_window_s  * sr))
    post_samples = max(1, int(post_window_s * sr))
    kept         = []

    for note in notes:
        conf = float(note.get("confidence", 0.0))
        if conf >= conf_override:
            kept.append(note)
            continue

        onset_i      = int(note["start"] * sr)
        pre_start    = max(0, onset_i - pre_samples)
        attack_end   = min(len(stem_audio), onset_i + post_samples)

        pre_segment  = stem_audio[pre_start:onset_i]
        attack_seg   = stem_audio[onset_i:attack_end]

        if len(pre_segment) < 4 or len(attack_seg) < 4:
            # Not enough audio context — keep the note
            kept.append(note)
            continue

        pre_rms    = float(np.sqrt(np.mean(pre_segment ** 2)))
        attack_rms = float(np.sqrt(np.mean(attack_seg ** 2)))
        ratio      = attack_rms / max(pre_rms, 1e-8)

        if ratio >= min_ratio:
            kept.append(note)

    return kept


# ── Pitch-aware isolation filter ─────────────────────────────────────────────

def filter_isolated_notes_pitch_aware(
    notes: list[dict],
    window_s: float = 0.5,
    min_neighbors: int = 2,
    max_interval_st: int = 12,
) -> list[dict]:
    """
    Isolation filter that only counts pitch-close neighbors.

    The original filter counted ALL notes within the time window as neighbors,
    so a ghost note surrounded by distant-pitch real notes would pass.
    This version only counts notes within max_interval_st semitones as "close."

    A note is kept if it has at least min_neighbors pitch-close neighbors
    within ±window_s of its start time.
    """
    if not notes:
        return notes

    sorted_notes = sorted(notes, key=lambda n: n["start"])
    starts       = [n["start"] for n in sorted_notes]
    n_total      = len(sorted_notes)
    keep_flags   = [False] * n_total

    import bisect
    for i, note in enumerate(sorted_notes):
        t       = note["start"]
        pitch_i = int(note["pitch"])

        lo = bisect.bisect_left(starts, t - window_s)
        hi = bisect.bisect_right(starts, t + window_s)

        close_count = 0
        for j in range(lo, hi):
            if j == i:
                continue
            if abs(int(sorted_notes[j]["pitch"]) - pitch_i) <= max_interval_st:
                close_count += 1
                if close_count >= min_neighbors:
                    break

        if close_count >= min_neighbors:
            keep_flags[i] = True

    return [n for n, keep in zip(sorted_notes, keep_flags) if keep]
