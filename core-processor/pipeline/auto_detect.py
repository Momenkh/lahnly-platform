"""
Stage 1.5 — Automatic Guitar Type + Role Detection
====================================================
Infers guitar_type (acoustic | clean | distorted) and guitar_role (lead | rhythm)
from the separated guitar stem using fast librosa spectral features.

No additional model or network download required — runs in ~2-3 seconds on a
60-second stem clip.

Features used
-------------
Type detection:
  spectral_flatness  — distorted guitar has a noise-like spectrum (high flatness)
                       even after stem separation; acoustic/clean are much lower
  spectral_contrast  — acoustic and clean have clear harmonic peaks vs valleys;
                       distortion compresses the spectrum reducing contrast
  spectral_centroid  — acoustic body resonance shifts the average energy centroid

Role detection:
  chroma polyphony   — active simultaneous chroma bins per frame:
                       rhythm chords activate many pitch classes at once;
                       lead lines are mono/duo-phonic
  onset density      — dense strumming (rhythm) vs sparse runs (lead)

Confidence
----------
Both type and role come with an independent confidence score (0–1).
Values below 0.60 indicate ambiguity — the caller may want to warn the user
or fall back to a default.
"""

import numpy as np

# ── Thresholds ────────────────────────────────────────────────────────────────
# Spectral flatness (median, linear 0–1) thresholds for distortion detection.
# These apply to the separated stem (not raw mix) — values are lower than raw
# because Demucs already suppresses non-guitar content.
_FLATNESS_DISTORTED   = 0.07   # above this → distorted
_FLATNESS_DISTORTED_H = 0.14   # above this → high-confidence distorted

# Spectral contrast (mean dB, 4 bands) threshold separating acoustic from clean.
# Acoustic guitars have stronger harmonic peaks relative to valleys than clean electric.
_CONTRAST_ACOUSTIC = 26.0

# Spectral centroid (Hz) — acoustic tends to be brighter due to body resonance.
_CENTROID_ACOUSTIC = 2600.0

# Chroma polyphony threshold: median active chroma bins per active frame.
# Rhythm chords typically activate 3+ pitch classes simultaneously.
_POLY_RHYTHM = 3.2


def detect_guitar_mode(stem_path: str) -> dict:
    """
    Classify guitar type and role from a separated stem file.

    Parameters
    ----------
    stem_path : str
        Path to the guitar stem WAV (output of Stage 1 Demucs separation).

    Returns
    -------
    dict with keys:
        guitar_type      : "acoustic" | "clean" | "distorted"
        guitar_role      : "lead" | "rhythm"
        type_confidence  : float 0–1
        role_confidence  : float 0–1
        features         : dict of raw feature values (for logging / debug)
    """
    import librosa

    # Load at most 60s — most songs have consistent tonal character throughout,
    # and 60s gives enough frames for reliable statistics at minimal cost.
    y, sr = librosa.load(stem_path, sr=22050, mono=True, duration=60.0)

    # ── Type features ─────────────────────────────────────────────────────────

    flatness = librosa.feature.spectral_flatness(y=y)
    med_flatness = float(np.median(flatness))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4)
    mean_contrast = float(np.mean(contrast))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = float(np.mean(centroid))

    # ── Type classification ───────────────────────────────────────────────────

    if med_flatness >= _FLATNESS_DISTORTED:
        guitar_type = "distorted"
        # Confidence scales from 0.55 at threshold to 1.0 above the high mark
        type_conf = 0.55 + 0.45 * min(1.0, (med_flatness - _FLATNESS_DISTORTED)
                                           / (_FLATNESS_DISTORTED_H - _FLATNESS_DISTORTED))
    else:
        # Not distorted — distinguish acoustic vs clean
        # Acoustic has higher spectral contrast (clear harmonic structure) and/or
        # brighter centroid (string + body resonance).
        acoustic_votes = 0
        if mean_contrast >= _CONTRAST_ACOUSTIC:
            acoustic_votes += 1
        if mean_centroid >= _CENTROID_ACOUSTIC:
            acoustic_votes += 1

        if acoustic_votes >= 1:
            guitar_type = "acoustic"
            type_conf   = 0.60 + 0.10 * acoustic_votes   # 0.70 or 0.80
        else:
            guitar_type = "clean"
            type_conf   = 0.65

    # ── Role features ─────────────────────────────────────────────────────────

    # Chroma CQT — use coarse hop for speed
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

    # Normalise each frame column so the threshold is pitch-class relative, not
    # amplitude-dependent.  Frames with very low energy are excluded.
    col_max   = np.max(chroma, axis=0)
    energy_mask = col_max > np.percentile(col_max, 20)   # skip silent frames
    chroma_norm = np.where(col_max > 0, chroma / col_max, 0.0)
    active_bins = np.sum(chroma_norm[:, energy_mask] > 0.50, axis=0)
    med_polyphony = float(np.median(active_bins)) if active_bins.size > 0 else 1.0

    # Onset density (onsets per second) — dense strumming → rhythm
    onset_times   = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration_s    = len(y) / sr
    onset_density = len(onset_times) / max(duration_s, 1.0)

    # Combined polyphony score: primary = median active bins, secondary = density
    poly_score = med_polyphony + 0.25 * min(onset_density / 4.0, 1.0)

    # ── Role classification ───────────────────────────────────────────────────

    margin = abs(poly_score - _POLY_RHYTHM)
    role_conf = 0.55 + 0.35 * min(1.0, margin / 1.5)

    if poly_score >= _POLY_RHYTHM:
        guitar_role = "rhythm"
    else:
        guitar_role = "lead"

    return {
        "guitar_type":     guitar_type,
        "guitar_role":     guitar_role,
        "type_confidence": round(type_conf, 2),
        "role_confidence": round(role_conf, 2),
        "features": {
            "spectral_flatness":   round(med_flatness, 4),
            "spectral_contrast":   round(mean_contrast, 2),
            "spectral_centroid_hz": round(mean_centroid, 1),
            "median_polyphony":    round(med_polyphony, 2),
            "onset_density_per_s": round(onset_density, 2),
        },
    }
