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


# ── Bass-specific thresholds ──────────────────────────────────────────────────
# Spectral flatness: slap bass has the highest flatness (sharp transients +
# pop harmonics produce a noise-like spectrum), fingered bass the lowest.
_BASS_FLATNESS_SLAP      = 0.05   # above this → slap candidate
_BASS_FLATNESS_SLAP_H    = 0.10   # above this → high-confidence slap

# Attack transient sharpness (onset strength mean): picked bass has crisp attacks.
# Below librosa's default unit, typical values: fingered ~0.15, picked ~0.25+.
_BASS_ONSET_PICKED       = 0.22   # mean onset strength above this → picked candidate

# ── Thresholds ────────────────────────────────────────────────────────────────
# Spectral flatness (median, linear 0–1) thresholds for distortion detection.
# These apply to the separated stem (not raw mix) — values are lower than raw
# because Demucs already suppresses non-guitar content.
_FLATNESS_DISTORTED   = 0.07   # above this → distorted
_FLATNESS_DISTORTED_H = 0.14   # above this → high-confidence distorted

# Spectral contrast (mean dB, 4 bands).
# After Demucs separation, acoustic guitars show HIGH contrast (30+) due to
# clear harmonic structure; clean electrics tend to fall in the 24–28 range.
# Require a higher bar to avoid false acoustic positives on clean electric.
_CONTRAST_ACOUSTIC = 29.0

# Spectral centroid (Hz) — on separated stems, acoustic body resonance keeps
# the centroid LOW (600–900 Hz); clean electric lead sits higher (1500–2500 Hz).
# Flag as acoustic when centroid is below this value (NOT above — opposite of
# the initial assumption).
_CENTROID_ACOUSTIC_MAX = 1200.0

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
        # Not distorted — distinguish acoustic vs clean.
        # On Demucs-separated stems, acoustic shows high contrast (clear harmonics
        # from string + body resonance) AND low centroid (body-resonant low-mids).
        # Clean electric sits in intermediate contrast with a higher centroid.
        acoustic_votes = 0
        if mean_contrast >= _CONTRAST_ACOUSTIC:
            acoustic_votes += 1
        if mean_centroid <= _CENTROID_ACOUSTIC_MAX:
            acoustic_votes += 1

        if acoustic_votes >= 2:
            # Both signals agree → confident acoustic
            guitar_type = "acoustic"
            type_conf   = 0.80
        elif acoustic_votes == 1 and mean_contrast >= _CONTRAST_ACOUSTIC:
            # Strong contrast alone is sufficient (high contrast + body-like harmonics)
            guitar_type = "acoustic"
            type_conf   = 0.65
        else:
            guitar_type = "clean"
            type_conf   = 0.70

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


def detect_vocals_mode(stem_path: str) -> dict:
    """
    Classify vocal type from a separated vocals stem.

    Returns
    -------
    dict with keys:
        vocals_mode      : "vocals_lead" | "vocals_harmony"
        mode_confidence  : float 0–1
        features         : dict of raw feature values
    """
    import librosa

    y, sr = librosa.load(stem_path, sr=22050, mono=True, duration=60.0)

    # RMS energy — lead vocals are louder relative to the stem ceiling
    rms        = librosa.feature.rms(y=y)[0]
    mean_rms   = float(np.mean(rms))
    max_rms    = float(np.max(rms)) if len(rms) > 0 else 1e-6
    rms_ratio  = mean_rms / max(max_rms, 1e-6)   # how consistently loud (dense)

    # Spectral flatness — harmony parts have more tonal similarity to instruments
    flatness     = librosa.feature.spectral_flatness(y=y)
    med_flatness = float(np.median(flatness))

    # Lead vocals tend to be louder relative to their own peak and less spectrally flat.
    # Harmony parts are quieter (blend role) and slightly more tonal.
    _RMS_RATIO_LEAD    = 0.35   # above this → consistent energy → lead
    _FLATNESS_LEAD_MAX = 0.08   # below this → tonal → lead (not noise-like)

    lead_votes = 0
    if rms_ratio >= _RMS_RATIO_LEAD:
        lead_votes += 1
    if med_flatness <= _FLATNESS_LEAD_MAX:
        lead_votes += 1

    if lead_votes >= 1:
        vocals_mode = "vocals_lead"
        mode_conf   = 0.60 + 0.20 * lead_votes
    else:
        vocals_mode = "vocals_harmony"
        mode_conf   = 0.65

    return {
        "vocals_mode":      vocals_mode,
        "mode_confidence":  round(min(mode_conf, 1.0), 2),
        "features": {
            "mean_rms":          round(mean_rms, 4),
            "rms_ratio":         round(rms_ratio, 3),
            "spectral_flatness": round(med_flatness, 4),
        },
    }


def detect_drums_mode(stem_path: str) -> dict:
    """
    Minimal characterization of a drum stem — returns onset density and
    a rough energy profile. Drums have no pitch mode; this is used for logging.

    Returns
    -------
    dict with keys:
        drums_character  : "sparse" | "dense"
        mode_confidence  : float 0–1
        features         : dict of raw feature values
    """
    import librosa

    y, sr = librosa.load(stem_path, sr=22050, mono=True, duration=60.0)

    onset_times   = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration_s    = len(y) / sr
    onset_density = len(onset_times) / max(duration_s, 1.0)

    # Spectral centroid — crash-heavy kits have a high centroid; kick-heavy are low
    centroid     = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = float(np.mean(centroid))

    _DENSE_ONSET_THRESH = 3.0   # > 3 hits/s = dense kit
    if onset_density >= _DENSE_ONSET_THRESH:
        drums_char = "dense"
        mode_conf  = 0.60 + 0.20 * min(1.0, (onset_density - _DENSE_ONSET_THRESH) / 3.0)
    else:
        drums_char = "sparse"
        mode_conf  = 0.70

    return {
        "drums_character": drums_char,
        "mode_confidence": round(mode_conf, 2),
        "features": {
            "onset_density_per_s": round(onset_density, 2),
            "mean_centroid_hz":    round(mean_centroid, 1),
        },
    }


def detect_piano_mode(stem_path: str) -> dict:
    """
    Classify piano playing style from a separated piano stem.

    Returns
    -------
    dict with keys:
        piano_mode       : "piano_melody" | "piano_chord"
        mode_confidence  : float 0–1
        features         : dict of raw feature values
    """
    import librosa

    y, sr = librosa.load(stem_path, sr=22050, mono=True, duration=60.0)

    # Chroma polyphony — how many pitch classes active per frame
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    col_max     = np.max(chroma, axis=0)
    energy_mask = col_max > np.percentile(col_max, 20)
    chroma_norm = np.where(col_max > 0, chroma / col_max, 0.0)
    active_bins = np.sum(chroma_norm[:, energy_mask] > 0.50, axis=0)
    med_polyphony = float(np.median(active_bins)) if active_bins.size > 0 else 1.0

    # Onset density — dense events (fast runs) vs sparse (chords held)
    onset_times   = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration_s    = len(y) / sr
    onset_density = len(onset_times) / max(duration_s, 1.0)

    # Combined score: high polyphony + moderate density → chord; low poly → melody
    # Melody lines usually have 1–2 active chroma bins; chords activate 3+
    _POLY_PIANO_CHORD = 3.0

    margin    = abs(med_polyphony - _POLY_PIANO_CHORD)
    mode_conf = 0.55 + 0.35 * min(1.0, margin / 1.5)

    if med_polyphony >= _POLY_PIANO_CHORD:
        piano_mode = "piano_chord"
    else:
        piano_mode = "piano_melody"

    return {
        "piano_mode":       piano_mode,
        "mode_confidence":  round(mode_conf, 2),
        "features": {
            "median_polyphony":    round(med_polyphony, 2),
            "onset_density_per_s": round(onset_density, 2),
        },
    }


def detect_bass_mode(stem_path: str) -> dict:
    """
    Classify bass playing style from a separated bass stem.

    Returns
    -------
    dict with keys:
        bass_style       : "bass_fingered" | "bass_picked" | "bass_slap"
        style_confidence : float 0–1
        features         : dict of raw feature values
    """
    import librosa

    y, sr = librosa.load(stem_path, sr=22050, mono=True, duration=60.0)

    # Spectral flatness: slap produces broadband transients → high flatness
    flatness     = librosa.feature.spectral_flatness(y=y)
    med_flatness = float(np.median(flatness))

    # Onset strength: picked bass has sharp, high-amplitude onsets
    onset_env        = librosa.onset.onset_strength(y=y, sr=sr)
    mean_onset_str   = float(np.mean(onset_env))

    # Style classification
    if med_flatness >= _BASS_FLATNESS_SLAP:
        bass_style = "bass_slap"
        style_conf = 0.55 + 0.35 * min(1.0,
            (med_flatness - _BASS_FLATNESS_SLAP)
            / (_BASS_FLATNESS_SLAP_H - _BASS_FLATNESS_SLAP))
    elif mean_onset_str >= _BASS_ONSET_PICKED:
        bass_style = "bass_picked"
        style_conf = 0.60 + 0.20 * min(1.0,
            (mean_onset_str - _BASS_ONSET_PICKED) / 0.15)
    else:
        bass_style = "bass_fingered"
        style_conf = 0.70   # default — most bass playing is fingered

    return {
        "bass_style":       bass_style,
        "style_confidence": round(style_conf, 2),
        "features": {
            "spectral_flatness":  round(med_flatness, 4),
            "mean_onset_strength": round(mean_onset_str, 3),
        },
    }
