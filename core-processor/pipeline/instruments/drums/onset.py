"""
Stage 2 (Drums): Onset Detection + Hit Classification
Drums bypass the pitch pipeline. This module replaces Stages 2–7.
Output: [{hit_class, start, confidence}]
        saved to outputs/02_drums_hits.json
"""

import json
import os
import numpy as np

from pipeline.config import get_instrument_dir
from pipeline.settings import (
    DRUMS_ONSET_DELTA, DRUMS_MIN_INTERVAL_S, DRUMS_ONSET_BACKTRACK,
    DRUMS_HIT_ANALYSIS_WINDOW_S,
    DRUMS_CENTROID_KICK_MAX, DRUMS_CENTROID_TOM_MAX, DRUMS_CENTROID_HIHAT_MIN,
    DRUMS_FLATNESS_SNARE_MIN, DRUMS_FLATNESS_HIHAT_MIN,
    DRUMS_HIHAT_OPEN_FLUX_THRESH,
    DRUMS_CONF_CLASS_WEIGHT, DRUMS_CONF_STRENGTH_WEIGHT,
    DRUMS_CONF_KICK_BASE, DRUMS_CONF_KICK_SCALE,
    DRUMS_CONF_HIHAT_BASE, DRUMS_CONF_CYMBAL_BASE,
    DRUMS_CONF_SNARE_BASE, DRUMS_CONF_SNARE_SCALE, DRUMS_CONF_SNARE_FLAT_RANGE,
    DRUMS_CONF_TOM_BASE, DRUMS_CONF_DEFAULT_BASE,
)

_SR = 22050


def detect_drum_hits(audio_path: str, save: bool = True) -> list[dict]:
    """
    Detect drum hit onsets and classify each as kick/snare/hihat/tom/cymbal.
    Returns list of {hit_class, start, confidence}.
    """
    import librosa

    print(f"[Stage 2D] Loading drum stem: {audio_path}")
    y, sr = librosa.load(audio_path, sr=_SR, mono=True)
    duration_s = len(y) / sr
    print(f"[Stage 2D] Audio: {duration_s:.1f}s @ {sr}Hz")

    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        delta=DRUMS_ONSET_DELTA,
        backtrack=DRUMS_ONSET_BACKTRACK,
        units="frames",
        hop_length=512,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    # Enforce minimum interval between hits
    onset_times = _enforce_min_interval(onset_times, DRUMS_MIN_INTERVAL_S)
    print(f"[Stage 2D] Detected {len(onset_times)} onsets")

    # Onset strength for confidence scoring
    onset_env    = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset_frames_filtered = librosa.time_to_frames(onset_times, sr=sr, hop_length=512)
    max_strength = float(np.max(onset_env)) if len(onset_env) > 0 else 1.0

    hits = []
    win_samples = int(DRUMS_HIT_ANALYSIS_WINDOW_S * sr)

    for i, t in enumerate(onset_times):
        start_sample = int(t * sr)
        end_sample   = min(len(y), start_sample + win_samples)
        window       = y[start_sample:end_sample]

        if len(window) < 64:
            continue

        hit_class, class_conf = _classify_hit(window, sr)

        # Confidence: blend onset strength with classifier confidence
        frame_idx = min(onset_frames_filtered[i], len(onset_env) - 1)
        strength_conf = float(onset_env[frame_idx]) / max(max_strength, 1e-6)
        confidence    = round(DRUMS_CONF_CLASS_WEIGHT * class_conf + DRUMS_CONF_STRENGTH_WEIGHT * strength_conf, 3)

        hits.append({
            "hit_class":  hit_class,
            "start":      round(float(t), 4),
            "confidence": confidence,
            "velocity":   round(strength_conf, 3),
        })

    hits.sort(key=lambda h: h["start"])

    # Print summary
    from collections import Counter
    counts = Counter(h["hit_class"] for h in hits)
    print(f"[Stage 2D] Hit classification: " +
          "  ".join(f"{cls}={cnt}" for cls, cnt in sorted(counts.items())))

    if save:
        out_path = os.path.join(get_instrument_dir("drums"), "02_hits.json")
        with open(out_path, "w") as f:
            json.dump(hits, f, indent=2)
        print(f"[Stage 2D] Saved -> {out_path}  ({len(hits)} hits)")

    return hits


def _classify_hit(window: np.ndarray, sr: int) -> tuple[str, float]:
    """
    Classify a drum hit based on spectral centroid and flatness.
    Returns (hit_class, confidence).
    """
    import librosa

    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=window, sr=sr)))
    flatness  = float(np.median(librosa.feature.spectral_flatness(y=window)))

    # Decision tree: centroid is the primary feature
    if centroid <= DRUMS_CENTROID_KICK_MAX:
        return "kick", DRUMS_CONF_KICK_BASE + DRUMS_CONF_KICK_SCALE * (1.0 - centroid / DRUMS_CENTROID_KICK_MAX)

    if centroid >= DRUMS_CENTROID_HIHAT_MIN:
        if flatness >= DRUMS_FLATNESS_HIHAT_MIN:
            # Open hi-hat has longer decay → higher frame-to-frame spectral flux
            flux = float(np.mean(np.abs(np.diff(window))))
            hit_class = "hihat_open" if flux > DRUMS_HIHAT_OPEN_FLUX_THRESH else "hihat_closed"
            return hit_class, DRUMS_CONF_HIHAT_BASE
        return "cymbal", DRUMS_CONF_CYMBAL_BASE   # sustained high-freq = ride/crash

    # Mid-centroid: snare vs tom discrimination via flatness
    if flatness >= DRUMS_FLATNESS_SNARE_MIN:
        return "snare", DRUMS_CONF_SNARE_BASE + DRUMS_CONF_SNARE_SCALE * min(1.0, (flatness - DRUMS_FLATNESS_SNARE_MIN) / DRUMS_CONF_SNARE_FLAT_RANGE)

    if centroid <= DRUMS_CENTROID_TOM_MAX:
        return "tom", DRUMS_CONF_TOM_BASE

    return "snare", DRUMS_CONF_DEFAULT_BASE   # default mid-band = snare


def _enforce_min_interval(times: np.ndarray, min_interval: float) -> np.ndarray:
    if len(times) == 0:
        return times
    filtered = [times[0]]
    for t in times[1:]:
        if t - filtered[-1] >= min_interval:
            filtered.append(t)
    return np.array(filtered)


def load_drum_hits() -> list[dict]:
    path = os.path.join(get_instrument_dir("drums"), "02_hits.json")
    with open(path) as f:
        return json.load(f)
