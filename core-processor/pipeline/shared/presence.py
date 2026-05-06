"""
Stem Presence Detection
=======================
Determines whether a target instrument is actually present in a Demucs stem
before running expensive pitch extraction.

A Demucs stem of a missing instrument contains residual mix bleed that can fool
the pitch extractor into producing hundreds of ghost notes.  Three independent
signals are combined (energy weighted 2×, flatness and onsets each 1×):

  1. energy_ratio_raw   — stem_rms / mix_rms at separation time (pre-normalization),
                          stored in 01_stem_meta.json as "energy_ratio_raw".
  2. spectral_flatness  — mean spectral flatness of the stem WAV.
                          Low = harmonic structure (real instrument).
                          High = flat / noisy = residual bleed.
  3. onset_density_s    — detected onsets per second in the stem WAV.
                          Real instruments have clear attack transients.

is_present = (2×energy_vote + flatness_vote + onset_vote) >= 3

Call check_and_update_stem_presence(instrument) at the top of each pitch
extraction function.  Results are cached in 01_stem_meta.json so repeated
--from-stage runs don't re-analyse.
"""

import json
import os

import numpy as np

from pipeline.config import get_instrument_dir
from pipeline.settings.shared.presence import (
    PRESENCE_HARD_ENERGY_FLOOR,
    PRESENCE_ANALYSIS_MAX_S,
    PRESENCE_ANALYSIS_SKIP_S,
    PRESENCE_THRESHOLDS,
    PRESENCE_CROSS_STEM_RIVALS,
    PRESENCE_CROSS_STEM_SIM_THRESH,
)


# ── Public API ────────────────────────────────────────────────────────────────

def check_and_update_stem_presence(instrument: str) -> bool:
    """
    Return True if the instrument should be processed (stem is present).

    Reads 01_stem_meta.json.  If 'is_present' is already cached there, returns
    it immediately.  Otherwise runs the 3-signal detection, writes the result
    back to the meta file, and returns the flag.

    Designed to be called at the very start of each instrument's pitch
    extraction function — before any model inference.
    """
    from pipeline.shared.separation import load_stem_meta_for, get_stem_path_for

    meta      = load_stem_meta_for(instrument)
    meta_path = os.path.join(get_instrument_dir(instrument), "01_stem_meta.json")

    if "is_present" in meta:
        if not meta["is_present"]:
            _log_absent(instrument, meta)
        return meta["is_present"]

    stem_path      = get_stem_path_for(instrument)
    energy_ratio   = meta.get("energy_ratio_raw")   # saved by separation stage
    presence_info  = _detect(stem_path, energy_ratio, instrument)

    # Cross-stem bleed check: if this stem looks like a rival instrument's stem,
    # the instrument is absent and the content is bleed from that rival.
    if presence_info["is_present"]:
        for rival in PRESENCE_CROSS_STEM_RIVALS.get(instrument, []):
            rival_path = get_stem_path_for(rival)
            if not os.path.isfile(rival_path):
                continue
            sim = _cross_stem_similarity(stem_path, rival_path)
            if sim >= PRESENCE_CROSS_STEM_SIM_THRESH:
                presence_info["is_present"]     = False
                presence_info["presence_score"] = 0.0
                presence_info["cross_stem_sim_vs"] = rival
                presence_info["cross_stem_sim"]    = round(sim, 4)
                presence_info["absence_reason"] = (
                    f"stem similarity to {rival} = {sim:.4f} >= {PRESENCE_CROSS_STEM_SIM_THRESH} "
                    f"(piano bleed into {instrument} channel)"
                )
                break

    meta.update(presence_info)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    status = "PRESENT" if presence_info["is_present"] else "ABSENT"
    er    = presence_info.get("energy_ratio_raw", "n/a")
    sf    = presence_info.get("spectral_flatness",  "n/a")
    od    = presence_info.get("onset_density_s",    "n/a")
    score = presence_info.get("presence_score", "n/a")
    xsim  = presence_info.get("cross_stem_sim")
    xvs   = presence_info.get("cross_stem_sim_vs")

    er_str = f"{er:.4f}" if isinstance(er, float) else str(er)
    sf_str = f"{sf:.4f}" if isinstance(sf, float) else str(sf)
    od_str = f"{od:.2f}"  if isinstance(od, float) else str(od)

    extra = f", cross_sim_vs_{xvs}={xsim:.4f}" if xsim is not None else ""
    print(
        f"[Presence] {instrument}: {status}  "
        f"(score={score}, energy={er_str}, flatness={sf_str}, onsets/s={od_str}{extra})"
    )
    if not presence_info["is_present"]:
        _log_absent(instrument, presence_info)

    return presence_info["is_present"]


# ── Internal ──────────────────────────────────────────────────────────────────

def _log_absent(instrument: str, info: dict) -> None:
    reason = info.get("absence_reason", "low presence score")
    print(f"[Presence] Skipping {instrument} pitch extraction — instrument not detected ({reason})")


def _detect(stem_path: str, energy_ratio_raw: float | None, instrument: str) -> dict:
    """
    Run the 3-signal detection and return a presence dict:
      {
        is_present:       bool,
        presence_score:   float (0–1, fraction of max weighted vote),
        energy_ratio_raw: float | None,
        spectral_flatness: float | None,
        onset_density_s:  float | None,
        absence_reason:   str  (only when is_present=False),
      }
    """
    thresholds = PRESENCE_THRESHOLDS.get(instrument, (0.035, 0.25, 0.30))
    energy_thresh, flatness_thresh, onset_thresh = thresholds

    # ── Signal 1: energy ratio ────────────────────────────────────────────────
    if energy_ratio_raw is None:
        # stem_meta predates this feature — fall back to loading the stem WAV
        energy_ratio_raw = _compute_stem_rms_proxy(stem_path)

    if energy_ratio_raw is not None and energy_ratio_raw < PRESENCE_HARD_ENERGY_FLOOR:
        return {
            "is_present":        False,
            "presence_score":    0.0,
            "energy_ratio_raw":  round(energy_ratio_raw, 5),
            "spectral_flatness": None,
            "onset_density_s":   None,
            "absence_reason":    f"energy_ratio_raw={energy_ratio_raw:.4f} < hard floor {PRESENCE_HARD_ENERGY_FLOOR}",
        }

    energy_vote = 1 if (energy_ratio_raw is not None and energy_ratio_raw >= energy_thresh) else 0

    # ── Signals 2 & 3: load stem WAV ─────────────────────────────────────────
    spectral_flatness = None
    onset_density     = None
    flatness_vote     = 1   # default present if we can't compute
    onset_vote        = 1

    if os.path.isfile(stem_path):
        try:
            import librosa
            y, sr = librosa.load(stem_path, sr=22050, mono=True)

            skip    = int(PRESENCE_ANALYSIS_SKIP_S * sr)
            max_len = int(PRESENCE_ANALYSIS_MAX_S  * sr)
            y_clip  = y[skip : skip + max_len] if len(y) > skip + sr else y

            if len(y_clip) >= sr:
                # Spectral flatness
                flatness_frames   = librosa.feature.spectral_flatness(y=y_clip)
                spectral_flatness = float(np.mean(flatness_frames))
                flatness_vote     = 1 if spectral_flatness < flatness_thresh else 0

                # Onset density
                onsets        = librosa.onset.onset_detect(y=y_clip, sr=sr, units="time")
                duration_s    = len(y_clip) / sr
                onset_density = len(onsets) / max(duration_s, 1.0)
                onset_vote    = 1 if onset_density >= onset_thresh else 0

        except Exception as exc:
            print(f"[Presence] Warning: could not analyse {instrument} stem ({exc})")

    # ── Vote ──────────────────────────────────────────────────────────────────
    weighted_score = 2 * energy_vote + flatness_vote + onset_vote
    is_present     = weighted_score >= 3      # max possible = 4

    result = {
        "is_present":        is_present,
        "presence_score":    round(weighted_score / 4, 3),
        "energy_ratio_raw":  round(energy_ratio_raw, 5) if energy_ratio_raw is not None else None,
        "spectral_flatness": round(spectral_flatness, 5) if spectral_flatness is not None else None,
        "onset_density_s":   round(onset_density, 3)    if onset_density    is not None else None,
    }
    if not is_present:
        votes = f"energy={energy_vote}x2, flatness={flatness_vote}, onsets={onset_vote}, score={weighted_score}/4"
        result["absence_reason"] = votes

    return result


def _cross_stem_similarity(stem_a: str, stem_b: str) -> float:
    """
    Cosine similarity of log-mel spectrograms between two stems.
    Loaded at 16 kHz, 64 mel bands, middle 60 seconds only (for speed).
    Returns 0.0 on any error.
    """
    try:
        import librosa
        skip = int(PRESENCE_ANALYSIS_SKIP_S * 16000)
        n    = int(PRESENCE_ANALYSIS_MAX_S  * 16000)
        y_a, sr = librosa.load(stem_a, sr=16000, mono=True)
        y_b, _  = librosa.load(stem_b, sr=16000, mono=True)

        y_a = y_a[skip : skip + n]
        y_b = y_b[skip : skip + n]

        if len(y_a) < sr or len(y_b) < sr:
            return 0.0

        mel_a = librosa.feature.melspectrogram(y=y_a, sr=sr, n_mels=64)
        mel_b = librosa.feature.melspectrogram(y=y_b, sr=sr, n_mels=64)

        n_frames = min(mel_a.size, mel_b.size)
        a = mel_a.flatten()[:n_frames]
        b = mel_b.flatten()[:n_frames]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a / norm_a, b / norm_b))
    except Exception:
        return 0.0


def _compute_stem_rms_proxy(stem_path: str) -> float | None:
    """
    Fallback when energy_ratio_raw was not saved in stem_meta (old runs).
    Loads the stem WAV, computes normalized RMS, and returns it.
    Note: this is the post-normalization RMS, not the pre-normalization ratio,
    so it's less reliable than the separation-time value but still useful.
    """
    if not os.path.isfile(stem_path):
        return None
    try:
        import soundfile as sf
        audio, _ = sf.read(stem_path, always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32)
        rms  = float(np.sqrt(np.mean(mono ** 2)))
        # Treat post-norm RMS as a proxy for energy_ratio_raw.
        # Real stems normalised to peak=0.95 have RMS ~0.05–0.25.
        # Ghost stems have RMS ~0.003–0.02 (noise spike drove the normalisation).
        return rms
    except Exception:
        return None
