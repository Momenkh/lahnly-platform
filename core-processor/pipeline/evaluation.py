"""
Evaluation — accuracy scoring for all instruments.

Public API
----------
score_transcription(stem, preview)          → float   legacy chroma similarity (0–1)
score_cqt_similarity(stem, preview)         → float   octave-aware CQT similarity (0–1)
score_drums(stem, hits)                     → float   onset F1 (0–1)
score_transcription_v2(stem, preview, notes)→ dict    full metric bundle (use this)
score_instrument(instrument, outputs_dir)   → dict    convenience wrapper (legacy)
score_pipeline_stages(outputs_dir, instr)   → dict    per-stage precision delta

Metric definitions
------------------
chroma_sim       : chroma cosine similarity between stem and synthesized preview.
                   Pitch-CLASS only — C4 and C5 are identical. Kept for backward compat.
cqt_sim          : CQT magnitude cosine similarity. Octave-aware: C4 ≠ C5.
                   This is the primary metric going forward.
spectral_precision: fraction of detected notes whose pitch exists in the stem CQT
                   at the note's time position (ghost-note proxy).
spectral_recall  : fraction of librosa-detected stem onsets that are covered by
                   a detected note within ±50ms (missed-note proxy).
pseudo_f1        : harmonic mean of spectral_precision and spectral_recall.
per_pitch_class  : per-semitone chroma similarity (C, C#, D, …, B).
high_note_precision: spectral_precision restricted to MIDI ≥ 69 (A4+).
temporal_clusters: (start_s, end_s, local_cqt_sim) windows where errors cluster.
"""

import json
import os

import numpy as np
import librosa

from pipeline.settings import (
    EVAL_ACTIVE_THRESH,
    EVAL_MAX_SHIFT_FRAMES,
    EVAL_ONSET_TOLERANCE_S,
    EVAL_CQT_N_BINS,
    EVAL_CQT_BINS_PER_OCTAVE,
    EVAL_CQT_HOP,
    EVAL_SPECTRAL_HIT_THRESH,
    EVAL_SPECTRAL_PITCH_TOL_BINS,
    EVAL_SPECTRAL_RECALL_ONSET_DELTA,
    EVAL_TEMPORAL_CLUSTER_WIN_S,
    EVAL_TEMPORAL_CLUSTER_THRESHOLD,
    EVAL_PER_PC_MIN_REF_ENERGY,
)

_CHROMA_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# MIDI pitch of the lowest CQT bin (C1).
_CQT_MIDI_MIN = 24


# ── Public: legacy single-float scores ───────────────────────────────────────

def score_transcription(stem_path: str, preview_path: str) -> float:
    """
    Chroma cosine similarity between stem and synthesized preview (legacy).
    Returns float in [0, 1].  Octave errors are invisible to this metric.
    Prefer score_transcription_v2() for new code.
    """
    hop = 512
    stem, sr = librosa.load(stem_path, sr=22050, mono=True)
    pred, _  = librosa.load(preview_path, sr=22050, mono=True)

    C_ref  = librosa.feature.chroma_cqt(y=stem, sr=sr, hop_length=hop)
    C_pred = librosa.feature.chroma_cqt(y=pred, sr=sr, hop_length=hop)

    best = 0.0
    for shift in range(-EVAL_MAX_SHIFT_FRAMES, EVAL_MAX_SHIFT_FRAMES + 1):
        s = _score_at_shift(C_ref, C_pred, shift)
        if s > best:
            best = s
    return best


def score_cqt_similarity(stem_path: str, preview_path: str) -> float:
    """
    CQT-magnitude cosine similarity between stem and synthesized preview.
    Octave-aware: C4 and C5 produce energy in different bins, so octave
    errors are penalised unlike chroma.
    Returns float in [0, 1].
    """
    stem, sr = librosa.load(stem_path, sr=22050, mono=True)
    pred, _  = librosa.load(preview_path, sr=22050, mono=True)
    C_ref, C_pred = _compute_cqt_pair(stem, pred, sr)

    best = 0.0
    for shift in range(-EVAL_MAX_SHIFT_FRAMES, EVAL_MAX_SHIFT_FRAMES + 1):
        s = _score_at_shift(C_ref, C_pred, shift)
        if s > best:
            best = s
    return best


def score_drums(stem_path: str, detected_hits: list[dict]) -> float:
    """
    Evaluate drum onset detection quality via F1 score.
    Reference onsets are extracted from the stem using librosa — this measures
    self-consistency as a proxy for detection completeness.
    Returns F1 in [0, 1].
    """
    hop = 512
    y, sr = librosa.load(stem_path, sr=22050, mono=True)
    ref_times  = librosa.onset.onset_detect(y=y, sr=sr, units="time",
                                            hop_length=hop, delta=0.05)
    pred_times = np.array([h["start"] for h in detected_hits])

    if len(ref_times) == 0 and len(pred_times) == 0:
        return 1.0
    if len(ref_times) == 0 or len(pred_times) == 0:
        return 0.0

    tp = _match_onsets(ref_times, pred_times, EVAL_ONSET_TOLERANCE_S)
    precision = tp / max(len(pred_times), 1)
    recall    = tp / max(len(ref_times),  1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# ── Public: rich v2 score dict ────────────────────────────────────────────────

def score_transcription_v2(
    stem_path: str,
    preview_path: str,
    cleaned_notes: list[dict] | None = None,
) -> dict:
    """
    Full metric bundle for a single instrument transcription.

    Parameters
    ----------
    stem_path      : path to the separated stem WAV (ground truth proxy)
    preview_path   : path to the synthesized preview WAV
    cleaned_notes  : list of note dicts [{pitch, start, duration, confidence}, ...]
                     Optional — enables spectral_precision, spectral_recall, pseudo_f1,
                     high_note_precision, and per-note spec_hit annotation.

    Returns dict with keys:
        chroma_sim, cqt_sim, spectral_precision, spectral_recall, pseudo_f1,
        per_pitch_class, high_note_precision, temporal_clusters
    """
    hop = EVAL_CQT_HOP

    stem, sr = librosa.load(stem_path, sr=22050, mono=True)
    pred, _  = librosa.load(preview_path, sr=22050, mono=True)

    # Chroma (legacy, backward compat)
    Ch_ref  = librosa.feature.chroma_cqt(y=stem, sr=sr, hop_length=hop)
    Ch_pred = librosa.feature.chroma_cqt(y=pred, sr=sr, hop_length=hop)
    chroma_sim = max(
        _score_at_shift(Ch_ref, Ch_pred, sh)
        for sh in range(-EVAL_MAX_SHIFT_FRAMES, EVAL_MAX_SHIFT_FRAMES + 1)
    )

    # CQT octave-aware
    C_ref, C_pred = _compute_cqt_pair(stem, pred, sr)
    cqt_sim = max(
        _score_at_shift(C_ref, C_pred, sh)
        for sh in range(-EVAL_MAX_SHIFT_FRAMES, EVAL_MAX_SHIFT_FRAMES + 1)
    )

    # Per-pitch-class chroma breakdown
    per_pitch_class = _compute_per_pc(Ch_ref, Ch_pred)

    # Temporal clusters — windows where local CQT sim is significantly below global
    temporal_clusters = _compute_temporal_clusters(C_ref, C_pred, sr, cqt_sim)

    # Spectral precision / recall (requires notes)
    spectral_precision  = None
    spectral_recall     = None
    pseudo_f1           = None
    high_note_precision = None

    if cleaned_notes is not None:
        stem_cqt = np.abs(librosa.cqt(
            stem, sr=sr, hop_length=hop,
            n_bins=EVAL_CQT_N_BINS, bins_per_octave=EVAL_CQT_BINS_PER_OCTAVE,
        ))
        spectral_precision, high_note_precision = _compute_spectral_precision(
            cleaned_notes, stem_cqt, sr, hop
        )
        spectral_recall = _compute_spectral_recall(stem, sr, hop, cleaned_notes)
        if spectral_precision is not None and spectral_recall is not None:
            denom = spectral_precision + spectral_recall
            pseudo_f1 = float(2 * spectral_precision * spectral_recall / denom) if denom > 0 else 0.0

    return {
        "chroma_sim":          round(float(chroma_sim), 4),
        "cqt_sim":             round(float(cqt_sim), 4),
        "spectral_precision":  round(float(spectral_precision), 4) if spectral_precision is not None else None,
        "spectral_recall":     round(float(spectral_recall), 4)    if spectral_recall    is not None else None,
        "pseudo_f1":           round(float(pseudo_f1), 4)          if pseudo_f1          is not None else None,
        "per_pitch_class":     per_pitch_class,
        "high_note_precision": round(float(high_note_precision), 4) if high_note_precision is not None else None,
        "temporal_clusters":   temporal_clusters,
    }


# ── Public: convenience wrappers ──────────────────────────────────────────────

def score_instrument(instrument: str, outputs_dir: str) -> dict:
    """
    Score any instrument from its saved pipeline outputs (legacy wrapper).
    Returns dict with keys: instrument, score, metric, details.
    """
    if instrument not in ("guitar", "bass", "piano", "vocals", "drums"):
        return {"instrument": instrument, "score": None,
                "metric": "unknown", "details": "unsupported instrument"}

    instr_dir = os.path.join(outputs_dir, instrument)
    stem_path = os.path.join(instr_dir, "01_stem.wav")

    if not os.path.isfile(stem_path):
        return {"instrument": instrument, "score": None,
                "metric": "chroma_similarity", "details": "stem not found"}

    if instrument == "drums":
        hits_path = os.path.join(instr_dir, "02_hits.json")
        if not os.path.isfile(hits_path):
            return {"instrument": instrument, "score": None,
                    "metric": "onset_f1", "details": "hits file not found"}
        with open(hits_path) as f:
            hits = json.load(f)
        s = score_drums(stem_path, hits)
        return {"instrument": instrument, "score": round(s, 4),
                "metric": "onset_f1", "details": f"{len(hits)} hits detected"}

    preview_path = os.path.join(instr_dir, "09_preview.wav")
    if not os.path.isfile(preview_path):
        return {"instrument": instrument, "score": None,
                "metric": "chroma_similarity", "details": "preview not found"}

    s = score_transcription(stem_path, preview_path)
    return {"instrument": instrument, "score": round(s, 4),
            "metric": "chroma_similarity", "details": "stem vs synthesized preview"}


def score_pipeline_stages(outputs_dir: str, instrument: str) -> dict:
    """
    Compute spectral_precision at each saved pipeline stage to diagnose where
    accuracy is lost.  Requires the stem WAV and per-stage note JSON files.

    Returns:
    {
        "raw":     {"note_count": int, "spectral_precision": float},
        "cleaned": {"note_count": int, "spectral_precision": float},
        "delta":   float,   # cleaned - raw (negative = cleaning removed real notes)
        "bottleneck": "false_positives" | "false_negatives" | "balanced"
    }
    """
    instr_dir  = os.path.join(outputs_dir, instrument)
    stem_path  = os.path.join(instr_dir, "01_stem.wav")
    raw_path   = os.path.join(instr_dir, "02_raw_notes.json")
    clean_path = os.path.join(instr_dir, "03_cleaned_notes.json")

    if not os.path.isfile(stem_path):
        return {"error": "stem not found"}

    stem, sr = librosa.load(stem_path, sr=22050, mono=True)
    stem_cqt = np.abs(librosa.cqt(
        stem, sr=sr, hop_length=EVAL_CQT_HOP,
        n_bins=EVAL_CQT_N_BINS, bins_per_octave=EVAL_CQT_BINS_PER_OCTAVE,
    ))

    result = {}

    for stage, path, key in [("raw", raw_path, "raw"), ("cleaned", clean_path, "cleaned")]:
        if not os.path.isfile(path):
            result[key] = {"note_count": None, "spectral_precision": None}
            continue
        with open(path) as f:
            notes = json.load(f)
        prec, _ = _compute_spectral_precision(notes, stem_cqt, sr, EVAL_CQT_HOP)
        result[key] = {
            "note_count":         len(notes),
            "spectral_precision": round(float(prec), 4) if prec is not None else None,
        }

    raw_p   = result.get("raw",     {}).get("spectral_precision")
    clean_p = result.get("cleaned", {}).get("spectral_precision")
    if raw_p is not None and clean_p is not None:
        delta             = round(clean_p - raw_p, 4)
        result["delta"]   = delta
        raw_count   = result.get("raw",     {}).get("note_count", 0) or 0
        clean_count = result.get("cleaned", {}).get("note_count", 0) or 0
        kept_ratio  = clean_count / max(raw_count, 1)
        if delta < -0.04:
            result["bottleneck"] = "cleaning_too_aggressive"
        elif kept_ratio < 0.40:
            result["bottleneck"] = "false_negatives"
        elif clean_p < 0.70:
            result["bottleneck"] = "false_positives"
        else:
            result["bottleneck"] = "balanced"

    return result


def write_diagnostics_sidecar(
    outputs_dir: str,
    instrument: str,
    mode: str = "",
    song_name: str = "",
) -> str | None:
    """
    Compute full diagnostics and write a JSON sidecar file:
      <outputs_dir>/<song_name>_diagnostics.json   (or  <instrument>_diagnostics.json)

    Returns the path written, or None on failure.
    """
    instr_dir    = os.path.join(outputs_dir, instrument)
    stem_path    = os.path.join(instr_dir, "01_stem.wav")
    preview_path = os.path.join(instr_dir, "09_preview.wav")
    notes_path   = os.path.join(instr_dir, "03_cleaned_notes.json")

    if not os.path.isfile(stem_path):
        return None

    # Stage-level precision
    stages_info = score_pipeline_stages(outputs_dir, instrument)

    # v2 metrics (temporal clusters, per-pitch-class)
    top_error_pitches: list[str] = []
    temporal_error_windows: list[list[float]] = []
    v2_metrics: dict = {}

    if os.path.isfile(preview_path):
        cleaned_notes = None
        if os.path.isfile(notes_path):
            with open(notes_path) as f:
                cleaned_notes = json.load(f)
        try:
            v2 = score_transcription_v2(stem_path, preview_path, cleaned_notes=cleaned_notes)
            v2_metrics = v2

            pc = v2.get("per_pitch_class", {})
            if pc:
                sorted_pc = sorted(pc.items(), key=lambda x: x[1] if x[1] is not None else 1.0)
                top_error_pitches = [k for k, v in sorted_pc[:4] if v is not None and v < 0.70]

            for cl in v2.get("temporal_clusters", []):
                if cl.get("severity") == "high":
                    temporal_error_windows.append([cl["start"], cl["end"]])

        except Exception:
            pass

    diagnostics = {
        "instrument": instrument,
        "mode":       mode,
        "stages":     stages_info,
        "bottleneck": stages_info.get("bottleneck", "unknown"),
        "top_error_pitches":       top_error_pitches,
        "temporal_error_windows":  temporal_error_windows,
    }
    if v2_metrics:
        diagnostics["v2"] = {
            k: v2_metrics[k]
            for k in ("cqt_sim", "spectral_precision", "spectral_recall", "pseudo_f1")
            if k in v2_metrics
        }

    label    = song_name or instrument
    out_path = os.path.join(outputs_dir, f"{label}_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    return out_path


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_cqt_pair(
    stem: np.ndarray,
    pred: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CQT magnitude spectrograms for a stem/preview pair."""
    C_ref = np.abs(librosa.cqt(
        stem, sr=sr, hop_length=EVAL_CQT_HOP,
        n_bins=EVAL_CQT_N_BINS, bins_per_octave=EVAL_CQT_BINS_PER_OCTAVE,
    ))
    C_pred = np.abs(librosa.cqt(
        pred, sr=sr, hop_length=EVAL_CQT_HOP,
        n_bins=EVAL_CQT_N_BINS, bins_per_octave=EVAL_CQT_BINS_PER_OCTAVE,
    ))
    return C_ref, C_pred


def _score_at_shift(C_ref: np.ndarray, C_pred: np.ndarray, shift: int) -> float:
    """Cosine similarity averaged over active frames at a given time shift."""
    n = min(C_ref.shape[1], C_pred.shape[1])

    if shift >= 0:
        ref  = C_ref[:, shift:shift + n]
        pred = C_pred[:, :n - shift] if shift < n else C_pred[:, :0]
    else:
        s    = -shift
        ref  = C_ref[:, :n - s]
        pred = C_pred[:, s:s + n]

    n = min(ref.shape[1], pred.shape[1])
    if n == 0:
        return 0.0
    ref, pred = ref[:, :n], pred[:, :n]

    ref_norms  = np.linalg.norm(ref,  axis=0)
    pred_norms = np.linalg.norm(pred, axis=0)
    active = (ref_norms > EVAL_ACTIVE_THRESH) | (pred_norms > EVAL_ACTIVE_THRESH)
    if not np.any(active):
        return 0.0

    dot        = np.sum(ref[:, active] * pred[:, active], axis=0)
    norms      = ref_norms[active] * pred_norms[active]
    frame_sim  = np.zeros_like(dot)
    np.divide(dot, norms, out=frame_sim, where=norms > 1e-6)
    return float(np.mean(frame_sim))


def _compute_per_pc(Ch_ref: np.ndarray, Ch_pred: np.ndarray) -> dict:
    """
    Per-pitch-class chroma similarity.
    For each of 12 semitones, compute chroma similarity restricted to frames
    where that pitch class has > EVAL_PER_PC_MIN_REF_ENERGY in the reference.
    """
    n      = min(Ch_ref.shape[1], Ch_pred.shape[1])
    ref    = Ch_ref[:, :n]
    pred   = Ch_pred[:, :n]
    result = {}

    for pc, name in enumerate(_CHROMA_NAMES):
        active = ref[pc, :] > EVAL_PER_PC_MIN_REF_ENERGY
        if not np.any(active):
            result[name] = None
            continue
        r = ref[:, active]
        p = pred[:, active]
        r_norms = np.linalg.norm(r, axis=0)
        p_norms = np.linalg.norm(p, axis=0)
        valid   = (r_norms > 1e-6) & (p_norms > 1e-6)
        if not np.any(valid):
            result[name] = None
            continue
        dot  = np.sum(r[:, valid] * p[:, valid], axis=0)
        norm = r_norms[valid] * p_norms[valid]
        sim  = dot / norm
        result[name] = round(float(np.mean(sim)), 4)

    return result


def _compute_temporal_clusters(
    C_ref: np.ndarray,
    C_pred: np.ndarray,
    sr: int,
    global_cqt_sim: float,
) -> list[dict]:
    """
    Identify time windows where local CQT similarity is significantly below global.
    Returns list of {"start": float, "end": float, "score": float, "severity": str}.
    """
    hop       = EVAL_CQT_HOP
    win_s     = EVAL_TEMPORAL_CLUSTER_WIN_S
    win_frames = max(1, int(win_s * sr / hop))
    n          = min(C_ref.shape[1], C_pred.shape[1])
    clusters   = []

    i = 0
    while i < n:
        j    = min(i + win_frames, n)
        r    = C_ref[:, i:j]
        p    = C_pred[:, i:j]
        r_n  = np.linalg.norm(r, axis=0)
        p_n  = np.linalg.norm(p, axis=0)
        act  = (r_n > EVAL_ACTIVE_THRESH) | (p_n > EVAL_ACTIVE_THRESH)
        if np.any(act):
            dot  = np.sum(r[:, act] * p[:, act], axis=0)
            nm   = r_n[act] * p_n[act]
            fs   = np.zeros_like(dot)
            np.divide(dot, nm, out=fs, where=nm > 1e-6)
            local_sim = float(np.mean(fs))
        else:
            local_sim = global_cqt_sim  # silent window — don't flag

        gap = global_cqt_sim - local_sim
        if gap >= EVAL_TEMPORAL_CLUSTER_THRESHOLD:
            severity = "high" if gap >= 0.25 else ("medium" if gap >= 0.18 else "low")
            clusters.append({
                "start":    round(i * hop / sr, 2),
                "end":      round(j * hop / sr, 2),
                "score":    round(local_sim, 4),
                "severity": severity,
            })
        i += win_frames

    return clusters


def _midi_to_cqt_bin(midi_pitch: int) -> int:
    """Map a MIDI pitch to its CQT bin index (bin 0 = C1 = MIDI 24)."""
    return midi_pitch - _CQT_MIDI_MIN


def _compute_spectral_precision(
    notes: list[dict],
    stem_cqt: np.ndarray,
    sr: int,
    hop: int,
) -> tuple[float | None, float | None]:
    """
    Precision proxy: fraction of notes whose pitch is present in the stem CQT.
    Also returns high_note_precision (MIDI ≥ 69 only).
    Returns (precision, high_note_precision) — both None if notes is empty.
    """
    if not notes:
        return None, None

    hits     = 0
    hi_hits  = 0
    hi_total = 0
    T        = stem_cqt.shape[1]

    for note in notes:
        p_bin   = _midi_to_cqt_bin(int(note["pitch"]))
        if p_bin < 0 or p_bin >= EVAL_CQT_N_BINS:
            hits += 1  # out of CQT range — assume real
            continue
        t_start = int(note["start"] * sr / hop)
        t_end   = max(t_start + 1, int((note["start"] + note["duration"]) * sr / hop))
        t_start = min(t_start, T - 1)
        t_end   = min(t_end, T)

        bin_lo  = max(0, p_bin - EVAL_SPECTRAL_PITCH_TOL_BINS)
        bin_hi  = min(EVAL_CQT_N_BINS, p_bin + EVAL_SPECTRAL_PITCH_TOL_BINS + 1)
        energy  = float(np.mean(stem_cqt[bin_lo:bin_hi, t_start:t_end]))
        hit     = energy >= EVAL_SPECTRAL_HIT_THRESH

        if hit:
            hits += 1
        if int(note["pitch"]) >= 69:
            hi_total += 1
            if hit:
                hi_hits += 1

    precision      = hits / len(notes)
    hi_prec        = (hi_hits / hi_total) if hi_total > 0 else None
    return precision, hi_prec


def _compute_spectral_recall(
    stem: np.ndarray,
    sr: int,
    hop: int,
    notes: list[dict],
) -> float:
    """
    Recall proxy: fraction of librosa-detected stem onsets covered by a detected note.
    """
    onset_times = librosa.onset.onset_detect(
        y=stem, sr=sr, units="time",
        hop_length=hop, delta=EVAL_SPECTRAL_RECALL_ONSET_DELTA,
    )
    if len(onset_times) == 0:
        return 1.0

    note_intervals = [(n["start"] - EVAL_ONSET_TOLERANCE_S,
                       n["start"] + n["duration"] + EVAL_ONSET_TOLERANCE_S)
                      for n in notes]

    covered = 0
    for t in onset_times:
        if any(lo <= t <= hi for lo, hi in note_intervals):
            covered += 1

    return covered / len(onset_times)


def _match_onsets(ref: np.ndarray, pred: np.ndarray, tol: float) -> int:
    """Count true positive onset matches within tolerance (greedy, O(n log n))."""
    ref  = np.sort(ref)
    pred = np.sort(pred)
    matched_pred = np.zeros(len(pred), dtype=bool)
    tp = 0
    j  = 0
    for r in ref:
        while j < len(pred) and pred[j] < r - tol:
            j += 1
        k = j
        while k < len(pred) and pred[k] <= r + tol:
            if not matched_pred[k]:
                matched_pred[k] = True
                tp += 1
                break
            k += 1
    return tp
