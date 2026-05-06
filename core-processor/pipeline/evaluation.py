import numpy as np
import librosa

from pipeline.settings import EVAL_ACTIVE_THRESH, EVAL_MAX_SHIFT_FRAMES, EVAL_ONSET_TOLERANCE_S


def score_transcription(stem_path: str, preview_path: str) -> float:
    """
    Compute chroma cosine similarity between a stem and a synthesized preview.
    Works for any pitched instrument (guitar, bass, piano, vocals).
    Returns a value in [0, 1]: higher = better pitch agreement.

    Improvements over naive frame-by-frame mean:
    1. Active-frame masking — only frames where at least one signal has
       non-trivial chroma energy are included.
    2. Best integer-frame shift — absorbs small systematic timing offsets.
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


def score_drums(stem_path: str, detected_hits: list[dict]) -> float:
    """
    Evaluate drum onset detection quality via F1 score.

    Reference onsets are extracted from the stem using librosa's onset detector
    (the same method the pipeline uses — this measures self-consistency as a
    proxy for detection completeness, not ground-truth accuracy).

    Returns F1 score in [0, 1].
    """
    hop = 512
    y, sr = librosa.load(stem_path, sr=22050, mono=True)
    ref_times = librosa.onset.onset_detect(y=y, sr=sr, units="time",
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


def score_instrument(instrument: str, outputs_dir: str) -> dict:
    """
    Score any instrument from its saved pipeline outputs.

    Returns dict with keys: instrument, score, metric, details.
    """
    import os

    if instrument not in ("guitar", "bass", "piano", "vocals", "drums"):
        return {"instrument": instrument, "score": None,
                "metric": "unknown", "details": "unsupported instrument"}

    instr_dir = os.path.join(outputs_dir, instrument)
    stem_path = os.path.join(instr_dir, "01_stem.wav")

    if not os.path.isfile(stem_path):
        return {"instrument": instrument, "score": None,
                "metric": "chroma_similarity", "details": "stem not found"}

    if instrument == "drums":
        import json
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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _score_at_shift(C_ref: np.ndarray, C_pred: np.ndarray, shift: int) -> float:
    n = min(C_ref.shape[1], C_pred.shape[1])

    if shift >= 0:
        ref  = C_ref[:, shift:shift + n]
        pred = C_pred[:, :n - shift] if shift < n else C_pred[:, :0]
    else:
        s = -shift
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

    dot   = np.sum(ref[:, active] * pred[:, active], axis=0)
    norms = ref_norms[active] * pred_norms[active]
    frame_sim = np.zeros_like(dot)
    np.divide(dot, norms, out=frame_sim, where=norms > 1e-6)
    return float(np.mean(frame_sim))


def _match_onsets(ref: np.ndarray, pred: np.ndarray, tol: float) -> int:
    """Count true positive onset matches within tolerance (greedy, O(n log n))."""
    ref  = np.sort(ref)
    pred = np.sort(pred)
    matched_ref  = np.zeros(len(ref),  dtype=bool)
    matched_pred = np.zeros(len(pred), dtype=bool)
    tp = 0
    j  = 0
    for i, r in enumerate(ref):
        while j < len(pred) and pred[j] < r - tol:
            j += 1
        k = j
        while k < len(pred) and pred[k] <= r + tol:
            if not matched_pred[k]:
                matched_ref[i]  = True
                matched_pred[k] = True
                tp += 1
                break
            k += 1
    return tp
