import numpy as np
import librosa


# Minimum chroma frame L2-norm below which a frame is considered silent.
# Demucs residual noise sits well below 0.10; real guitar frames are above 0.15.
_ACTIVE_NORM_THRESH = 0.10

# Maximum integer-frame shift to search when aligning preview against stem.
# 15 frames × (512/22050 s/frame) ≈ ±350 ms — enough to absorb quantization
# drift and slow-tempo rounding without allowing implausible alignment.
_MAX_SHIFT_FRAMES = 15


def score_transcription(stem_path: str, preview_path: str) -> float:
    """
    Compute chroma cosine similarity between guitar stem and synthesized preview.
    Returns a value in [0, 1]: higher = better pitch agreement.

    Two improvements over a naive frame-by-frame mean:

    1. Active-frame masking — only frames where at least one signal has
       non-trivial chroma energy are included.  Mutual-silence frames
       (guitar not playing, nothing synthesized) would contribute 0.0 to
       the mean even though we got them "right"; excluding them gives a
       more accurate picture of transcription quality.

    2. Best integer-frame shift — we evaluate the score at every shift in
       [-MAX_SHIFT, +MAX_SHIFT] frames and return the maximum.  This absorbs
       small systematic timing offsets from quantization rounding without
       requiring full dynamic time warping.
    """
    hop = 512
    stem, sr = librosa.load(stem_path, sr=22050, mono=True)
    pred, _  = librosa.load(preview_path, sr=22050, mono=True)

    C_ref  = librosa.feature.chroma_cqt(y=stem, sr=sr, hop_length=hop)
    C_pred = librosa.feature.chroma_cqt(y=pred, sr=sr, hop_length=hop)

    best = 0.0
    for shift in range(-_MAX_SHIFT_FRAMES, _MAX_SHIFT_FRAMES + 1):
        score = _score_at_shift(C_ref, C_pred, shift)
        if score > best:
            best = score
    return best


def _score_at_shift(C_ref: np.ndarray, C_pred: np.ndarray, shift: int) -> float:
    """Frame-by-frame cosine similarity at a given integer-frame shift."""
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

    # Mask: at least one signal is active in the frame
    active = (ref_norms > _ACTIVE_NORM_THRESH) | (pred_norms > _ACTIVE_NORM_THRESH)
    if not np.any(active):
        return 0.0

    dot   = np.sum(ref[:, active] * pred[:, active], axis=0)
    norms = ref_norms[active] * pred_norms[active]
    frame_sim = np.zeros_like(dot)
    np.divide(dot, norms, out=frame_sim, where=norms > 1e-6)
    return float(np.mean(frame_sim))
