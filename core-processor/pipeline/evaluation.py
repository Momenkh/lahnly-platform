import numpy as np
import librosa


def score_transcription(stem_path: str, preview_path: str) -> float:
    """
    Compute chroma cosine similarity between guitar stem and synthesized preview.
    Returns a value in [0, 1]: higher = better pitch agreement.
    """
    hop = 512
    stem, sr = librosa.load(stem_path, sr=22050, mono=True)
    pred, _  = librosa.load(preview_path, sr=22050, mono=True)

    C_ref  = librosa.feature.chroma_cqt(y=stem, sr=sr, hop_length=hop)
    C_pred = librosa.feature.chroma_cqt(y=pred, sr=sr, hop_length=hop)

    n = min(C_ref.shape[1], C_pred.shape[1])
    C_ref, C_pred = C_ref[:, :n], C_pred[:, :n]

    dot       = np.sum(C_ref * C_pred, axis=0)
    norms     = np.linalg.norm(C_ref, axis=0) * np.linalg.norm(C_pred, axis=0)
    frame_sim = np.where(norms > 1e-6, dot / norms, 0.0)
    return float(np.mean(frame_sim))
