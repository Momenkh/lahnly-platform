"""
Evaluation — chroma similarity and onset F1 scoring settings.
"""

# Minimum chroma frame L2-norm below which a frame is considered silent.
# Frames below this threshold are excluded from the chroma similarity calculation
# so that silent sections (intro, outro, gaps) don't dilute the score.
EVAL_ACTIVE_THRESH = 0.10

# Maximum integer-frame shift to search when aligning preview against stem.
# 15 frames × (512/22050 s/frame) ≈ ±350 ms — absorbs systematic timing offsets
# from quantization grid-snapping without allowing arbitrary repositioning.
EVAL_MAX_SHIFT_FRAMES = 15

# Tolerance window for onset F1 scoring (drums).
# Two onsets closer than this (seconds) count as matched.
# 50ms is standard in Music Information Retrieval evaluation.
EVAL_ONSET_TOLERANCE_S = 0.050
