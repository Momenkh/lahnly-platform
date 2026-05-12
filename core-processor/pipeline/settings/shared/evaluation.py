"""
Evaluation — chroma similarity, CQT similarity, spectral precision/recall, and onset F1 settings.
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

# ── CQT-based octave-aware similarity (score_transcription_v2) ────────────────
# CQT preserves octave identity — C4 and C5 produce energy in different bins,
# so octave errors are penalised unlike chroma which folds all octaves into 12 bins.
EVAL_CQT_N_BINS          = 84    # 7 octaves: C1 (MIDI 24) → B7 (MIDI 107)
EVAL_CQT_BINS_PER_OCTAVE = 12
EVAL_CQT_HOP             = 512   # same as chroma hop for consistent timing

# ── Spectral presence / precision proxy ──────────────────────────────────────
# For each detected note, verify the stem's CQT has energy at that pitch + time.
# Notes where the stem energy is below this are likely ghost notes.
EVAL_SPECTRAL_HIT_THRESH         = 0.08   # CQT magnitude floor — below = pitch absent in stem
EVAL_SPECTRAL_PITCH_TOL_BINS     = 2      # ±2 CQT bins ≈ ±1 semitone matching tolerance
EVAL_SPECTRAL_RECALL_ONSET_DELTA = 0.05   # librosa onset_detect delta for stem reference onsets

# ── Temporal cluster analysis ─────────────────────────────────────────────────
# Flag 2-second time windows where local CQT similarity is significantly below
# the global average — these windows have the most transcription errors.
EVAL_TEMPORAL_CLUSTER_WIN_S      = 2.0    # window size in seconds
EVAL_TEMPORAL_CLUSTER_THRESHOLD  = 0.15   # flag window if local_score < global - this value

# ── Per-pitch-class breakdown ─────────────────────────────────────────────────
# Minimum chroma energy in the reference for a frame to count toward
# a particular pitch class's per-PC score.
EVAL_PER_PC_MIN_REF_ENERGY = 0.15
