"""
Stage 1 — Instrument Separation
================================
Controls Demucs source-separation: which models to try, how to run them,
and how to score the quality of the resulting stem.

Demucs splits a full mix into instrument stems using a neural network.
We try a cascade of models from best (dedicated guitar stem) to worst (generic
"other" stem) and stop as soon as the stem isn't silent.
"""

# Sample rate Demucs was trained at.  Audio is resampled to this before separation
# and the output stem is saved at this rate.  Do not change unless you know the
# model you are using was trained at a different rate.
SEPARATION_SAMPLE_RATE = 44100   # Hz

# RMS energy below which a stem is considered silent and the next model is tried.
# If your guitar part is very quiet in the mix, raise this slightly so a very
# faint stem still counts as usable.  If you get empty stems on sparse parts,
# lower it.
SEPARATION_RMS_SILENCE_THRESH = 0.005

# Number of equivariant shift passes Demucs averages over.
# 1 = single pass (fast).  2 = two slightly shifted passes averaged together
# (slightly cleaner, roughly doubles separation time).
SEPARATION_SHIFTS = 1

# Overlap between consecutive audio chunks fed to the model (0 – 1).
# Higher overlap = fewer stitching artifacts at chunk boundaries, but slower.
# 0.25 is Demucs default; 0.5 is a good trade-off for guitar recordings with
# fast transients.
SEPARATION_OVERLAP = 0.5

# Loudness normalisation target applied to the full mix before separation.
# Demucs misallocates energy when the input is very loud or very quiet.
# −16 LUFS is a good broadcast level that keeps the model in its training range.
SEPARATION_TARGET_LUFS = -16.0

# ── Model cascade ─────────────────────────────────────────────────────────────
# Tried in order; the first model whose stem is non-silent is used.
# htdemucs_6s has a dedicated "guitar" stem and is almost always the best choice.
# The fallback models extract an "other" stem (everything not drums/bass/vocals)
# which includes guitar but also keyboards, synths, etc.
SEPARATION_MODELS = [
    ("htdemucs_6s",       "guitar"),   # 6-stem model — best for guitar isolation
    ("htdemucs_ft_other", "other"),    # fine-tuned 4-stem, fallback
    ("htdemucs",          "other"),    # base 4-stem, last resort
]

# ── Stem confidence scoring ────────────────────────────────────────────────────
# stem_confidence ∈ [0, 1] is a quality estimate saved to 01_stem_meta.json and
# used by later stages to tighten or relax their thresholds.
#
# Formula: stem_confidence = base_conf * 0.7 + energy_ratio * 0.3
#   base_conf    — model-level prior (below); 6s/guitar is most reliable
#   energy_ratio — how much energy survived separation relative to the input
#
# The blended output is: alpha * stem + (1 − alpha) * original_mix
# where alpha = stem_confidence.  A low-confidence stem blends in the original
# to recover transients and brightness that the model stripped out.
SEPARATION_MODEL_BASE_CONF = {
    ("htdemucs_6s",       "guitar"): 0.85,   # dedicated stem — high prior
    ("htdemucs_ft_other", "other"):  0.60,   # generic "other" — moderate prior
    ("htdemucs",          "other"):  0.45,   # base model — lower prior
    ("raw mix",           "mix"):    0.20,   # fallback: no separation at all
}
