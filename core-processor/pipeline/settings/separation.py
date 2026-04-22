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
SEPARATION_MODEL_BASE_CONF = {
    ("htdemucs_6s",       "guitar"): 0.85,   # dedicated stem — high prior
    ("htdemucs_ft_other", "other"):  0.60,   # generic "other" — moderate prior
    ("htdemucs",          "other"):  0.45,   # base model — lower prior
    ("raw mix",           "mix"):    0.20,   # fallback: no separation at all
}

# ── Raw mix blending ───────────────────────────────────────────────────────────
# When True, a small portion of the original mix is added back into the stem to
# recover transients and high-frequency brightness that Demucs strips from
# low-confidence stems.
#
# Formula:
#   raw_weight = SEPARATION_MAX_RAW_BLEND × (1 − confidence)
#   output     = confidence × stem + raw_weight × raw_mix
#
# Example at confidence=0.70: 70% stem + 9% raw  (not 30% — intentionally asymmetric)
# Example at confidence=0.90: 90% stem + 3% raw
# Example at confidence=0.40: 40% stem + 18% raw
#
# The raw contribution is a fraction of a fraction so it stays small even when
# confidence is low. This also limits bleed during silent sections of the stem
# (no guitar playing) — only raw_weight of the full mix leaks through instead of
# the full (1 − confidence) that a naive linear blend would produce.
#
# SEPARATION_MAX_RAW_BLEND caps how much raw can ever contribute (at confidence=0).
# Raise it if the stem sounds too dry; lower it if you hear too much bleed.
#
# SEPARATION_BLEND_GATE_WINDOW controls the granularity of the silence gate.
# The stem is split into windows of this size; windows whose RMS is below
# SEPARATION_RMS_SILENCE_THRESH get a gate value of 0 (no raw bleed).
# ~93ms at 44100 Hz is fine-grained enough to catch note attacks while
# being coarse enough to avoid flickering on sustained notes.
SEPARATION_MIX_WITH_RAW      = False
SEPARATION_MAX_RAW_BLEND     = 0.30
SEPARATION_BLEND_GATE_WINDOW = 4096   # samples (~93ms at 44100 Hz)
