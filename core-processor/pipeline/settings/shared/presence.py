"""
Stem Presence Detection — Settings
====================================
Thresholds used to decide whether a target instrument is actually present
in a Demucs-separated stem, before running expensive pitch extraction.

Three independent signals are combined (majority vote, energy weighted 2x):

  1. energy_ratio_raw  — stem_rms / mix_rms computed at separation time
                         (pre-normalization, saved in stem_meta).
                         Low = stem carries little of the mix → likely absent.

  2. spectral_flatness — mean spectral flatness of the stem WAV.
                         A real instrument has clear harmonic structure (low value).
                         Residual bleed / noise is spectrally flat (high value).

  3. onset_density_s   — detected onsets per second in the stem WAV.
                         Real instruments have clear attack transients.
                         Ghost stems have very few sharp onsets.

Voting rule: each signal votes present (1) or absent (0).
  weighted_score = 2×energy_vote + flatness_vote + onset_vote
  is_present     = weighted_score >= 3  (out of maximum 4)

This means:
  - Energy absent → always absent (0+1+1=2 < 3), regardless of other signals.
  - Energy present + at least one other present → is_present=True.
  - Energy barely present but both other signals absent → is_present=False.
"""

# ── Hard energy floor ─────────────────────────────────────────────────────────
# If energy_ratio_raw < this, instrument is definitely absent — skip all further checks.
# Based on observation: ghost stems are typically < 1.5% of mix energy.
PRESENCE_HARD_ENERGY_FLOOR = 0.015

# ── Analysis window ───────────────────────────────────────────────────────────
# Only analyze a middle slice of the audio (avoid silence at start/end).
PRESENCE_ANALYSIS_MAX_S  = 60.0   # cap at 60s for performance
PRESENCE_ANALYSIS_SKIP_S =  5.0   # skip first/last 5s (intro silence, fadeout)

# ── Per-instrument thresholds ─────────────────────────────────────────────────
# Tuple: (energy_ratio_thresh, spectral_flatness_thresh, onset_density_per_s_thresh)
#   energy:   below → vote absent      (separation-time energy_ratio_raw)
#   flatness: above → vote absent      (high flatness = noisy = ghost stem)
#   onsets:   below → vote absent      (few onsets = no real playing)
#
# Guitar   — present: rms ~10–50% of mix, clear picks/strums, low flatness
# Bass     — present: rms ~8–40% of mix, fewer onsets than guitar
# Piano    — present: rms ~5–35% of mix, sustained notes → use lower onset thresh
# Vocals   — present: rms ~5–35% of mix, vibrato → onset detection is noisier
# Drums    — present: rms ~10–50%, very high onset density (many hits)
PRESENCE_THRESHOLDS = {
    #              energy_ratio   flatness   onsets/s
    "guitar":     (0.040,         0.250,     0.50),
    "bass":       (0.035,         0.300,     0.30),
    "piano":      (0.030,         0.220,     0.20),
    "vocals":     (0.025,         0.280,     0.15),
    "drums":      (0.050,         0.350,     1.00),
}

# ── Cross-stem bleed detection ────────────────────────────────────────────────
# When Demucs can't find an instrument, it bleeds a nearby harmonic instrument
# into that stem instead of leaving it silent.  The most common case is piano
# content appearing in the guitar stem when no guitar is present.
#
# If cosine similarity of the log-mel spectrograms of two stems exceeds this
# threshold, the lower-priority stem is declared absent (bleed from the other).
#
# Pairs are defined asymmetrically: guitar checks against piano (not vice versa)
# because piano is a dedicated htdemucs_6s output and is trusted as the source.
#
# At 0.90: clearly the same content (96% for Easy On Me guitar vs piano).
# Songs with real guitar + piano typically score 0.55–0.75 on this metric.
PRESENCE_CROSS_STEM_SIM_THRESH = 0.90

# instrument -> list of rivals to compare against (checked in order; first hit wins)
PRESENCE_CROSS_STEM_RIVALS = {
    "guitar": ["piano"],   # guitar stem ≈ piano stem → guitar is absent (piano bleed)
    "bass":   [],
    "piano":  [],
    "vocals": [],
    "drums":  [],
}
