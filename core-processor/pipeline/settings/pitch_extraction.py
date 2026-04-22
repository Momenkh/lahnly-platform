"""
Stage 2 — Pitch Extraction
===========================
Controls the basic-pitch polyphonic ML model: detection thresholds, the
multi-pass strategy, and how confidence is assigned to each detected note.

basic-pitch outputs three probability arrays per audio frame:
  note   — probability that a given MIDI pitch is active in this frame
  onset  — probability that a new note starts in this frame
  contour — fine-grained pitch contour (used internally)

A note is created when both onset and frame probabilities exceed their
respective thresholds for long enough.  Lower thresholds = more notes
detected (higher recall, more noise).  Higher thresholds = fewer, cleaner
notes (lower recall, less noise).
"""

# ── basic-pitch internal constants ────────────────────────────────────────────
# These match the ICASSP 2022 model.  Only change if you upgrade to a model
# that was trained at a different sample rate or hop length.
BP_SAMPLE_RATE  = 22050   # Hz — model's native sample rate
BP_HOP_LENGTH   = 256     # samples per frame → ~86.1 frames/second
BP_MIDI_OFFSET  = 21      # MIDI number of the lowest pitch in the model's output array

# ── Single-pass adaptive threshold bases ──────────────────────────────────────
# For modes that use one adaptive pass, thresholds are computed from stem_confidence:
#   onset_thresh = onset_base − stem_conf × onset_scale
#   frame_thresh = frame_base − stem_conf × frame_scale
#
# At stem_conf = 1.0 (perfect stem): onset = base − scale  (minimum, most notes)
# At stem_conf = 0.0 (raw mix):      onset = base          (maximum, fewest notes)

# acoustic type (clean nylon/steel string)
PITCH_ACOUSTIC_ONSET_BASE  = 0.55
PITCH_ACOUSTIC_ONSET_SCALE = 0.15
PITCH_ACOUSTIC_FRAME_BASE  = 0.40
PITCH_ACOUSTIC_FRAME_SCALE = 0.10

# clean/distorted rhythm — electric chords, clear onsets
PITCH_RHYTHM_ONSET_BASE    = 0.65
PITCH_RHYTHM_ONSET_SCALE   = 0.15
PITCH_RHYTHM_FRAME_BASE    = 0.48
PITCH_RHYTHM_FRAME_SCALE   = 0.10

# Single-pass fallback thresholds (onset, frame, min_note_ms).
# Keyed by "{guitar_type}_{guitar_role}" compound key.
# None → computed adaptively from stem_confidence using the formulas above.
PITCH_THRESHOLDS = {
    #                          onset   frame  min_note_ms
    "acoustic_lead":        (None,   None,   40),   # adaptive (acoustic base)
    "acoustic_rhythm":      (None,   None,   50),   # adaptive (acoustic base)
    "clean_lead":           (0.12,   0.06,   40),   # fixed fallback — multi-pass normally used
    "clean_rhythm":         (None,   None,   50),   # adaptive (rhythm base)
    "distorted_lead":       (0.15,   0.10,   40),   # slightly tighter than clean_lead
    "distorted_rhythm":     (None,   None,   50),   # adaptive (rhythm base)
}

# ── HPSS pre-processing for distorted guitar ──────────────────────────────────
# librosa.effects.hpss margin: higher = more aggressive harmonic/percussive split.
# 3.0 is a good balance — reduces distortion harmonics without removing fundamentals.
PITCH_DISTORTED_HPSS_MARGIN = 3.0

# ── pyin fallback ─────────────────────────────────────────────────────────────
# If basic-pitch is unavailable the pipeline falls back to librosa pyin
# (monophonic only).  Notes shorter than this are discarded.
PITCH_PYIN_MIN_NOTE_DURATION_S = 0.06   # seconds

# ── Multi-pass extraction ─────────────────────────────────────────────────────
# Some guitar types run basic-pitch multiple times with different thresholds.
# The model inference runs ONCE; only the note-extraction post-processing repeats.
#
# Pass order: loosest → strictest.
# Each tuple: (onset_threshold, frame_threshold, min_note_ms)
#
# The strictest pass defines the "trusted" note set (see confidence weights below).
# Earlier passes add notes the strict pass missed, at lower confidence.
#
# Set to None for a single adaptive pass.
# NOTE: melodia_trick is disabled globally (pitch_extraction.py).
# Without it, every string's harmonics pass through at low frame thresholds.
# Guitar fundamentals score ~0.55–0.90 frame confidence; their harmonics score
# ~0.20–0.40.  The base pass frame threshold must be above the harmonic ceiling
# (~0.40) so harmonics are excluded by threshold, not by melodia suppression.
#
# Lead role is intentionally looser: single-voice lines have few harmonics
# and we need to catch bends/vibrato at lower confidence.
# Distorted type gets slightly tighter thresholds: HPSS removes most distortion
# harmonics but some residue remains above the clean-guitar harmonic ceiling.
PITCH_MULTI_PASS_CONFIGS = {
    #                           base pass              strict pass
    "acoustic_lead":    [(0.40, 0.28, 40),    (0.55, 0.42, 80)],
    "acoustic_rhythm":  [(0.45, 0.32, 50),    (0.58, 0.45, 80)],
    "clean_lead":       [(0.25, 0.20, 40),    (0.45, 0.35, 80)],
    "clean_rhythm":     [(0.50, 0.40, 60),    (0.62, 0.52, 80)],
    "distorted_lead":   [(0.30, 0.25, 40),    (0.50, 0.40, 80)],   # tighter than clean_lead
    "distorted_rhythm": [(0.52, 0.43, 60),    (0.65, 0.55, 80)],   # tighter than clean_rhythm
}

# Maximum time between two notes of the same pitch for them to be considered
# the same physical note across passes (accounts for timing jitter between runs).
PITCH_MERGE_PROXIMITY_S = 0.060   # seconds

# ── Multi-pass confidence weights ─────────────────────────────────────────────
# After merging, each note's final confidence = frame_probability × weight.
# The weight reflects which passes detected it, rewarding notes the strict
# pass confirmed and penalising notes only the loosest pass saw.
#
# At lead conf_thresh ≈ 0.096 a note must have frame_prob above:
#   STRONG   : 0.096 / 1.00 = 0.096  — almost anything the model saw clearly
#   CONFIRMED: 0.096 / 0.88 = 0.109  — slightly stronger evidence required
#   BASE     : 0.096 / 0.45 = 0.213  — needs strong frame evidence to survive
#
# BASE is a mild discount, not a near-rejection — the cleaning stage's adaptive
# threshold is the primary quality gate.  Real notes detected in one pass but
# not a stricter one (e.g. a bent note the model is uncertain about) should
# still be reachable.  Raise BASE toward 1.00 for less discrimination between
# passes; lower it toward 0.30 to treat base-only notes as near-noise.
PITCH_CONF_WEIGHT_STRONG    = 1.00   # confirmed by the strictest pass
PITCH_CONF_WEIGHT_CONFIRMED = 0.88   # confirmed by at least one stricter pass
PITCH_CONF_WEIGHT_BASE      = 0.65   # only in the loosest pass — mild discount
