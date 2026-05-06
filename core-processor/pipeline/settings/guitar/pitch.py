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

# CREPE uses the same hop length as basic-pitch so their timing grids align during merge.
CREPE_HOP_LENGTH = BP_HOP_LENGTH

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
PITCH_CONF_WEIGHT_BASE      = 0.45   # only in the loosest pass — needs strong frame evidence to survive

# ── CREPE lead-guitar overlay ─────────────────────────────────────────────────
# CREPE (torchcrepe) is a CNN monophonic pitch tracker run as a second layer
# on top of basic-pitch for lead-role stems.  Where CREPE is confident, its
# note takes priority; basic-pitch fills temporal gaps.
#
# CREPE_ENABLED       — master switch; set False to skip CREPE entirely
# CREPE_MODEL         — "tiny" (fast, ~0.5s/60s on CPU) or "full" (more accurate, slower)
# CREPE_FMAX          — frequency ceiling passed to torchcrepe.predict.
#                       Must stay below 1975 Hz — the model's internal grid max.
#                       Above that threshold, periodicity values collapse to ~0
#                       (all frames appear unvoiced).  1760 Hz (A6) covers the
#                       entire practical guitar fretboard while staying safely
#                       inside the model's valid range.  Notes above this limit
#                       are still captured by basic-pitch.
# CREPE_CONF_THRESH   — per-type voiced-frame threshold (periodicity score).
#                       Frames below this are considered polyphonic/noisy.
#                       Distorted guitar scores lower periodicity than clean
#                       (distortion adds inharmonic content), so its threshold
#                       is reduced to compensate.
# CREPE_MIN_NOTE_S    — minimum note duration to keep after frame segmentation (seconds)
# CREPE_MAX_GAP_S     — gap between same-pitch CREPE frames that still counts as one note
# CREPE_REPET_ENABLED — run librosa REPET-SIM foreground extraction before CREPE
#                       to suppress repeating chord background (experimental, off by default)
CREPE_ENABLED       = True
CREPE_MODEL         = "tiny"
CREPE_FMAX          = 1760.0   # Hz — safe ceiling; do NOT raise above 1975
CREPE_CONF_THRESHOLDS = {
    "acoustic":  0.72,
    "clean":     0.75,
    "distorted": 0.65,   # lower: distortion reduces periodicity scores
}
CREPE_MIN_NOTE_S      = 0.040
CREPE_MAX_GAP_S       = 0.060
CREPE_PITCH_TOLERANCE = 1     # semitones — bp note pitch must be within this many semitones
                               # of the CREPE pitch to be considered a match.  1 semitone
                               # captures bends and vibrato where basic-pitch and CREPE
                               # disagree on the exact MIDI bin.
CREPE_REPET_ENABLED       = False
CREPE_REPET_AUTO_THRESHOLD = 0.55   # stem_confidence below this → auto-enable REPET-SIM for lead role

# ── Octave error correction ───────────────────────────────────────────────────
# A detected note is a candidate for octave correction if its distance to all
# neighbours within ±1 second exceeds this threshold (semitones).
# 5 semitones means "the note is a tritone or more from everything around it"
# — very likely an octave mistracking error rather than a genuine interval.
PITCH_OCTAVE_ERROR_THRESHOLD_ST = 5

# ── High-note recovery pass ───────────────────────────────────────────────────
# A third basic-pitch post-processing pass restricted to MIDI 69+ (A4 and above).
# Runs with very low thresholds on the already-computed model_output (no extra
# inference cost) to recover high notes the main 2-pass merge missed.
# Recovered notes enter at PITCH_CONF_WEIGHT_BASE and are further filtered by
# the graduated confidence gate in cleaning.
PITCH_HIGH_NOTE_RECOVERY_MIDI   = 69      # A4 — recovery covers MIDI 69 and above
PITCH_HIGH_NOTE_RECOVERY_HZ     = 440.0   # Hz equivalent of MIDI 69 (A4)
PITCH_HIGH_NOTE_RECOVERY_ONSET  = 0.35    # moderate — avoids flooding on piano bleed
PITCH_HIGH_NOTE_RECOVERY_FRAME  = 0.28
PITCH_HIGH_NOTE_RECOVERY_MIN_MS = 30      # short — high notes can be brief
PITCH_HIGH_NOTE_RECOVERY_MIN_CONF = 0.25  # frame confidence gate for recovered notes

# ── Sustained note confidence boost ──────────────────────────────────────────
# Notes that survived the multi-pass merge with a long duration have proven
# stability — CREPE/basic-pitch tend to underrate them.  Boost their confidence
# by a small fixed amount, capped to avoid over-saturation.
PITCH_SUSTAINED_BOOST_MIN_S  = 0.150   # minimum duration to qualify (seconds)
PITCH_SUSTAINED_BOOST_AMOUNT = 0.08    # added to confidence
PITCH_SUSTAINED_BOOST_CAP    = 0.95    # ceiling after boost
