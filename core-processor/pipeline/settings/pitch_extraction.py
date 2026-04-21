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

# ── Single-pass thresholds (acoustic and rhythm) ──────────────────────────────
# For types that use one adaptive pass, thresholds are computed from
# stem_confidence so a noisier stem gets stricter filtering:
#   onset_thresh = onset_base − stem_conf × onset_scale
#   frame_thresh = frame_base − stem_conf × frame_scale
#
# At stem_conf = 1.0 (perfect stem): onset = base − scale  (minimum, most notes)
# At stem_conf = 0.0 (raw mix):      onset = base          (maximum, fewest notes)

# "acoustic" — fingerpicked or strummed acoustic guitar
PITCH_ACOUSTIC_ONSET_BASE  = 0.45
PITCH_ACOUSTIC_ONSET_SCALE = 0.15
PITCH_ACOUSTIC_FRAME_BASE  = 0.30
PITCH_ACOUSTIC_FRAME_SCALE = 0.12

# "rhythm" — electric or acoustic rhythm guitar (full chords, clear onsets)
PITCH_RHYTHM_ONSET_BASE    = 0.60
PITCH_RHYTHM_ONSET_SCALE   = 0.20
PITCH_RHYTHM_FRAME_BASE    = 0.40
PITCH_RHYTHM_FRAME_SCALE   = 0.15

# Single-pass fallback thresholds (onset, frame, min_note_ms).
# "lead" uses multi-pass (below) so its single-pass entry is a last-resort fallback.
# None → computed adaptively from stem_confidence using the formulas above.
PITCH_THRESHOLDS = {
    #             onset   frame  min_note_ms
    "lead":     (0.12,   0.06,  40),    # loose fallback — multi-pass normally used
    "acoustic": (None,   None,  40),    # adaptive
    "rhythm":   (None,   None,  50),    # adaptive
}

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
PITCH_MULTI_PASS_CONFIGS = {
    "lead": [
        (0.50, 0.40, 100),   # single ultra-strict pass — clearest notes only
                             # previously 3-pass; merged output was noisier than
                             # using this pass alone.  All surviving notes carry
                             # full confidence weight (no BASE/CONFIRMED penalty).
    ],
    "acoustic": [
        (0.28, 0.18, 50),    # base — clear fingerpicked notes
        (0.42, 0.30, 80),    # strict — confirms strongest notes
    ],
    "rhythm": [
        (0.40, 0.25, 60),    # base — catches all chord tones at typical stem quality
        (0.55, 0.40, 80),    # strict — confirms only the clearest chord tones
    ],                       # strict pass boosts confidence of well-detected notes
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
# BASE is intentionally punitive.  If pass 3 is "almost perfect", pass-1-only
# notes are likely noise.  Raise BASE toward 0.80 if you find too many real
# notes are being cut; lower it toward 0.30 to push output closer to pass 3.
PITCH_CONF_WEIGHT_STRONG    = 1.00   # confirmed by the strictest pass
PITCH_CONF_WEIGHT_CONFIRMED = 0.88   # confirmed by at least one stricter pass
PITCH_CONF_WEIGHT_BASE      = 0.45   # only in the loosest pass — heavy penalty
