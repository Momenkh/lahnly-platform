"""
Bass — Pitch Extraction Settings
==================================
Controls basic-pitch detection thresholds and CREPE for bass guitar.

Bass characteristics that shape these settings:
  - Fundamental frequencies are lower (41–440 Hz) → basic-pitch has fewer
    ambiguous harmonics to deal with, so lower onset thresholds are viable.
  - Notes are typically sustained longer than guitar → higher min_note_ms.
  - Bass is usually monophonic or duo-phonic (no dense chords).
  - CREPE works well on bass fundamentals; threshold kept moderate because
    bass harmonics can confuse the periodicity estimator.

Three playing styles, each with a 2-pass config (base → strict):
  fingered  — fingers plucking strings, moderate attack, long sustain
  picked    — pick attack, sharper transients, slightly shorter notes
  slap      — thumb/pop, very sharp percussive onset + pop harmonics
"""

# ── Multi-pass extraction ─────────────────────────────────────────────────────
# Each tuple: (onset_threshold, frame_threshold, min_note_ms)
# Pass order: loosest (base) → strictest.
BASS_MULTI_PASS_CONFIGS = {
    #                           base pass              strict pass
    "bass_fingered": [(0.20, 0.14, 80),    (0.40, 0.28, 120)],
    "bass_picked":   [(0.25, 0.18, 60),    (0.45, 0.32, 100)],
    "bass_slap":     [(0.30, 0.22, 50),    (0.50, 0.38,  80)],
}

# Single-pass fallback thresholds (onset, frame, min_note_ms).
# None → computed adaptively from stem_confidence.
BASS_PITCH_THRESHOLDS = {
    "bass_fingered": (None, None, 80),
    "bass_picked":   (None, None, 60),
    "bass_slap":     (None, None, 50),
}

# ── Adaptive single-pass threshold bases ──────────────────────────────────────
# onset_thresh = base − stem_conf × scale
BASS_PITCH_ONSET_BASE  = 0.45
BASS_PITCH_ONSET_SCALE = 0.15
BASS_PITCH_FRAME_BASE  = 0.32
BASS_PITCH_FRAME_SCALE = 0.10

# Maximum time between two notes of the same pitch to be merged as one
BASS_PITCH_MERGE_PROXIMITY_S = 0.080   # seconds (wider than guitar — bass notes sustain longer)

# ── Confidence weights (multi-pass merge) ─────────────────────────────────────
BASS_CONF_WEIGHT_STRONG    = 1.00
BASS_CONF_WEIGHT_CONFIRMED = 0.88
BASS_CONF_WEIGHT_BASE      = 0.65

# ── pyin fallback ─────────────────────────────────────────────────────────────
BASS_PYIN_MIN_NOTE_DURATION_S = 0.08   # bass notes are longer than guitar

# ── CREPE overlay ─────────────────────────────────────────────────────────────
# CREPE is useful for bass — fundamentals are very clear for the CNN.
# BASS_CREPE_FMAX must stay below 1975 Hz (CREPE model ceiling); bass notes
# rarely exceed 440 Hz so 880 Hz gives ample headroom without hitting the limit.
BASS_CREPE_ENABLED       = True
BASS_CREPE_MODEL         = "tiny"
BASS_CREPE_FMAX          = 880.0    # Hz — well within CREPE model range
BASS_CREPE_CONF_THRESHOLDS = {
    "bass_fingered": 0.70,
    "bass_picked":   0.72,
    "bass_slap":     0.65,   # slap produces more noise → lower threshold
}
BASS_CREPE_MIN_NOTE_S    = 0.060   # seconds (bass notes are sustained)
BASS_CREPE_MAX_GAP_S     = 0.100   # wider gap — bass players sometimes leave space
BASS_CREPE_PITCH_TOLERANCE = 1     # semitones
BASS_CREPE_REPET_ENABLED       = False
BASS_CREPE_REPET_AUTO_THRESHOLD = 0.55   # auto-enable REPET-SIM when stem_confidence < this

# ── Sustained note confidence boost ──────────────────────────────────────────
BASS_SUSTAINED_BOOST_MIN_S  = 0.200   # bass notes sustain longer than guitar
BASS_SUSTAINED_BOOST_AMOUNT = 0.08
BASS_SUSTAINED_BOOST_CAP    = 0.95
