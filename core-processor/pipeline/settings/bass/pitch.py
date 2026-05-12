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
    # Bass is monophonic — single pass at a threshold above the harmonic floor.
    # Old pass1 (frame≤0.22) sat below bass's overtone ceiling and flooded raw
    # notes with 2nd/3rd harmonics, requiring heavy cleaning to undo.
    "bass_fingered": [(0.32, 0.22, 80)],    # single pass — soft finger-plucks need moderate gate
    "bass_picked":   [(0.36, 0.26, 60)],    # single pass — picked notes have cleaner onsets
    "bass_slap":     [(0.40, 0.30, 50)],    # single pass — slap transients are strong; tight threshold ok
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

# ── High-note recovery pass ───────────────────────────────────────────────────
# Upper-register bass (C3+, MIDI 48+) is less common so basic-pitch underrates it.
# Runs with adaptive thresholds on the already-computed model_output (no extra inference).
BASS_HIGH_NOTE_RECOVERY_MIDI   = 48      # C3 — start of upper-register bass
BASS_HIGH_NOTE_RECOVERY_HZ     = 130.81  # Hz equivalent of MIDI 48 (C3)
BASS_HIGH_NOTE_RECOVERY_ONSET  = 0.30    # base onset threshold at MIDI 48
BASS_HIGH_NOTE_RECOVERY_FRAME  = 0.24    # base frame threshold at MIDI 48
BASS_HIGH_NOTE_RECOVERY_MIN_MS = 50      # bass notes are longer
BASS_HIGH_NOTE_RECOVERY_MIN_CONF = 0.20  # global frame confidence floor

# Adaptive scaling: decays from base (at MIDI 48) to floor (at MIDI 60)
BASS_HIGH_NOTE_RECOVERY_ONSET_FLOOR = 0.12
BASS_HIGH_NOTE_RECOVERY_FRAME_FLOOR = 0.08
BASS_HIGH_NOTE_RECOVERY_PITCH_ZERO  = 48   # C3 — where scaling starts
BASS_HIGH_NOTE_RECOVERY_PITCH_FULL  = 60   # C4 — floor reached here
BASS_HIGH_NOTE_RECOVERY_ONSET_GATE  = 0.18  # min onset spike to confirm recovery
