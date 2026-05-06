"""
Piano — Pitch Extraction Settings
===================================
Controls basic-pitch thresholds and CREPE for piano / keyboard.

Piano characteristics that shape these settings:
  - Wide frequency range (A0–C8, MIDI 21–108) — basic-pitch covers this well.
  - Can be highly polyphonic (up to 10 voices) — melody mode is more restrictive.
  - Sustain pedal creates long, overlapping notes — lower onset thresholds needed.
  - CREPE is useful for melody mode (single dominant voice); disabled for chord mode.

Two modes:
  piano_melody  — single-voice / melody lines (right-hand runs, solos)
  piano_chord   — full chord voicings (both hands, high polyphony)
"""

# ── Multi-pass extraction ─────────────────────────────────────────────────────
# Each tuple: (onset_threshold, frame_threshold, min_note_ms)
PIANO_MULTI_PASS_CONFIGS = {
    #                             base pass              strict pass
    "piano_melody": [(0.30, 0.22, 50),     (0.50, 0.38,  80)],
    "piano_chord":  [(0.40, 0.30, 60),     (0.58, 0.46, 100)],
}

# Single-pass fallback thresholds (onset, frame, min_note_ms).
# None → computed adaptively from stem_confidence.
PIANO_PITCH_THRESHOLDS = {
    "piano_melody": (None, None, 50),
    "piano_chord":  (None, None, 60),
}

# ── Adaptive single-pass threshold bases ──────────────────────────────────────
PIANO_PITCH_ONSET_BASE  = 0.50
PIANO_PITCH_ONSET_SCALE = 0.15
PIANO_PITCH_FRAME_BASE  = 0.38
PIANO_PITCH_FRAME_SCALE = 0.10

# Maximum time between two notes of the same pitch to be merged as one
PIANO_PITCH_MERGE_PROXIMITY_S = 0.060   # seconds

# ── Confidence weights (multi-pass merge) ─────────────────────────────────────
PIANO_CONF_WEIGHT_STRONG    = 1.00
PIANO_CONF_WEIGHT_CONFIRMED = 0.88
PIANO_CONF_WEIGHT_BASE      = 0.45

# ── pyin fallback ─────────────────────────────────────────────────────────────
PIANO_PYIN_MIN_NOTE_DURATION_S = 0.06

# ── High-note recovery pass ───────────────────────────────────────────────────
PIANO_HIGH_NOTE_RECOVERY_MIDI   = 72      # C5 — start earlier; piano range is wider
PIANO_HIGH_NOTE_RECOVERY_HZ     = 523.25  # Hz equivalent of MIDI 72 (C5)
PIANO_HIGH_NOTE_RECOVERY_ONSET  = 0.35
PIANO_HIGH_NOTE_RECOVERY_FRAME  = 0.28
PIANO_HIGH_NOTE_RECOVERY_MIN_MS = 25      # piano high notes can be very short
PIANO_HIGH_NOTE_RECOVERY_MIN_CONF = 0.25

# ── CREPE overlay ─────────────────────────────────────────────────────────────
# CREPE is only useful for melody mode — chord mode has too many simultaneous
# voices for a monophonic tracker to add value.
# PIANO_CREPE_FMAX must stay safely below 1975 Hz (CREPE model ceiling).
# Piano top note C8 = ~4186 Hz, which is above CREPE's range, but notes above
# CREPE_FMAX are still captured by basic-pitch.
PIANO_CREPE_ENABLED = True
PIANO_CREPE_MODEL   = "tiny"
PIANO_CREPE_FMAX    = 1760.0   # Hz — same safe ceiling as guitar
PIANO_CREPE_CONF_THRESHOLDS = {
    "piano_melody": 0.72,
    "piano_chord":  0.72,   # used only when melody is forced for chord stems
}
PIANO_CREPE_MIN_NOTE_S    = 0.040
PIANO_CREPE_MAX_GAP_S     = 0.060
PIANO_CREPE_PITCH_TOLERANCE = 1   # semitones
PIANO_CREPE_REPET_ENABLED   = False

# ── Sustained note confidence boost ──────────────────────────────────────────
PIANO_SUSTAINED_BOOST_MIN_S  = 0.150
PIANO_SUSTAINED_BOOST_AMOUNT = 0.08
PIANO_SUSTAINED_BOOST_CAP    = 0.95
