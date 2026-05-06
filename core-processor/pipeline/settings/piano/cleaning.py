"""
Piano — Note Cleaning Settings
================================
Cleaning parameters for piano notes.

Piano differs from guitar in several ways:
  1. No fretboard — "bleed" filtering is irrelevant.
  2. Sustain pedal causes long overlapping notes — merge gaps are wider.
  3. Chord mode can have up to 10 simultaneous voices (full two-hand chords).
  4. Melody mode is treated like a lead guitar line (low polyphony, wider confidence).

Two modes:
  piano_melody  — single-voice right-hand line, or solo piano
  piano_chord   — full chord voicings, both hands
"""

# ── Per-mode cleaning parameters ─────────────────────────────────────────────
# Keyed by mode key (e.g. "piano_melody").
#   min_dur_s        — minimum note duration fallback (BPM unknown)
#   conf_floor       — base confidence threshold
#   conf_stem_scale  — added to conf_floor as stem quality drops
#   max_polyphony    — simultaneous notes allowed
#   bpm_subdiv       — BPM-aware min duration divisor
#   merge_ratio      — ratio threshold for secondary merge condition
PIANO_CLEANING_TYPE_PARAMS = {
    #                           min_dur  conf_floor  stem_scale  max_poly  bpm_subdiv  merge
    "piano_melody": (0.050,     0.08,       0.16,       3,         8,         0.20),
    "piano_chord":  (0.060,     0.10,       0.18,      10,         6,         0.25),
}

# ── BPM-aware duration clamp ──────────────────────────────────────────────────
PIANO_BPM_MIN_DUR_CLAMP_MIN_S = 0.030   # never shorter than 30ms
PIANO_BPM_MIN_DUR_CLAMP_MAX_S = 0.300   # never longer than 300ms

# ── Fragment merge ────────────────────────────────────────────────────────────
# Wider than guitar — sustain pedal causes gaps in detection
PIANO_DEFAULT_MERGE_GAP_S    = 0.08
PIANO_MERGE_BEAT_DIVISOR     = 8
PIANO_MERGE_GAP_FLOOR_S      = 0.040   # pedal notes need a wider floor
PIANO_MERGE_GAP_CEILING_S    = 0.160   # sustain pedal causes longer detection gaps

# ── Bend detection — OFF for piano ───────────────────────────────────────────
# Pianos don't bend (no sustain pitch modulation); skip _merge_bends entirely.
PIANO_BEND_ENABLED = False

# ── Local pitch context filter ────────────────────────────────────────────────
PIANO_LOCAL_PITCH_WINDOW_S    = 3.0
PIANO_LOCAL_PITCH_MIN_CONTEXT = 5
PIANO_LOCAL_PITCH_MAX_DEV = {
    "piano_melody": 24,   # wide — pianists jump octaves freely
    "piano_chord":  36,   # very wide — chord voicings span multiple octaves
}

# ── Key-confidence feedback ───────────────────────────────────────────────────
PIANO_KEY_CONFIDENCE_CUTOFF = 0.10   # lower than guitar — piano chromatic runs are common

# ── Octave correction ─────────────────────────────────────────────────────────
PIANO_OCTAVE_CONF_GATE       = 0.50
PIANO_OCTAVE_WINDOW_S        = 2.0
PIANO_OCTAVE_WIDE_WINDOW_S   = 4.0
PIANO_OCTAVE_MIN_NEIGHBORS   = 3
PIANO_OCTAVE_SHIFT_THRESHOLD = 13

# ── High-pitch confidence & duration compensation ─────────────────────────────
# Piano high notes (C5+) are shorter and softer — basic-pitch underrates them.
# Tier 1 (C5–B5, MIDI 72–83): moderate conf reduction, shorter min_dur.
# Tier 2 (C6+, MIDI 84+)    : strong conf reduction — hammers are very light here.
PIANO_HIGH_PITCH_FLOOR_1     = 72    # C5
PIANO_HIGH_PITCH_REDUCTION_1 = 0.045
PIANO_HIGH_PITCH_FLOOR_2     = 84    # C6
PIANO_HIGH_PITCH_REDUCTION_2 = 0.080
PIANO_HIGH_PITCH_MIN_DUR_SCALE = 0.60  # high piano notes allowed 40% shorter

# ── Stem energy gate ──────────────────────────────────────────────────────────
PIANO_STEM_GATE_ENABLED = True
PIANO_STEM_GATE_WINDOW  = 0.10
PIANO_STEM_GATE_THRESH  = 0.005

# ── Ghost note isolation filter ───────────────────────────────────────────────
PIANO_ISOLATION_ENABLED   = True
PIANO_ISOLATION_WINDOW_S  = 0.800   # piano chords are denser but can have gaps between phrases
PIANO_ISOLATION_MIN_TOTAL = 15
