"""
Bass — Note Cleaning Settings
================================
Cleaning parameters for bass guitar notes.

Bass differs from guitar in two key ways:
  1. Notes are longer and fewer — minimum duration is higher.
  2. No "bass bleed gate" needed — we ARE processing the bass stem, so low-
     frequency notes are the target, not bleed to be removed.

Three playing styles:
  fingered  — moderate sustain, low polyphony (usually monophonic)
  picked    — sharper attack, slightly shorter sustain
  slap      — percussive onset, wider confidence range due to pop harmonics
"""

# ── Per-style cleaning parameters ────────────────────────────────────────────
# Keyed by bass_style compound key (e.g. "bass_fingered").
#   min_dur_s        — minimum note duration fallback (BPM unknown)
#   conf_floor       — base confidence threshold on a clean stem
#   conf_stem_scale  — added to conf_floor as stem quality drops
#   max_polyphony    — simultaneous notes allowed (bass is mostly 1-2)
#   bpm_subdiv       — BPM-aware min duration divisor (min_dur = 60/(bpm×subdiv))
#   merge_ratio      — ratio threshold for secondary merge condition
BASS_CLEANING_TYPE_PARAMS = {
    #                          min_dur  conf_floor  stem_scale  max_poly  bpm_subdiv  merge
    "bass_fingered": (0.080,   0.10,       0.18,       2,         6,         0.25),
    "bass_picked":   (0.060,   0.08,       0.16,       2,         8,         0.20),
    "bass_slap":     (0.050,   0.07,       0.14,       2,         8,         0.15),
}

# ── BPM-aware duration clamp ──────────────────────────────────────────────────
BASS_BPM_MIN_DUR_CLAMP_MIN_S = 0.040   # never shorter than 40ms
BASS_BPM_MIN_DUR_CLAMP_MAX_S = 0.400   # never longer than 400ms (bass notes can be very long)

# ── Fragment merge ────────────────────────────────────────────────────────────
BASS_DEFAULT_MERGE_GAP_S     = 0.06
BASS_MERGE_BEAT_DIVISOR      = 8
BASS_MERGE_GAP_FLOOR_S       = 0.030   # bass needs slightly wider floor than guitar
BASS_MERGE_GAP_CEILING_S     = 0.120   # wider ceiling — bass notes sustain and detection dips are longer

# ── Bend / vibrato merge ──────────────────────────────────────────────────────
BASS_BEND_MAX_SEMITONES = 2
BASS_BEND_MAX_GAP_S     = 0.15

# ── Local pitch context filter ────────────────────────────────────────────────
BASS_LOCAL_PITCH_WINDOW_S    = 4.0    # wider window: bass lines have wider intervals
BASS_LOCAL_PITCH_MIN_CONTEXT = 4      # fewer notes needed to apply filter
BASS_LOCAL_PITCH_MAX_DEV = {
    "bass_fingered": 24,
    "bass_picked":   24,
    "bass_slap":     28,   # slap has more octave jumps
}

# ── Key-confidence feedback ───────────────────────────────────────────────────
BASS_KEY_CONFIDENCE_CUTOFF = 0.12   # lower than guitar — bass often plays chromatic walks

# ── Octave correction ─────────────────────────────────────────────────────────
BASS_OCTAVE_CONF_GATE       = 0.50
BASS_OCTAVE_WINDOW_S        = 2.0
BASS_OCTAVE_WIDE_WINDOW_S   = 5.0    # bass lines are sparse — wider fallback window
BASS_OCTAVE_MIN_NEIGHBORS   = 3
BASS_OCTAVE_SHIFT_THRESHOLD = 13

# ── High-pitch confidence & duration compensation ─────────────────────────────
# Upper-register bass (above C3) is less common and basic-pitch is less confident there.
BASS_HIGH_PITCH_FLOOR_1     = 48    # C3 — upper-register bass
BASS_HIGH_PITCH_REDUCTION_1 = 0.030
BASS_HIGH_PITCH_FLOOR_2     = 60    # C4 — very high for bass
BASS_HIGH_PITCH_REDUCTION_2 = 0.050
BASS_HIGH_PITCH_MIN_DUR_SCALE = 0.70

# ── Stem energy gate ──────────────────────────────────────────────────────────
BASS_STEM_GATE_ENABLED = True
BASS_STEM_GATE_WINDOW  = 0.10
BASS_STEM_GATE_THRESH  = 0.005

# ── Ghost note isolation filter ───────────────────────────────────────────────
BASS_ISOLATION_ENABLED   = True
BASS_ISOLATION_WINDOW_S  = 1.000   # bass lines are sparser — wider window
BASS_ISOLATION_MIN_TOTAL = 15      # higher than guitar default — bass lines are naturally sparser
