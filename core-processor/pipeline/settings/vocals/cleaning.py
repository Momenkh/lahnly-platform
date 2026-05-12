"""
Vocals — Note Cleaning Settings
==================================
Cleaning parameters for vocal notes.

Key differences from guitar/bass:
  - Always monophonic (max_polyphony = 1).
  - Vibrato merging: consecutive notes within 2 semitones and a tiny gap are slides.
  - Wide merge gap to handle vibrato dips and breath-support fluctuations.
  - No "bleed gate" needed — Demucs vocals stem is fairly clean.
"""

# conf_floor: minimum confidence threshold regardless of stem quality.
# Must be high enough to filter basic-pitch noise additions. CREPE notes are
# already pre-filtered at 0.65 during extraction; this floor only matters for
# the basic-pitch secondary pass whose confidence scores are in [0, 1] from
# the frame array. Set at 0.35 so low-energy basic-pitch hallucinations are removed.
VOCALS_CLEANING_TYPE_PARAMS = {
    #                           min_dur  conf_floor  stem_scale  max_poly  bpm_subdiv  merge
    "vocals_lead":    (0.060,   0.22,       0.15,       1,         8,         0.30),   # lowered: 0.35/0.20 was cutting 55% of real notes
    "vocals_harmony": (0.060,   0.20,       0.13,       2,         8,         0.25),
}

# Hard minimum: conf_thresh never drops below this regardless of stem_conf.
VOCALS_CONF_THRESH_MIN = 0.20   # was 0.35 — too high; CREPE already gates at 0.65 independently

# ── BPM-aware duration clamp ──────────────────────────────────────────────────
VOCALS_BPM_MIN_DUR_CLAMP_MIN_S = 0.040
VOCALS_BPM_MIN_DUR_CLAMP_MAX_S = 0.400

# ── Vibrato / portamento merge ────────────────────────────────────────────────
# Consecutive notes within VOCALS_BEND_MAX_SEMITONES and VOCALS_BEND_MAX_GAP_S
# are merged into one (vibrato, portamento, or legato slides).
VOCALS_BEND_MAX_SEMITONES = 2
VOCALS_BEND_MAX_GAP_S     = 0.150   # wider than guitar — vocal transitions are slower

# ── Nearby merge ─────────────────────────────────────────────────────────────
VOCALS_DEFAULT_MERGE_GAP_S   = 0.10
VOCALS_MERGE_BEAT_DIVISOR    = 6     # slightly wider subdivision — vibrato causes detection gaps
VOCALS_MERGE_GAP_FLOOR_S     = 0.050
VOCALS_MERGE_GAP_CEILING_S   = 0.200  # breath pauses within a phrase

# ── Local pitch context filter ────────────────────────────────────────────────
VOCALS_LOCAL_PITCH_WINDOW_S    = 4.0
VOCALS_LOCAL_PITCH_MIN_CONTEXT = 4
VOCALS_LOCAL_PITCH_MAX_DEV = {
    "vocals_lead":    18,   # voice range is narrower than guitar
    "vocals_harmony": 18,
}

# ── Key-confidence feedback ───────────────────────────────────────────────────
VOCALS_KEY_CONFIDENCE_CUTOFF = 0.10   # low — singers often have blue notes / chromatic lines

# ── Octave correction ─────────────────────────────────────────────────────────
VOCALS_OCTAVE_CONF_GATE       = 0.50
VOCALS_OCTAVE_WINDOW_S        = 2.0
VOCALS_OCTAVE_WIDE_WINDOW_S   = 4.0
VOCALS_OCTAVE_MIN_NEIGHBORS   = 3
VOCALS_OCTAVE_SHIFT_THRESHOLD = 13

# ── High-pitch confidence & duration compensation ─────────────────────────────
# High soprano/tenor notes — CREPE handles these better but basic-pitch secondary
# still benefits from a lower confidence gate.
VOCALS_HIGH_PITCH_FLOOR_1     = 72    # C5 — high tenor / soprano range
VOCALS_HIGH_PITCH_REDUCTION_1 = 0.040
VOCALS_HIGH_PITCH_FLOOR_2     = 84    # C6 — very high soprano
VOCALS_HIGH_PITCH_REDUCTION_2 = 0.070
VOCALS_HIGH_PITCH_MIN_DUR_SCALE = 0.70  # short high syllables are real

# ── Stem energy gate ──────────────────────────────────────────────────────────
VOCALS_STEM_GATE_ENABLED = True
VOCALS_STEM_GATE_WINDOW  = 0.10
VOCALS_STEM_GATE_THRESH  = 0.003   # lower than guitar — vocal stems can be quiet

# ── Ghost note isolation filter ───────────────────────────────────────────────
VOCALS_ISOLATION_ENABLED   = True
VOCALS_ISOLATION_WINDOW_S  = 1.000  # vocals can have long inter-note gaps (slow ballads)
VOCALS_ISOLATION_MIN_TOTAL = 12    # vocals can have fewer notes (slow ballads)

# Pitch-aware isolation: vocals stay in ±1 octave of melody
# min_neighbors=1 not 2: monophonic vocals are naturally sparse; requiring 2 close
# neighbors removes real slow-ballad notes (the spectral gates handle ghost notes).
VOCALS_ISOLATION_PITCH_AWARE     = True
VOCALS_ISOLATION_MAX_INTERVAL_ST = 12
VOCALS_ISOLATION_MIN_NEIGHBORS   = 1

# Smart polyphony — vocals are monophonic so this mainly defends against bleed
VOCALS_POLY_ROOT_PROTECTION     = False   # monophonic; no chord root to protect
VOCALS_POLY_HEIGHT_PENALTY_ST   = 12
VOCALS_POLY_HEIGHT_PENALTY_COEF = 0.02

# Fast-BPM duration filter — vocals rarely do extremely fast runs
VOCALS_BPM_FAST_THRESHOLD_BPM = 160
VOCALS_BPM_FAST_SUBDIV        = 12
