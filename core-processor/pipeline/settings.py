"""
Pipeline Configuration
======================
All tuneable constants in one place.  Edit here; every stage picks up the change.

Sections:
  GUITAR RANGE          — MIDI and Hz bounds shared across stages
  SEPARATION            — Demucs model cascade and stem quality
  PITCH EXTRACTION      — basic-pitch thresholds per guitar type
  NOTE CLEANING         — confidence, duration, polyphony, merge gaps per guitar type
  QUANTIZATION          — BPM detection and grid-snap settings
  KEY DETECTION         — Krumhansl-Schmuckler and scale-selection bias
  GUITAR MAPPING        — fretboard geometry, hand window, string preference
  CHORD DETECTION       — strum grouping and chord validity rules
  TAB GENERATION        — column layout
  AUDIO SYNTHESIS       — waveform and amplitude settings
  VISUALIZATION         — fretboard diagram appearance
  CHORD SHEET           — chord-box diagram appearance
  MELODY ISOLATION      — low-pitch floor per guitar type
"""

# ── GUITAR RANGE ─────────────────────────────────────────────────────────────
# Shared by pitch extraction, note cleaning, and key-octave correction.

GUITAR_MIDI_MIN = 40          # E2  — lowest open string (low E)
GUITAR_MIDI_MAX = 89          # F6  — fret 21 on high e + small bend headroom

GUITAR_HZ_MIN   = 82.41       # E2 in Hz  (for basic-pitch frequency range)
GUITAR_HZ_MAX   = 1396.91     # F6 in Hz

# Lead guitar ceiling — raised to allow bends at the highest frets.
# A 24-fret guitar (high e fret 24 = MIDI 88) bent up 2 steps reaches MIDI 92.
# MIDI 96 (C7, 2093 Hz) gives comfortable headroom above that.
GUITAR_MIDI_MAX_LEAD = 96     # C7
GUITAR_HZ_MAX_LEAD   = 2093.0 # C7 in Hz


# ── SEPARATION ────────────────────────────────────────────────────────────────
# Stage 1 — Demucs neural source separation.

SEPARATION_SAMPLE_RATE      = 44100   # Hz — Demucs native sample rate
SEPARATION_RMS_SILENCE_THRESH = 0.005 # RMS below this → stem treated as silent
SEPARATION_SHIFTS           = 1       # equivariant shifts (1 = 1 pass, 2 = averaged)
SEPARATION_OVERLAP          = 0.5     # chunk overlap (0.25 default; 0.5 fewer artifacts)
SEPARATION_TARGET_LUFS      = -16.0   # integrated loudness target before separation

# Model cascade — tried in order until a non-silent stem is found
SEPARATION_MODELS = [
    ("htdemucs_6s",       "guitar"),   # best: dedicated guitar stem
    ("htdemucs_ft_other", "other"),    # fine-tuned fallback
    ("htdemucs",          "other"),    # base 4-stem last resort
]

# Base stem quality per model — used to compute stem_confidence
SEPARATION_MODEL_BASE_CONF = {
    ("htdemucs_6s",       "guitar"): 0.85,
    ("htdemucs_ft_other", "other"):  0.60,
    ("htdemucs",          "other"):  0.45,
    ("raw mix",           "mix"):    0.20,
}


# ── PITCH EXTRACTION ──────────────────────────────────────────────────────────
# Stage 2 — basic-pitch polyphonic ML model.
# Thresholds are per guitar type.  Lower = more notes detected (recall ↑, noise ↑).

# basic-pitch internal frame-rate constants (do not change unless upgrading model)
BP_SAMPLE_RATE    = 22050
BP_HOP_LENGTH     = 256
BP_MIDI_OFFSET    = 21       # lowest MIDI note in basic-pitch's 88-note range

# onset_threshold: probability above which a new note onset is accepted
# frame_threshold: probability above which a frame is considered active
# min_note_ms:     shortest accepted note in milliseconds
PITCH_THRESHOLDS = {
    #             onset  frame  min_note_ms
    "lead":     (0.12,  0.06,  40),    # low thresholds — catch bends, hammer-ons
    "acoustic": (None,  None,  40),    # None → computed from stem_confidence (see below)
    "rhythm":   (None,  None,  50),    # None → computed from stem_confidence
}

# For "acoustic" and "rhythm" types, thresholds scale with stem_confidence:
#   onset = onset_base - stem_conf * onset_scale
#   frame = frame_base - stem_conf * frame_scale
PITCH_ACOUSTIC_ONSET_BASE  = 0.45
PITCH_ACOUSTIC_ONSET_SCALE = 0.15
PITCH_ACOUSTIC_FRAME_BASE  = 0.30
PITCH_ACOUSTIC_FRAME_SCALE = 0.12

PITCH_RHYTHM_ONSET_BASE    = 0.60
PITCH_RHYTHM_ONSET_SCALE   = 0.20
PITCH_RHYTHM_FRAME_BASE    = 0.40
PITCH_RHYTHM_FRAME_SCALE   = 0.15

# Minimum note duration used by pyin fallback (seconds)
PITCH_PYIN_MIN_NOTE_DURATION_S = 0.06

# Multi-pass pitch extraction
# Each tuple: (onset_threshold, frame_threshold, min_note_ms)
# Passes run sequentially; results are merged into a single note list.
# Set to None to use a single adaptive pass (rhythm guitar default).
PITCH_MULTI_PASS_CONFIGS = {
    "lead": [
        (0.26, 0.17, 60),   # base: well-supported notes — all passes build on this
        (0.38, 0.28, 80),   # very strict: confirms prominent notes (boosts confidence)
        (0.50, 0.40, 100),  # ultra strict: only the clearest, most prominent notes
    ],
    "acoustic": [
        (0.28, 0.18, 50),   # base: clear fingerpicked notes
        (0.42, 0.30, 80),   # very strict: confirms strongest notes
    ],
    "rhythm": [
        (0.40, 0.25, 60),   # base: catches all chord tones at typical stem quality
        (0.55, 0.40, 80),   # strict: confirms only the clearest chord tones
    ],                      # strict pass boosts confidence of well-detected notes
}

# Two notes of the same pitch are considered the same note if they start within
# this many seconds of each other (covers timing jitter between passes)
PITCH_MERGE_PROXIMITY_S = 0.060   # 60ms — wider window to match stricter min_note_ms

# ── Confidence-boost merge parameters ────────────────────────────────────────
# Passes run from least-strict (base) to most-strict.
# The strictest pass (pass 3 for lead) defines the trusted note set — its notes
# receive STRONG weight. Earlier passes supplement with notes the strict pass
# missed, but at progressively lower weights so cleaning favours pass-3 quality.
#
# Confidence = frame_prob × weight, where weight reflects strictest confirming pass:
#   confirmed by ALL stricter passes → PITCH_CONF_WEIGHT_STRONG   (in pass 3)
#   confirmed by 1 stricter pass    → PITCH_CONF_WEIGHT_CONFIRMED  (in pass 2, not 3)
#   base pass only                  → PITCH_CONF_WEIGHT_BASE       (not in pass 2 or 3)
#
# BASE is intentionally low so pass-1-only notes need strong frame evidence to
# survive cleaning. At lead conf_thresh ≈ 0.096, a BASE note must have
# frame_conf > 0.096/0.45 ≈ 0.21 — much stricter than pass-3 notes (any > 0.096).

PITCH_CONF_WEIGHT_BASE      = 0.45   # base-pass only — heavy penalty, needs strong evidence
PITCH_CONF_WEIGHT_CONFIRMED = 0.88   # in pass 2 but not 3 — moderate penalty
PITCH_CONF_WEIGHT_STRONG    = 1.00   # confirmed by all passes — full trust


# ── NOTE CLEANING ─────────────────────────────────────────────────────────────
# Stage 3 — filters applied to raw pitch-detection output.

# MIDI pitch below which stricter confidence gating is applied (low E3 = bass range)
CLEANING_BASS_CUTOFF_MIDI = 52        # E3

# Confidence gate for bass-range notes when using a dedicated guitar stem
CLEANING_BASS_CONF_DEDICATED_STEM    = 0.50   # htdemucs_6s already removed the bass
# Confidence gate for bass-range notes when using a non-dedicated stem
CLEANING_BASS_CONF_NONDEDICATED_STEM = 0.75

# Fallback merge gap when no BPM is available (seconds)
CLEANING_DEFAULT_MERGE_GAP_S = 0.04

# Maximum merge gap regardless of tempo (seconds)
CLEANING_MAX_MERGE_GAP_S     = 0.08

# Bend / vibrato merge: merge consecutive notes within this many semitones …
CLEANING_BEND_MAX_SEMITONES  = 2
# … if the gap between them is no larger than this (seconds)
CLEANING_BEND_MAX_GAP_S      = 0.12

# Per-guitar-type cleaning parameters:
#   min_duration_s     — fallback minimum note duration (used when BPM is unknown)
#   conf_floor         — base confidence threshold (clean stem, no noise adjustment)
#   conf_stem_scale    — added to conf_floor proportional to (1 - stem_confidence)
#                        e.g. conf_thresh = conf_floor + (1 - stem_conf) * conf_stem_scale
#   max_polyphony      — maximum simultaneously active notes
#   bpm_min_dur_subdiv — when BPM is known: min_duration = beat / subdiv
#                        lead=12 (≈32nd triplet), acoustic=8 (32nd), rhythm=6 (16th triplet)
#                        At 100 BPM these give ~50ms / ~75ms / ~100ms respectively —
#                        matching the fallback values, then scaling naturally with tempo.
#   merge_ratio        — secondary merge condition: also merge two same-pitch fragments
#                        if their gap <= merge_ratio * min(dur1, dur2).
#                        This catches long sustained notes with small detection dips
#                        without touching short repeated notes (whose gap ratio is large).
#                        Example: 50ms gap in a 500ms note → 10% → merges at 0.25.
#                                 50ms gap between two 60ms notes → 83% → stays split.
CLEANING_TYPE_PARAMS = {
    #             min_dur_s  conf_floor  conf_stem_scale  max_polyphony  bpm_subdiv  merge_ratio
    "lead":     (0.040,      0.06,       0.16,            5,             12,          0.25),
    "acoustic": (0.080,      0.18,       0.30,            5,             8,           0.15),
    "rhythm":   (0.100,      0.20,       0.40,            6,             6,           0.10),
}

# BPM-aware min_duration clamp: prevents extreme values at very fast or very slow tempos
CLEANING_BPM_MIN_DUR_CLAMP_MIN_S = 0.030   # never shorter than 30 ms (below basic-pitch resolution)
CLEANING_BPM_MIN_DUR_CLAMP_MAX_S = 0.160   # never longer than 160 ms (would drop real slow notes)

# Local pitch context filter — removes notes whose pitch is far from the local median.
# For each note at time t, the median pitch of all other notes within ±window_s is
# computed. Notes more than max_deviation semitones from that median are removed.
# Requires at least min_context notes in the window to fire (avoids sparse passages).
# Set max_deviation to None to disable for a given type.
#
# Intuition: a solo passage around MIDI 70 should not have a MIDI 94 artifact;
# 24 semitones (2 octaves) from the median catches clear outliers without touching
# legitimate wide-interval jumps.
CLEANING_LOCAL_PITCH_WINDOW_S    = 3.0   # seconds of context on each side
CLEANING_LOCAL_PITCH_MIN_CONTEXT = 6     # minimum notes in window to apply the filter
CLEANING_LOCAL_PITCH_MAX_DEV = {
    #           semitones from local median  (None = disabled)
    "lead":     24,
    "acoustic": 30,
    "rhythm":   None,
}

# Key-confidence feedback filter (applied after Stage 5 key detection)
CLEANING_KEY_CONFIDENCE_CUTOFF = 0.35  # off-key notes below this confidence are removed

# Stage 5b octave error correction (applied after key analysis)
# Only low-confidence notes are candidates; a shift fires when the note is
# ≥THRESHOLD semitones from the local pitch median AND the shifted version
# is significantly closer to that median.
CLEANING_OCTAVE_CONF_GATE       = 0.60   # skip correction for notes at or above this confidence
CLEANING_OCTAVE_WINDOW_S        = 1.5    # neighbor search window in seconds (each side)
CLEANING_OCTAVE_SHIFT_THRESHOLD = 10     # semitones from local median to trigger a shift


# ── QUANTIZATION ──────────────────────────────────────────────────────────────
# Stage 4 — tempo detection and note grid-snap.

QUANTIZATION_SUBDIVISION          = 4     # grid resolution: 4 subdivisions per beat (16th notes)
QUANTIZATION_SNAP_TOLERANCE_FRAC  = 0.35  # snap if within this fraction of a subdivision
QUANTIZATION_DENSE_WINDOW_S       = 60.0  # seconds used for onset-dense BPM detection window
QUANTIZATION_BPM_MIN              = 40.0
QUANTIZATION_BPM_MAX              = 240.0
QUANTIZATION_INSTABILITY_WARN_MS  = 15.0  # warn if inter-beat std exceeds this (ms)


# ── KEY DETECTION ─────────────────────────────────────────────────────────────
# Stage 5 — Krumhansl-Schmuckler key profiles.

# Small score bonus for pentatonic minor during scale selection.
# Pentatonic minor is by far the most common lead guitar scale.
KEY_PENTATONIC_MINOR_BIAS = 0.04


# ── GUITAR MAPPING ────────────────────────────────────────────────────────────
# Stage 6 — assign each note to a (string, fret) position.

MAPPING_MAX_FRET    = 21   # highest fret number (standard guitar)
MAPPING_HAND_SPAN   = 4    # frets reachable without shifting the hand
# Pitches up to this many semitones above MAX_FRET are clamped to fret 21 (bend headroom).
# Lead guitar uses a wider tolerance: bends from high frets on a 24-fret neck can push
# 7+ semitones above fret 21 (e.g. fret 24 on high e + 2-step bend = +7 semitones).
MAPPING_BEND_TOLERANCE      = 2   # acoustic / rhythm
MAPPING_BEND_TOLERANCE_LEAD = 7   # lead — covers 24-fret neck + 2-step bend

# String-stability thresholds: notes longer than LONG prefer staying on the same string;
# notes shorter than SHORT move freely.
MAPPING_LONG_NOTE_S  = 0.30  # seconds
MAPPING_SHORT_NOTE_S = 0.10  # seconds

# String change penalty scores (added to position score; lower total = more preferred)
MAPPING_STRING_CHANGE_PENALTY_LONG   = 2.5   # sustained notes strongly stay on string
MAPPING_STRING_CHANGE_PENALTY_MEDIUM = 1.2   # medium notes prefer same string
MAPPING_STRING_CHANGE_PENALTY_SHORT  = 0.4   # fast notes mildly prefer same string

# Open-string in-key bonus (subtracted from score — negative = more preferred)
MAPPING_OPEN_STRING_IN_KEY_BONUS = 2.5

# Proximity-to-hand-centre weight (score += distance_in_frets * this)
MAPPING_HAND_CENTRE_WEIGHT = 0.2   # kept for reference; no longer used in Viterbi mapping

# Viterbi DP transition costs (replace greedy hand-centre weight)
MAPPING_SHIFT_COST_PER_FRET       = 0.8    # cost per fret the hand must shift beyond the span
MAPPING_CHORD_GAP_S               = 0.06   # max gap (s) between notes treated as simultaneous
MAPPING_SAME_STRING_CHORD_PENALTY = 10.0   # heavy penalty: two chord notes on one string


# ── CHORD DETECTION ───────────────────────────────────────────────────────────
# Stage 7 — group simultaneous notes into named chords.

# Fallback strum window when no tempo info is available (seconds)
CHORD_DEFAULT_STRUM_S = 0.06

# Strum window as a fraction of a beat (capped at CHORD_DEFAULT_STRUM_S)
CHORD_STRUM_BEAT_FRACTION = 0.20

CHORD_MIN_NOTES           = 3      # minimum simultaneous notes to qualify as a chord
CHORD_MIN_UNIQUE_PCS      = 2      # minimum distinct pitch classes (filters octave doublings)
CHORD_MIN_DURATION_S      = 0.12   # chord group must last at least this long (seconds)

# Penalty for extra pitch classes not in the chord template:
#   score = matches - CHORD_EXTRA_PC_PENALTY * extras
CHORD_EXTRA_PC_PENALTY    = 0.3

# 2-note power chord detection.
# Groups of exactly 2 notes whose pitch-class interval is in this set are
# detected as power chords ("E5", "A5", etc.) rather than treated as solo notes.
# P5 = 7 semitones (root + 5th), P4 = 5 semitones (inverted 5th / bass + root).
# Seconds (1, 2) and 3rds (3, 4) are left as solo notes — those are more
# likely to be two melody notes that happened to start simultaneously.
CHORD_POWER_INTERVALS = frozenset({5, 7})


# ── TAB GENERATION ────────────────────────────────────────────────────────────
# Stage 8 — ASCII guitar tab layout.

TAB_COLS_PER_BLOCK          = 32    # 16th-note columns per tab block (32 = 2 bars at 4/4)
TAB_DEFAULT_SECONDS_PER_COL = 0.125 # fallback column width when no tempo info (= 16th at 120 BPM)


# ── AUDIO SYNTHESIS ───────────────────────────────────────────────────────────
# Stage 9 — synthesized preview WAV.

AUDIO_SAMPLE_RATE = 44100   # Hz

# Additive synthesis harmonic amplitudes (relative to fundamental = 1.0)
AUDIO_HARMONIC_2ND = 0.30   # second harmonic (one octave up)
AUDIO_HARMONIC_3RD = 0.15   # third harmonic (octave + fifth)

# Per-note amplitude = AUDIO_AMP_BASE + AUDIO_AMP_CONF_SCALE * confidence
AUDIO_AMP_BASE       = 0.15
AUDIO_AMP_CONF_SCALE = 0.25  # at confidence=1.0 → amp=0.40; at 0.0 → amp=0.15

# Amplitude envelope timings (seconds)
AUDIO_ATTACK_S  = 0.008   # pluck attack
AUDIO_RELEASE_S = 0.060   # tail release


# ── VISUALIZATION ─────────────────────────────────────────────────────────────
# Stage 10 — fretboard diagram (PNG).

VIZ_NOTE_HEIGHT_STRING_UNITS = 0.55   # pill height in string-axis units
VIZ_MIN_NOTE_WIDTH_S         = 0.18   # minimum pill width in seconds (readability floor)
VIZ_ROW_HEIGHT_INCHES        = 2.8    # figure height per section row (inches)
VIZ_DPI                      = 140    # output image DPI

# Adaptive section length: clamp(total_dur / VIZ_SECTION_ROWS, min, max)
VIZ_SECTION_TARGET_ROWS      = 16     # aim for this many rows in the diagram
VIZ_SECTION_MIN_S            = 10.0   # minimum seconds per section row
VIZ_SECTION_MAX_S            = 30.0   # maximum seconds per section row

# Confidence pill colour: dark (#222) at high confidence, light (#aaa) at low
VIZ_PILL_COLOR_HIGH = 0x22   # hex component for high-confidence notes (dark)
VIZ_PILL_COLOR_LOW  = 0xaa   # hex component for low-confidence notes (light)

# Text colour threshold: notes above this confidence get white text
VIZ_TEXT_LIGHT_CONF_THRESHOLD = 0.4
VIZ_FONT_SIZE_WIDE_NOTE       = 6.5   # fret label font size for wider pills
VIZ_FONT_SIZE_NARROW_NOTE     = 5.5   # fret label font size for narrow pills
VIZ_WIDE_NOTE_THRESHOLD_S     = 0.3   # pills wider than this use VIZ_FONT_SIZE_WIDE_NOTE


# ── CHORD SHEET ───────────────────────────────────────────────────────────────
# Stage 11 — chord box diagram (PNG).

CHORD_SHEET_DISPLAY_FRETS      = 5   # fret rows shown per chord diagram
CHORD_SHEET_COLS_PER_ROW       = 6   # chord diagrams per row
CHORD_SHEET_PROGRESSION_PER_LINE = 8 # chord symbols per line in the progression section
CHORD_SHEET_BARRE_MIN_STRINGS  = 4   # min strings at same fret to draw a barre bar
CHORD_SHEET_DPI                = 140


# ── MELODY ISOLATION ─────────────────────────────────────────────────────────
# Applied between Stage 6 and Stage 7.
# Notes below the min_pitch floor go straight to harmony (prevents low-string bleed).

MELODY_MIN_PITCH = {
    "lead":     50,   # D3 — exclude low-E / chord-harmonic bleed from melody track
    "acoustic": 0,    # no floor — fingerpicked pieces often have bass melody notes
    "rhythm":   0,    # no floor — rhythm guitar chords are handled by chord detection
}
