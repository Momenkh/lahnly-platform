"""
Stage 3 — Note Cleaning
=======================
Filters applied to the raw pitch-detection output to remove noise, merge
fragmented notes, and limit polyphony to what is physically playable.

Processing order
----------------
1. Confidence gate        — drop notes below a stem-quality-adjusted threshold
2. Bass-range gate        — stricter threshold for MIDI < 52 (low-string bleed)
3. Duration filter        — drop notes shorter than the BPM-aware minimum
4. Merge nearby fragments — join consecutive same-pitch detections that are
                            likely one physical note split by a detection dip
5. Local pitch filter     — remove notes whose pitch is a clear outlier relative
                            to the surrounding passage (catches high-freq artifacts)
5b. Key feedback filter   — drop off-key notes with very low confidence after key
                            detection (Stage 5) completes
6. Polyphony limit        — keep only the N most confident simultaneous notes
"""

# ── Bass-range gating ─────────────────────────────────────────────────────────
# Notes below MIDI 36 (C2) are below every string of a standard guitar in any
# tuning, so they can only be bleed from a bass guitar or very low keyboard.
# We apply a stricter confidence threshold to filter them without touching the
# main confidence gate for legitimate guitar notes.
#
# The guitar's lowest standard string is E2 (MIDI 40); extended-range 7-string
# drops to B1 (MIDI 35).  Setting the cutoff at MIDI 36 leaves every playable
# guitar note unaffected while still catching sub-bass bleed.
#
# The threshold depends on how well the stem was isolated:
#   - Dedicated guitar stem (htdemucs_6s "guitar"): the model already suppressed
#     bass, so a moderate gate (0.50) suffices.
#   - Generic "other" stem: includes keyboards, bass, etc., so we tighten to 0.75.
CLEANING_BASS_CUTOFF_MIDI            = 36     # C2 — below every guitar string
CLEANING_BASS_CONF_DEDICATED_STEM    = 0.50   # for htdemucs_6s guitar stem
CLEANING_BASS_CONF_NONDEDICATED_STEM = 0.75   # for generic "other" stem

# ── Fragment merge ────────────────────────────────────────────────────────────
# basic-pitch sometimes breaks one physical note into two detections separated
# by a very short silence (e.g. at a fret-squeak or slight finger lift).
# We merge consecutive same-pitch fragments if their gap is small.
#
# Two complementary conditions are used (either triggers a merge):
#   1. Absolute gap — merge if gap_s ≤ merge_gap (fast, predictable)
#   2. Ratio gap    — merge if gap_s ≤ merge_ratio × min(dur1, dur2)
#                     A 50ms gap in a 500ms note → 10% → merges (merge_ratio=0.25).
#                     A 50ms gap between two 60ms notes → 83% → stays split.
#                     This handles sustained notes with detection dips without
#                     incorrectly joining genuinely repeated short notes.

# Fallback absolute merge gap when BPM is unavailable (seconds)
CLEANING_DEFAULT_MERGE_GAP_S = 0.04

# BPM-based merge gap: merge_gap = max(FLOOR, min(CEILING, beat_s / DIVISOR)).
# At 120 BPM: beat=0.500s → gap=0.063s (clamped to 0.060s ceiling).
# At 180 BPM: beat=0.333s → gap=0.042s (within [FLOOR, CEILING]).
# At 60  BPM: beat=1.000s → gap=0.125s → clamped to 0.060s ceiling.
CLEANING_MERGE_BEAT_DIVISOR  = 8      # beat / 8 = half a 16th note
CLEANING_MERGE_GAP_FLOOR_S   = 0.020  # never narrower than 20ms
CLEANING_MERGE_GAP_CEILING_S = 0.080  # hard cap — matches original behaviour at slow tempos

# Bend / vibrato merge: notes within a few semitones that overlap or are very
# close together are collapsed into a single note (the stronger one survives).
CLEANING_BEND_MAX_SEMITONES = 2    # semitone tolerance for pitch proximity
CLEANING_BEND_MAX_GAP_S     = 0.12 # maximum gap between the two notes (seconds)

# ── Per-mode cleaning parameters ─────────────────────────────────────────────
# Keyed by "{guitar_type}_{guitar_role}" compound key.
# guitar_type ∈ {acoustic, clean, distorted}
# guitar_role ∈ {lead, rhythm}
#
#   min_dur_s        — fallback minimum note duration when BPM is unknown (seconds).
#   conf_floor       — base confidence threshold on a clean, well-isolated stem.
#   conf_stem_scale  — added to conf_floor as stem quality drops:
#                      threshold = conf_floor + (1 − stem_confidence) × conf_stem_scale
#   max_polyphony    — maximum simultaneously sounding notes.
#                      Lead is monophonic (max 2); rhythm chords use up to 6 strings.
#   bpm_subdiv       — BPM-aware min duration divisor (min_dur = 60 / (bpm × subdiv)).
#   merge_ratio      — ratio threshold for the secondary merge condition.
CLEANING_TYPE_PARAMS = {
    #                       min_dur  conf_floor  stem_scale  max_poly  bpm_subdiv  merge
    "acoustic_lead":    (0.040,   0.15,       0.25,       3,        12,         0.20),
    "acoustic_rhythm":  (0.080,   0.18,       0.30,       5,         8,         0.15),
    "clean_lead":       (0.040,   0.10,       0.16,       3,        12,         0.25),   # 3 allows double-stops and grace-note bends
    "clean_rhythm":     (0.100,   0.17,       0.40,       6,         6,         0.10),
    "distorted_lead":   (0.040,   0.12,       0.20,       3,        12,         0.25),   # tighter conf: HPSS + distortion still noisy
    "distorted_rhythm": (0.100,   0.20,       0.45,       6,         6,         0.10),   # tighter conf: distortion raises harmonic floor
}

# ── BPM-aware duration clamp ──────────────────────────────────────────────────
# Prevents extreme min_dur values at very fast or very slow tempos.
# Below 30ms is below basic-pitch's temporal resolution (~12ms per frame at
# 22050Hz / 256 hop), so shorter thresholds would never filter anything useful.
# At 40 BPM a 16th note is 375ms; capping at 300ms keeps slow ballad notes alive.
CLEANING_BPM_MIN_DUR_CLAMP_MIN_S = 0.030   # never shorter than 30 ms
CLEANING_BPM_MIN_DUR_CLAMP_MAX_S = 0.300   # never longer than 300 ms

# ── Local pitch context filter ────────────────────────────────────────────────
# For each note at time t, compute the median pitch of all notes within
# ±WINDOW_S seconds.  If the note deviates more than MAX_DEV semitones from
# that median AND there are at least MIN_CONTEXT other notes in the window,
# the note is removed.
#
# Why this works: a solo passage centred around MIDI 70 should not contain a
# MIDI 94 artifact.  A 24-semitone deviation (2 octaves) catches obvious
# outliers while leaving legitimate wide-interval jumps and high bends intact.
# The minimum context guard prevents the filter from misfiring on very sparse
# passages where a single high note IS the melody.
#
# Set MAX_DEV to None for a given guitar type to disable the filter entirely.
CLEANING_LOCAL_PITCH_WINDOW_S    = 3.0   # seconds of context on each side
CLEANING_LOCAL_PITCH_MIN_CONTEXT = 6     # notes in window required to apply filter

CLEANING_LOCAL_PITCH_MAX_DEV = {
    #                       semitones from local median  (None = disabled)
    "acoustic_lead":    24,
    "acoustic_rhythm":  30,
    "clean_lead":       24,   # 2 octaves — catches clear artifacts, allows high bends
    "clean_rhythm":     36,   # 3 octaves — only removes truly absurd outliers
    "distorted_lead":   20,   # tighter: distortion produces more octave errors
    "distorted_rhythm": 30,
}

# ── Key-confidence feedback filter ───────────────────────────────────────────
# Applied as step 5b, after Stage 5 (key detection) has produced key_info.
# Off-key notes (notes whose pitch class is not diatonic in the detected key)
# are removed if their confidence is below this cutoff.
# A higher cutoff is more aggressive: more off-key notes get removed.
# Set to 0.0 to disable.
#
# Kept deliberately low (0.15) because:
# - Non-Western music (Arabic maqam, etc.) uses scales that don't map to the
#   KS-detected Western key, so many real notes appear "off-key".
# - Chromatic passing tones are legitimate even in Western music.
# - The confidence gate and polyphony limit already handle most noise.
# 0.35 was too aggressive and dropped valid notes in non-diatonic contexts.
CLEANING_KEY_CONFIDENCE_CUTOFF = 0.15

# ── Stage 5b octave error correction ─────────────────────────────────────────
# basic-pitch occasionally detects a note one octave too high or low (it
# mistakes a strong harmonic for the fundamental, or vice versa).  After the
# note list is cleaned and the key is known, we do a second-pass correction:
#
# For each low-confidence note, find all neighbours within ±WINDOW_S seconds.
# Compute the median pitch of those neighbours.  If the note is ≥THRESHOLD
# semitones from the median AND shifting by ±12 would bring it significantly
# closer to the median, apply the shift.
#
# Only notes below CONF_GATE are candidates — high-confidence notes are trusted.
# The 3-neighbour minimum prevents the filter misfiring in sparse passages.
CLEANING_OCTAVE_CONF_GATE       = 0.50   # notes at or above this confidence are skipped
                                         # lowered from 0.60: the G5 solo note at t=28.7s
                                         # had conf=0.552 and was being shifted to G4 because
                                         # the local median (chord tones) sat ~19 semitones below.
CLEANING_OCTAVE_WINDOW_S        = 1.5    # neighbour search window in seconds (each side)
CLEANING_OCTAVE_WIDE_WINDOW_S   = 4.0    # fallback window for sparse passages (< 3 neighbours in primary window)
CLEANING_OCTAVE_MIN_NEIGHBORS   = 3      # minimum neighbours required before attempting correction
CLEANING_OCTAVE_SHIFT_THRESHOLD = 13     # semitones from local median to trigger shift
                                         # 10 was too low — legitimate octave leaps (12st)
                                         # and major-7th phrases (11st) were being pulled
                                         # toward the passage median incorrectly.
                                         # 13 = minor 9th: only shifts genuine gross errors.

# ── Stem energy gate ─────────────────────────────────────────────────────────
# Removes notes that fall entirely within a silent region of the stem WAV.
# Demucs never produces true silence — residual bleed and model noise persist
# even when the target instrument is not playing.  The pitch extractor picks
# these up as ghost notes that survive confidence and duration filters because
# the residual energy is consistent (not random) enough to score reasonably.
#
# The gate reads 01_guitar_stem.wav, divides it into windows of WINDOW_S
# seconds, computes the RMS of each window, and rejects any note whose entire
# span only covers windows below THRESH.  If any overlapping window is above
# THRESH the note is kept — a note is only dropped when the instrument is
# provably silent for that whole duration.
#
# THRESH should match SEPARATION_RMS_SILENCE_THRESH (0.005).  It is defined
# separately here so it can be tuned for cleaning without affecting separation.
CLEANING_STEM_GATE_ENABLED = True
CLEANING_STEM_GATE_WINDOW  = 0.08    # seconds per RMS window (~93ms at 44100 Hz)
CLEANING_STEM_GATE_THRESH  = 0.005   # RMS below this → window considered silent

# ── High-pitch confidence gate ────────────────────────────────────────────────
# basic-pitch assigns systematically lower confidence to high-frequency pitches
# (the model is less certain when harmonic content is sparse above the fundamental).
# Notes above CLEANING_HIGH_PITCH_FLOOR get a reduced confidence threshold so
# they are not incorrectly dropped at the global confidence gate.
#
# Only applies when the guitar type has a non-zero conf_floor (lead/acoustic).
# The reduction is intentionally small — we still want a meaningful gate; we
# just compensate for the model's known high-pitch bias.
# Two-tier graduated reduction: basic-pitch confidence drops as pitch rises.
# Tier 1 (A4–B5, MIDI 69–83): moderate reduction — model is somewhat uncertain here.
# Tier 2 (C6+, MIDI 84+)    : strong reduction — fundamentals sparse, harmonics weak.
CLEANING_HIGH_PITCH_FLOOR_1     = 69    # A4 — first tier starts here
CLEANING_HIGH_PITCH_REDUCTION_1 = 0.040 # subtract from conf_thresh for MIDI 69–83
CLEANING_HIGH_PITCH_FLOOR_2     = 84    # C6 — second tier (model struggles most)
CLEANING_HIGH_PITCH_REDUCTION_2 = 0.075 # subtract from conf_thresh for MIDI 84+

# High notes are naturally shorter — scale min_dur down to avoid over-filtering.
CLEANING_HIGH_PITCH_MIN_DUR_SCALE = 0.65  # high notes allowed 35% shorter duration

# ── Ghost note isolation filter ───────────────────────────────────────────────
# Notes with no neighbour within ±WINDOW seconds are almost always separation
# bleed artifacts — real guitar lines don't play a single note surrounded by
# silence on both sides.  Skip when note count < MIN_TOTAL (too few notes to
# judge context reliably).
CLEANING_ISOLATION_ENABLED   = True
CLEANING_ISOLATION_WINDOW_S  = 1.000   # ±1s window — 500ms was too narrow for slow-tempo songs
CLEANING_ISOLATION_MIN_TOTAL = 20      # skip filter when note count < this; 10 was too aggressive
