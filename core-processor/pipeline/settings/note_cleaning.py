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
# Notes below E3 on guitar are almost always bleed from bass guitar or low
# open-string harmonics.  We apply a second, stricter confidence threshold to
# weed them out without touching the main confidence gate.
#
# The threshold depends on how well the stem was isolated:
#   - Dedicated guitar stem (htdemucs_6s "guitar"): the model already suppressed
#     bass, so a moderate gate (0.50) suffices.
#   - Generic "other" stem: includes keyboards, bass, etc., so we tighten to 0.75.
CLEANING_BASS_CUTOFF_MIDI            = 52     # E3 — below this = bass-range
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

# BPM-based merge gap: merge_gap = beat / CLEANING_MERGE_BEAT_DIVISOR,
# capped at CLEANING_MAX_MERGE_GAP_S.
# At 120 BPM: beat=0.500s → gap=0.0625s (< 80ms cap → used as-is).
# At 200 BPM: beat=0.300s → gap=0.038s.
CLEANING_MAX_MERGE_GAP_S = 0.08

# Bend / vibrato merge: notes within a few semitones that overlap or are very
# close together are collapsed into a single note (the stronger one survives).
CLEANING_BEND_MAX_SEMITONES = 2    # semitone tolerance for pitch proximity
CLEANING_BEND_MAX_GAP_S     = 0.12 # maximum gap between the two notes (seconds)

# ── Per-type cleaning parameters ─────────────────────────────────────────────
# Each guitar type has different playing characteristics and therefore different
# cleaning aggressiveness.  Tuple fields:
#
#   min_dur_s        — fallback minimum note duration when BPM is unknown (seconds).
#                      Also used as the lower bound of the BPM-derived min_dur.
#
#   conf_floor       — base confidence threshold on a clean, well-isolated stem.
#                      confidence = frame_probability × multi-pass weight.
#                      Notes below this threshold are dropped.
#
#   conf_stem_scale  — added to conf_floor as stem quality drops:
#                      threshold = conf_floor + (1 − stem_confidence) × conf_stem_scale
#                      A noisy stem (stem_conf=0.2) raises the bar more than a clean one.
#
#   max_polyphony    — maximum simultaneously sounding notes.  Lead guitar rarely
#                      plays more than 2 notes at once; rhythm chords can use all 6 strings.
#
#   bpm_subdiv       — BPM-aware min duration divisor:
#                        min_dur = 60.0 / (bpm × bpm_subdiv)
#                      lead=12  → ≈32nd triplet at 100 BPM ≈ 50ms
#                      acoustic=8 → ≈32nd note at 100 BPM ≈ 75ms
#                      rhythm=6   → ≈16th triplet at 100 BPM ≈ 100ms
#                      Clamped to [CLAMP_MIN, CLAMP_MAX] (see below).
#
#   merge_ratio      — ratio threshold for the secondary merge condition (see above).
#                      Higher = more aggressive merging of long sustained notes.
CLEANING_TYPE_PARAMS = {
    #             min_dur_s  conf_floor  conf_stem_scale  max_polyphony  bpm_subdiv  merge_ratio
    "lead":     (0.040,      0.06,       0.16,            5,             12,          0.25),
    "acoustic": (0.080,      0.18,       0.30,            5,             8,           0.15),
    "rhythm":   (0.100,      0.20,       0.40,            6,             6,           0.10),
}

# ── BPM-aware duration clamp ──────────────────────────────────────────────────
# Prevents extreme min_dur values at very fast or very slow tempos.
# Below 30ms is below basic-pitch's temporal resolution (~12ms per frame at
# 22050Hz / 256 hop), so shorter thresholds would never filter anything useful.
# Above 160ms would discard real slow notes in a sparse ballad passage.
CLEANING_BPM_MIN_DUR_CLAMP_MIN_S = 0.030   # never shorter than 30 ms
CLEANING_BPM_MIN_DUR_CLAMP_MAX_S = 0.160   # never longer than 160 ms

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
    #            semitones from local median  (None = disabled)
    "lead":      24,    # 2 octaves — catches clear artifacts, allows high bends
    "acoustic":  30,    # 2.5 octaves — fingerpicked pieces can have wide ranges
    "rhythm":    None,  # disabled — chords deliberately span wide intervals
}

# ── Key-confidence feedback filter ───────────────────────────────────────────
# Applied as step 5b, after Stage 5 (key detection) has produced key_info.
# Off-key notes (notes whose pitch class is not diatonic in the detected key)
# are removed if their confidence is below this cutoff.
# A higher cutoff is more aggressive: more off-key notes get removed.
# Set to 0.0 to disable.
CLEANING_KEY_CONFIDENCE_CUTOFF = 0.35

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
CLEANING_OCTAVE_SHIFT_THRESHOLD = 13     # semitones from local median to trigger shift
                                         # 10 was too low — legitimate octave leaps (12st)
                                         # and major-7th phrases (11st) were being pulled
                                         # toward the passage median incorrectly.
                                         # 13 = minor 9th: only shifts genuine gross errors.

# ── High-pitch confidence gate ────────────────────────────────────────────────
# basic-pitch assigns systematically lower confidence to high-frequency pitches
# (the model is less certain when harmonic content is sparse above the fundamental).
# Notes above CLEANING_HIGH_PITCH_FLOOR get a reduced confidence threshold so
# they are not incorrectly dropped at the global confidence gate.
#
# Only applies when the guitar type has a non-zero conf_floor (lead/acoustic).
# The reduction is intentionally small — we still want a meaningful gate; we
# just compensate for the model's known high-pitch bias.
CLEANING_HIGH_PITCH_FLOOR          = 76   # MIDI 76 = E5 (~3rd fret on high-e string)
CLEANING_HIGH_PITCH_CONF_REDUCTION = 0.030  # subtract from conf_thresh for notes >= floor
