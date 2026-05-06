"""
Stage 7 — Chord Detection
==========================
Groups simultaneous notes into named chords.

A "strum window" is computed per beat: notes that start within one strum
window of each other are grouped into a candidate chord.  The candidate is
then matched against a library of chord templates (major, minor, 7th, sus,
power chords, etc.) using a scoring function that rewards matching pitch
classes and penalises extra pitch classes not in the template.

A candidate group must pass three quality gates before it is labelled a chord:
  1. Enough notes     — CHORD_MIN_NOTES simultaneous notes
  2. Enough variety   — CHORD_MIN_UNIQUE_PCS distinct pitch classes
                        (filters octave doublings that are technically one pitch)
  3. Long enough      — CHORD_MIN_DURATION_S combined duration
     (filters short ornamentation that happens to land simultaneously)
"""

# ── Strum window ──────────────────────────────────────────────────────────────
# Notes are strummed one string at a time; even a fast strum takes ~50–100ms.
# Notes starting within the strum window are considered part of the same chord.
#
# The window is a rolling anchor: each new note resets the group reference to
# its own start time, so a slow strum (6 strings across 90ms) stays in one group.
# The cap prevents very slow tempos from creating an absurdly wide grouping window.
CHORD_DEFAULT_STRUM_S     = 0.09   # fallback strum window when BPM is unknown (seconds)
CHORD_STRUM_BEAT_FRACTION = 0.25   # strum window as a fraction of one beat

# ── Chord validity gates ──────────────────────────────────────────────────────
CHORD_MIN_NOTES      = 3      # minimum simultaneous notes to form a chord
CHORD_MIN_UNIQUE_PCS = 2      # minimum distinct pitch classes (filters octave clusters)
CHORD_MIN_DURATION_S = 0.12   # combined group duration floor (seconds)

# ── Template matching penalty ─────────────────────────────────────────────────
# score = (pitch classes matched) − CHORD_EXTRA_PC_PENALTY × (extra pitch classes)
# "Extra" means pitch classes present in the detected notes but absent from the
# chord template.  A penalty of 0.3 means three unmatched extras cancel one match.
# Raise toward 1.0 for stricter matching (fewer chord labels, more "unknown");
# lower toward 0.1 to label chords even with significant chromatic embellishment.
CHORD_EXTRA_PC_PENALTY = 0.3

# ── 2-note power chord detection ─────────────────────────────────────────────
# CHORD_MIN_NOTES = 3 normally gates out 2-note groups, but power chords
# (root + 5th, or root + 4th as an inverted 5th) are a core rhythm-guitar
# voicing and should appear on the chord sheet.
#
# Only these specific intervals qualify as power chords:
#   7 semitones = perfect 5th  (e.g. E2 + B2 → E5)
#   5 semitones = perfect 4th  (e.g. B2 + E3 → E5, inverted)
#
# Seconds (1, 2 semitones) and thirds (3, 4) are more likely to be two
# melody notes that happen to start simultaneously — they stay as solo notes.
CHORD_POWER_INTERVALS = frozenset({5, 7})
