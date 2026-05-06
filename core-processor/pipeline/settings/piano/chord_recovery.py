"""
Piano — Chord-Guided Recovery Settings
========================================
After Stage 3 cleaning, a chromagram analysis of the stem is used to detect
which chords are harmonically active in each time window.  Windows that have
strong chroma energy but too few transcribed notes trigger a recovery pass that
pulls matching notes back from the raw (pre-cleaning) output.

This catches notes that were legitimately removed by the confidence/isolation
filters but are harmonically required (e.g. the V-chord bar in a I-V-vi-IV
progression that the confidence filter deemed "too sparse").

Key parameters
--------------
WINDOW_S           — width of each analysis window.  BPM-aware: computed as
                     one beat when BPM is known, else this fallback.
CHROMA_TOP_N       — number of top pitch-classes in the chromagram to treat as
                     "chord tones".  4 covers most 7th-chord voicings.
CHROMA_MIN_ENERGY  — minimum mean chroma magnitude for the window to be
                     considered harmonically active (avoids noise from silence).
MAX_NOTES_TRIGGER  — skip recovery if the window already has this many or more
                     cleaned notes (already well covered).
MAX_PER_WINDOW     — maximum notes added per window (prevents flooding).
MIN_RAW_CONF       — minimum confidence a raw note must have to be a candidate.
"""

PIANO_CHORD_RECOVERY_ENABLED       = True
PIANO_CHORD_RECOVERY_WINDOW_S      = 0.50   # fallback when BPM unknown (≈1 beat @120BPM)
PIANO_CHORD_RECOVERY_CHROMA_TOP_N  = 4      # top 4 pitch-classes = chord tones
PIANO_CHORD_RECOVERY_CHROMA_MIN_ENERGY = 0.20  # below this → treat window as silent
PIANO_CHORD_RECOVERY_MAX_NOTES_TRIGGER = 2  # recover only if <2 notes in window
PIANO_CHORD_RECOVERY_MAX_PER_WINDOW    = 3  # add at most 3 notes per window
PIANO_CHORD_RECOVERY_MIN_RAW_CONF      = 0.12  # raw candidate minimum confidence
