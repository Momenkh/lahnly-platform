"""
Guitar Range
============
MIDI and Hz bounds shared across Stages 2, 3, and 5b.

Standard guitar spans E2 (MIDI 40) to roughly the 21st fret of the high-e string.
These bounds are passed to basic-pitch as hard frequency limits so the model does not
waste probability mass on out-of-range pitches, and are used as post-extraction filters
to discard any detections that slipped through.
"""

# Lowest open string (low-E string, standard tuning).
GUITAR_MIDI_MIN = 40        # E2

# Highest reachable note for acoustic/rhythm guitar.
# Fret 21 on the high-e string (E4 = MIDI 64) is MIDI 85 (C#6).
# The extra 4 semitones give headroom for small bends and upper-fret playing
# without letting in clearly out-of-range harmonics.
GUITAR_MIDI_MAX = 89        # F6  (fret 21 + 4 semitones headroom)

# Same bounds expressed in Hz for basic-pitch's frequency-range arguments.
GUITAR_HZ_MIN   = 82.41     # E2
GUITAR_HZ_MAX   = 1396.91   # F6

# ── Lead guitar ceiling ───────────────────────────────────────────────────────
# Lead players bend notes at the very top of the fretboard.  On a 24-fret guitar
# the highest note is MIDI 88 (E6, fret 24 on high-e); a whole-step bend from
# there reaches MIDI 90, a 2-step bend MIDI 92.  MIDI 96 (C7, 2093 Hz) gives
# comfortable room above that while staying below overtone territory.
# Raise this if you play a guitar with a higher-than-24-fret neck.
GUITAR_MIDI_MAX_LEAD = 96       # C7
GUITAR_HZ_MAX_LEAD   = 2093.0   # C7 in Hz
