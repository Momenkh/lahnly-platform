"""
Bass Guitar Range
=================
MIDI and Hz bounds for 4-string bass guitar in standard tuning (EADG).

Standard 4-string bass tuning (low to high):
  String 4 (low E): E1  = MIDI 28   41.2 Hz
  String 3 (A):     A1  = MIDI 33   55.0 Hz
  String 2 (D):     D2  = MIDI 38   73.4 Hz
  String 1 (G):     G2  = MIDI 43   98.0 Hz

Highest reachable note on a 24-fret bass:
  String 1 (G), fret 24 = G4 = MIDI 67   392 Hz

These bounds are passed to basic-pitch as hard frequency limits and used
as post-extraction filters in the bass pitch extraction stage.
"""

# Lowest open string: E1 (string 4, standard tuning)
BASS_MIDI_MIN = 28       # E1

# Highest reachable note: fret 24 on G string (G4)
# +2 semitones headroom for slight bends at the top of the neck
BASS_MIDI_MAX = 69       # A4 — fret 24 on G string (67) + 2st headroom

# Same bounds in Hz for basic-pitch frequency-range arguments
BASS_HZ_MIN = 41.2       # E1
BASS_HZ_MAX = 440.0      # A4 with headroom
