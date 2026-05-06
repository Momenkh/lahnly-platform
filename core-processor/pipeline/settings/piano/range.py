"""
Piano — Range Settings
=======================
MIDI and frequency bounds for piano / keyboard.

Standard 88-key piano: A0 (MIDI 21) to C8 (MIDI 108).
In practice the model rarely detects below MIDI 28 or above MIDI 103
with high confidence, but we use the full range so nothing is clipped.
"""

import math

PIANO_MIDI_MIN = 21     # A0 — lowest key on a standard piano
PIANO_MIDI_MAX = 108    # C8 — highest key on a standard piano

PIANO_HZ_MIN = 440.0 * 2 ** ((PIANO_MIDI_MIN - 69) / 12)   # ~27.5 Hz
PIANO_HZ_MAX = 440.0 * 2 ** ((PIANO_MIDI_MAX - 69) / 12)   # ~4186 Hz

# Middle C split: notes below this MIDI pitch are assigned to the left hand
PIANO_LH_RH_SPLIT_MIDI = 60   # C4 — middle C
