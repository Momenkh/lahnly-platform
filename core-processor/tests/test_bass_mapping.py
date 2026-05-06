"""
Tests for pipeline/instruments/bass/mapping.py

Covers:
  - Every open string maps to fret 0
  - Fret BASS_MAX_FRET is reachable on every string
  - A pitch above the max fret on all strings returns no positions
  - A pitch below the lowest open string returns no positions
  - A mid-range pitch has positions on multiple strings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from pipeline.instruments.bass.mapping import (
    _all_positions_bass,
    BASS_TUNING,
    BASS_MAX_FRET,
)


class TestAllPositionsBass(unittest.TestCase):

    def test_open_strings(self):
        """Each open string pitch should map to fret 0 on that string."""
        for string_num, open_pitch in BASS_TUNING.items():
            positions = _all_positions_bass(open_pitch)
            self.assertIn((string_num, 0), positions,
                          f"String {string_num} open pitch {open_pitch} should map to fret 0")

    def test_max_fret_reachable(self):
        """Pitch at exactly BASS_MAX_FRET on each string should be included."""
        for string_num, open_pitch in BASS_TUNING.items():
            top_pitch = open_pitch + BASS_MAX_FRET
            positions = _all_positions_bass(top_pitch)
            self.assertIn((string_num, BASS_MAX_FRET), positions,
                          f"Fret {BASS_MAX_FRET} on string {string_num} should be reachable")

    def test_above_max_fret_excluded(self):
        """A pitch that exceeds BASS_MAX_FRET on every string returns no positions."""
        # String 1 (G2, MIDI 43) is the highest open string.
        # A pitch beyond its max fret and beyond any other string's range:
        above_all = max(BASS_TUNING.values()) + BASS_MAX_FRET + 5
        positions = _all_positions_bass(above_all)
        self.assertEqual(positions, [],
                         f"MIDI {above_all} should be unreachable on all strings")

    def test_below_open_strings_excluded(self):
        """A pitch lower than the lowest open string returns no positions."""
        below_all = min(BASS_TUNING.values()) - 1
        positions = _all_positions_bass(below_all)
        self.assertEqual(positions, [],
                         f"MIDI {below_all} is below all open strings and should return []")

    def test_mid_range_has_multiple_positions(self):
        """MIDI 38 (D2) is the open pitch of string 2 and fret 5 on string 3 — at least 2 positions."""
        # D2 = MIDI 38 = string 2 open (38) and string 3 fret 5 (33 + 5 = 38)
        positions = _all_positions_bass(38)
        self.assertGreaterEqual(len(positions), 2,
                                "MIDI 38 should have positions on at least string 2 and string 3")
        # Verify the two expected positions are present
        self.assertIn((2, 0), positions, "String 2, fret 0 expected for MIDI 38")
        self.assertIn((3, 5), positions, "String 3, fret 5 expected for MIDI 38")

    def test_fret_values_are_non_negative(self):
        """All returned fret numbers must be >= 0."""
        for midi in range(min(BASS_TUNING.values()), max(BASS_TUNING.values()) + BASS_MAX_FRET + 1):
            for _, fret in _all_positions_bass(midi):
                self.assertGreaterEqual(fret, 0, f"Fret number for MIDI {midi} must be >= 0")

    def test_fret_values_do_not_exceed_max(self):
        """All returned fret numbers must be <= BASS_MAX_FRET."""
        for midi in range(min(BASS_TUNING.values()), max(BASS_TUNING.values()) + BASS_MAX_FRET + 1):
            for _, fret in _all_positions_bass(midi):
                self.assertLessEqual(fret, BASS_MAX_FRET,
                                     f"Fret {fret} for MIDI {midi} exceeds BASS_MAX_FRET={BASS_MAX_FRET}")


if __name__ == "__main__":
    unittest.main()
