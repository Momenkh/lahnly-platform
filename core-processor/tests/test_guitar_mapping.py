"""
Tests for pipeline/guitar_mapping.py

Covers:
  - Every open string maps to fret 0
  - Fret 21 is reachable on every string
  - Out-of-range pitches return no positions
  - Hand window keeps consecutive notes within HAND_SPAN
  - Window shifts correctly when notes are far apart
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from pipeline.instruments.guitar.mapping import (
    _all_positions,
    map_to_guitar,
    STANDARD_TUNING,
)
from pipeline.settings import MAPPING_MAX_FRET as MAX_FRET, MAPPING_HAND_SPAN as HAND_SPAN


class TestAllPositions(unittest.TestCase):

    def test_open_strings(self):
        """Each open string pitch should map to fret 0 on its string."""
        expected = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
        for string, open_pitch in expected.items():
            positions = _all_positions(open_pitch)
            self.assertIn((string, 0), positions,
                          f"String {string} open pitch {open_pitch} should map to fret 0")

    def test_max_fret_reachable(self):
        """Pitch at fret MAX_FRET on each string should be included."""
        for string, open_pitch in STANDARD_TUNING.items():
            top_pitch = open_pitch + MAX_FRET
            positions = _all_positions(top_pitch)
            self.assertIn((string, MAX_FRET), positions,
                          f"Fret {MAX_FRET} on string {string} should be reachable")

    def test_above_max_fret_excluded(self):
        """A pitch requiring fret > MAX_FRET should not appear."""
        # Highest string 1 open is 64; fret 22 = pitch 86
        too_high_on_string1 = STANDARD_TUNING[1] + MAX_FRET + 1
        positions = _all_positions(too_high_on_string1)
        for string, fret in positions:
            self.assertLessEqual(fret, MAX_FRET,
                                 f"Fret {fret} exceeds MAX_FRET={MAX_FRET}")

    def test_below_open_excluded(self):
        """Pitch below every open string should return no positions."""
        below_all = STANDARD_TUNING[6] - 1  # below E2
        positions = _all_positions(below_all)
        self.assertEqual(positions, [], "Pitch below all open strings should have no positions")

    def test_multiple_positions_for_mid_range(self):
        """A mid-range pitch should be playable on multiple strings."""
        # A3 = MIDI 57 — playable on strings 5, 4, 3
        positions = _all_positions(57)
        self.assertGreater(len(positions), 1,
                           "Mid-range pitch should have multiple string options")


class TestHandWindow(unittest.TestCase):

    def _make_note(self, pitch, start, duration=0.5):
        return {"pitch": pitch, "start": start, "duration": duration, "confidence": 0.9}

    def test_consecutive_notes_stay_in_window(self):
        """A scale run should not spread across the entire fretboard."""
        # D minor scale ascending from D3 (MIDI 50) to D4 (MIDI 62) — one octave.
        # The Viterbi DP minimises global hand travel; the fret span may exceed a
        # single HAND_SPAN when the scale covers an octave (12 semitones), but it
        # should never exceed roughly 3× HAND_SPAN (i.e. 3 position shifts).
        d_minor = [50, 52, 53, 55, 57, 58, 60, 62]
        notes = [self._make_note(p, i * 0.5) for i, p in enumerate(d_minor)]
        mapped = map_to_guitar(notes, save=False)
        frets = [n["fret"] for n in mapped]
        span = max(frets) - min(frets)
        self.assertLessEqual(span, HAND_SPAN * 3,
                             f"Hand span {span} is too large for a simple scale run")

    def test_all_notes_mapped(self):
        """Every note in guitar range should produce a mapped output."""
        notes = [
            self._make_note(40, 0.0),   # E2  — lowest (string 6 fret 0)
            self._make_note(64, 0.5),   # E4  — open high e (string 1 fret 0)
            self._make_note(85, 1.0),   # C#6 — highest on 21-fret guitar (string 1 fret 21)
        ]
        mapped = map_to_guitar(notes, save=False)
        self.assertEqual(len(mapped), 3)

    def test_out_of_range_note_skipped(self):
        """Notes outside guitar range should be silently dropped."""
        notes = [
            self._make_note(39, 0.0),   # below E2
            self._make_note(50, 0.5),   # D3 — valid
            self._make_note(89, 1.0),   # above E6
        ]
        mapped = map_to_guitar(notes, save=False)
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0]["pitch"], 50)

    def test_window_shift_on_large_jump(self):
        """A large pitch jump should shift the hand window."""
        notes = [
            self._make_note(40, 0.0),   # E2 fret 0 on string 6
            self._make_note(83, 0.5),   # B5 — very high, forces shift
        ]
        mapped = map_to_guitar(notes, save=False)
        self.assertEqual(len(mapped), 2)
        # Second note's fret should be far from first
        self.assertNotEqual(mapped[0]["fret"], mapped[1]["fret"])


class TestFretLimits(unittest.TestCase):

    def test_max_fret_is_21(self):
        self.assertEqual(MAX_FRET, 21, "MAX_FRET should be 21")


if __name__ == "__main__":
    unittest.main(verbosity=2)
