"""
Tests for pipeline/instruments/piano/mapping.py

Covers:
  - MIDI 60 (middle C) is assigned to the right hand
  - MIDI 48 (C3) is assigned to the left hand
  - The split point itself (PIANO_LH_RH_SPLIT_MIDI) goes to the right hand
  - Notes below PIANO_MIDI_MIN are silently excluded
  - Notes above PIANO_MIDI_MAX are silently excluded
  - key_index is 0-based from PIANO_MIDI_MIN (A0)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from pipeline.instruments.piano.mapping import map_to_piano
from pipeline.settings import PIANO_MIDI_MIN, PIANO_MIDI_MAX, PIANO_LH_RH_SPLIT_MIDI


def _note(pitch, start=0.0, duration=0.5, confidence=0.9):
    return {"pitch": pitch, "start": start, "duration": duration, "confidence": confidence}


class TestMapToPiano(unittest.TestCase):

    def test_middle_c_is_right_hand(self):
        """MIDI 60 (C4, middle C) is at the split point and should go right hand."""
        result = map_to_piano([_note(60)], save=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["hand"], "right")

    def test_below_split_is_left_hand(self):
        """MIDI 48 (C3) is below middle C and should go to the left hand."""
        result = map_to_piano([_note(48)], save=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["hand"], "left")

    def test_above_split_is_right_hand(self):
        """MIDI 72 (C5) is above middle C and should go to the right hand."""
        result = map_to_piano([_note(72)], save=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["hand"], "right")

    def test_split_point_is_right_hand(self):
        """The exact split pitch should be assigned to the right hand (>= split → right)."""
        result = map_to_piano([_note(PIANO_LH_RH_SPLIT_MIDI)], save=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["hand"], "right")

    def test_one_below_split_is_left_hand(self):
        """One semitone below the split should be assigned to the left hand."""
        result = map_to_piano([_note(PIANO_LH_RH_SPLIT_MIDI - 1)], save=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["hand"], "left")

    def test_below_min_excluded(self):
        """MIDI below PIANO_MIDI_MIN should be silently dropped."""
        result = map_to_piano([_note(PIANO_MIDI_MIN - 1)], save=False)
        self.assertEqual(result, [])

    def test_above_max_excluded(self):
        """MIDI above PIANO_MIDI_MAX should be silently dropped."""
        result = map_to_piano([_note(PIANO_MIDI_MAX + 1)], save=False)
        self.assertEqual(result, [])

    def test_boundary_min_included(self):
        """PIANO_MIDI_MIN itself should be included and have key_index=0."""
        result = map_to_piano([_note(PIANO_MIDI_MIN)], save=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["key_index"], 0)

    def test_boundary_max_included(self):
        """PIANO_MIDI_MAX itself should be included."""
        result = map_to_piano([_note(PIANO_MIDI_MAX)], save=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["key_index"], PIANO_MIDI_MAX - PIANO_MIDI_MIN)

    def test_key_index_is_zero_based(self):
        """key_index should equal pitch - PIANO_MIDI_MIN."""
        for pitch in [21, 40, 60, 88, 108]:
            result = map_to_piano([_note(pitch)], save=False)
            self.assertEqual(result[0]["key_index"], pitch - PIANO_MIDI_MIN,
                             f"key_index for MIDI {pitch} should be {pitch - PIANO_MIDI_MIN}")

    def test_output_preserves_timing(self):
        """start and duration should pass through unchanged."""
        result = map_to_piano([_note(60, start=1.5, duration=0.25)], save=False)
        self.assertAlmostEqual(result[0]["start"], 1.5)
        self.assertAlmostEqual(result[0]["duration"], 0.25)

    def test_mixed_notes_hand_split(self):
        """A mix of low and high pitches correctly distributes hands."""
        notes = [_note(48), _note(60), _note(72)]
        result = map_to_piano(notes, save=False)
        hands = {n["pitch"]: n["hand"] for n in result}
        self.assertEqual(hands[48], "left")
        self.assertEqual(hands[60], "right")
        self.assertEqual(hands[72], "right")


if __name__ == "__main__":
    unittest.main()
