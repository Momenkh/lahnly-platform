"""
Tests for pipeline/note_cleaning.py

Covers:
  - Short notes are removed
  - Low-confidence notes are removed
  - Nearby same-pitch notes are merged
  - Output is sorted by start time
  - Empty input is handled
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from pipeline.instruments.guitar.cleaning import clean_notes, _limit_polyphony
from pipeline.settings import CLEANING_TYPE_PARAMS, CLEANING_DEFAULT_MERGE_GAP_S

# Pull thresholds from the default guitar mode so tests stay in sync with settings.
_DEFAULT_MODE   = "clean_lead"
_PARAMS         = CLEANING_TYPE_PARAMS[_DEFAULT_MODE]
MIN_DURATION_S      = _PARAMS[0]
CONFIDENCE_THRESHOLD = _PARAMS[1]   # conf_floor (minimum confidence before stem scaling)
MERGE_GAP_S         = CLEANING_DEFAULT_MERGE_GAP_S
MAX_POLYPHONY       = _PARAMS[4]


def make_note(pitch=60, start=0.0, duration=0.5, confidence=0.9):
    return {"pitch": pitch, "start": start, "duration": duration, "confidence": confidence}


class TestDurationFilter(unittest.TestCase):

    def test_short_note_removed(self):
        notes = [make_note(duration=MIN_DURATION_S - 0.01)]
        result = clean_notes(notes, save=False)
        self.assertEqual(result, [])

    def test_exact_min_duration_kept(self):
        notes = [make_note(duration=MIN_DURATION_S)]
        result = clean_notes(notes, guitar_type="clean", guitar_role="lead", save=False)
        self.assertEqual(len(result), 1)

    def test_long_note_kept(self):
        notes = [make_note(duration=1.0)]
        result = clean_notes(notes, save=False)
        self.assertEqual(len(result), 1)


class TestConfidenceFilter(unittest.TestCase):

    def test_low_confidence_removed(self):
        notes = [make_note(confidence=CONFIDENCE_THRESHOLD - 0.01)]
        result = clean_notes(notes, save=False)
        self.assertEqual(result, [])

    def test_exact_threshold_kept(self):
        # conf_floor is the threshold with a perfect stem (stem_conf=1.0); in tests
        # stem_conf defaults to 0.5, so the effective threshold is conf_floor + 0.5*scale.
        # Use a value reliably above the adaptive threshold for clean_lead (max=0.26).
        notes = [make_note(confidence=CONFIDENCE_THRESHOLD + 0.2)]
        result = clean_notes(notes, guitar_type="clean", guitar_role="lead", save=False)
        self.assertEqual(len(result), 1)


class TestMerging(unittest.TestCase):

    def test_same_pitch_tiny_gap_merged(self):
        """Two same-pitch notes with a gap < MERGE_GAP_S should become one."""
        gap = MERGE_GAP_S - 0.01
        n1 = make_note(pitch=60, start=0.0,           duration=0.3)
        n2 = make_note(pitch=60, start=0.3 + gap,     duration=0.3)
        result = clean_notes([n1, n2], save=False)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["start"], 0.0, places=3)

    def test_same_pitch_large_gap_not_merged(self):
        """Two same-pitch notes with a large gap should stay separate."""
        n1 = make_note(pitch=60, start=0.0, duration=0.3)
        n2 = make_note(pitch=60, start=1.0, duration=0.3)
        result = clean_notes([n1, n2], save=False)
        self.assertEqual(len(result), 2)

    def test_different_pitch_not_merged(self):
        """Two different pitches close together should not be merged."""
        n1 = make_note(pitch=60, start=0.0, duration=0.3)
        n2 = make_note(pitch=61, start=0.31, duration=0.3)
        result = clean_notes([n1, n2], save=False)
        self.assertEqual(len(result), 2)

    def test_merged_duration_covers_both(self):
        """Merged note duration should span from first start to last end."""
        n1 = make_note(pitch=60, start=0.0, duration=0.3)
        n2 = make_note(pitch=60, start=0.32, duration=0.3)
        result = clean_notes([n1, n2], save=False)
        self.assertEqual(len(result), 1)
        expected_end = 0.32 + 0.3
        actual_end = result[0]["start"] + result[0]["duration"]
        self.assertAlmostEqual(actual_end, expected_end, places=3)


class TestSorting(unittest.TestCase):

    def test_output_sorted_by_start(self):
        notes = [
            make_note(pitch=62, start=1.0),
            make_note(pitch=60, start=0.0),
            make_note(pitch=64, start=0.5),
        ]
        result = clean_notes(notes, save=False)
        starts = [n["start"] for n in result]
        self.assertEqual(starts, sorted(starts))


class TestEdgeCases(unittest.TestCase):

    def test_empty_input(self):
        result = clean_notes([], save=False)
        self.assertEqual(result, [])

    def test_single_valid_note_passes_through(self):
        notes = [make_note()]
        result = clean_notes(notes, save=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pitch"], 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPolyphonyLimit(unittest.TestCase):

    def _make(self, pitch, start, duration, confidence):
        return {"pitch": pitch, "start": start, "duration": duration, "confidence": confidence}

    def test_under_limit_unchanged(self):
        """Fewer notes than limit should all survive."""
        notes = [
            self._make(60, 0.0, 1.0, 0.9),
            self._make(64, 0.0, 1.0, 0.8),
        ]
        result = _limit_polyphony(notes, max_poly=4)
        self.assertEqual(len(result), 2)

    def test_over_limit_drops_lowest_confidence(self):
        """When 5 notes overlap and limit=4, the quietest is removed."""
        notes = [self._make(60 + i, 0.0, 1.0, (i + 1) * 0.1) for i in range(5)]
        result = _limit_polyphony(notes, max_poly=4)
        self.assertEqual(len(result), 4)
        confidences = [n["confidence"] for n in result]
        self.assertNotIn(0.1, confidences)  # quietest removed

    def test_non_overlapping_notes_unaffected(self):
        """Notes that don't overlap should never be removed by polyphony limit."""
        notes = [self._make(60 + i, i * 2.0, 1.0, 0.5) for i in range(10)]
        result = _limit_polyphony(notes, max_poly=1)
        self.assertEqual(len(result), 10)

    def test_high_polyphony_limit_keeps_all(self):
        """With a very high polyphony limit, all notes survive."""
        notes = [self._make(60 + i, 0.0, 1.0, 0.9) for i in range(4)]
        result = _limit_polyphony(notes, max_poly=100)
        self.assertEqual(len(result), 4)
