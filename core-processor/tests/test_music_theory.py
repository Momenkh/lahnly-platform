"""
Tests for pipeline/music_theory.py

Covers:
  - Key detection on synthetic pitch histograms
  - Scale pitch class generation
  - Pentatonic detection preference
  - Confidence is a valid correlation value
  - All 12 roots detectable
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
from pipeline.shared.music_theory import (
    analyze_key,
    _detect_key,
    _build_histogram,
    _scale_pitch_classes,
    SCALES,
    CHROMATIC,
)


def notes_from_pitches(midi_pitches: list[int], duration: float = 1.0) -> list[dict]:
    """Helper: build a note list from MIDI pitches."""
    return [
        {"pitch": p, "start": i * duration, "duration": duration, "confidence": 0.9}
        for i, p in enumerate(midi_pitches)
    ]


class TestKeyDetection(unittest.TestCase):

    def _dominant_histogram(self, root: int, mode: str) -> np.ndarray:
        """
        Build a histogram that mimics real playing: root and fifth are
        heavily emphasised, other scale tones have moderate weight, and
        non-scale tones are absent. This gives the KS algorithm enough
        information to separate a key from its relative major/minor.
        """
        hist = np.zeros(12)
        intervals = [0, 2, 3, 5, 7, 8, 10] if mode == "minor" else [0, 2, 4, 5, 7, 9, 11]
        weights   = [5.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0]  # root=5, fifth=3, others=1
        for interval, weight in zip(intervals, weights):
            hist[(root + interval) % 12] = weight
        hist /= hist.sum()
        return hist

    def test_c_major_detected(self):
        hist = self._dominant_histogram(0, "major")
        root, mode, conf, _ = _detect_key(hist, n_notes=12)
        self.assertEqual(root, 0)
        self.assertEqual(mode, "major")

    def test_e_minor_detected(self):
        hist = self._dominant_histogram(4, "minor")  # E = pitch class 4
        root, mode, conf, _ = _detect_key(hist, n_notes=12)
        self.assertEqual(root, 4)
        self.assertEqual(mode, "minor")

    def test_d_minor_detected(self):
        hist = self._dominant_histogram(2, "minor")  # D = pitch class 2
        root, mode, conf, _ = _detect_key(hist, n_notes=12)
        self.assertEqual(root, 2)
        self.assertEqual(mode, "minor")

    def test_confidence_between_minus1_and_1(self):
        hist = self._dominant_histogram(0, "major")
        _, _, conf, _ = _detect_key(hist, n_notes=12)
        self.assertGreaterEqual(conf, -1.0)
        self.assertLessEqual(conf, 1.0)

    def test_all_12_roots_detectable(self):
        """Each root should be detectable when given a pure scale histogram."""
        for root in range(12):
            hist = self._dominant_histogram(root, "minor")
            detected_root, _, _, _ = _detect_key(hist, n_notes=12)
            self.assertEqual(detected_root, root,
                             f"Root {CHROMATIC[root]} not detected correctly")


class TestScalePitchClasses(unittest.TestCase):

    def test_c_major_pitch_classes(self):
        pcs = _scale_pitch_classes(0, "major")
        # C D E F G A B = 0 2 4 5 7 9 11
        self.assertEqual(sorted(pcs), [0, 2, 4, 5, 7, 9, 11])

    def test_a_minor_pitch_classes(self):
        pcs = _scale_pitch_classes(9, "natural_minor")
        # A B C D E F G = 9 11 0 2 4 5 7
        self.assertEqual(sorted(pcs), [0, 2, 4, 5, 7, 9, 11])

    def test_e_pentatonic_minor(self):
        pcs = _scale_pitch_classes(4, "pentatonic_minor")
        # E G A B D = 4 7 9 11 2
        self.assertEqual(sorted(pcs), [2, 4, 7, 9, 11])

    def test_scale_length(self):
        """Each scale should have the correct number of notes."""
        expected_lengths = {
            "major": 7, "natural_minor": 7,
            "pentatonic_major": 5, "pentatonic_minor": 5,
            "blues": 6, "dorian": 7, "mixolydian": 7,
        }
        for scale, expected_len in expected_lengths.items():
            pcs = _scale_pitch_classes(0, scale)
            self.assertEqual(len(pcs), expected_len,
                             f"{scale} should have {expected_len} notes, got {len(pcs)}")

    def test_no_duplicate_pitch_classes(self):
        for scale in SCALES:
            pcs = _scale_pitch_classes(0, scale)
            self.assertEqual(len(pcs), len(set(pcs)),
                             f"{scale} has duplicate pitch classes")


class TestBuildHistogram(unittest.TestCase):

    def test_single_pitch_dominates(self):
        notes = notes_from_pitches([60] * 10)  # 10x C4
        hist = _build_histogram(notes)
        self.assertAlmostEqual(hist[0], 1.0, places=3)  # C = pitch class 0

    def test_histogram_sums_to_one(self):
        notes = notes_from_pitches([60, 62, 64, 65, 67, 69, 71])
        hist = _build_histogram(notes)
        self.assertAlmostEqual(hist.sum(), 1.0, places=5)

    def test_empty_notes_returns_zeros(self):
        hist = _build_histogram([])
        self.assertTrue(np.all(hist == 0))

    def test_duration_weighting(self):
        """Longer notes should contribute more to the histogram."""
        notes = [
            {"pitch": 60, "start": 0.0, "duration": 4.0, "confidence": 0.9},  # C
            {"pitch": 62, "start": 4.0, "duration": 1.0, "confidence": 0.9},  # D
        ]
        hist = _build_histogram(notes)
        self.assertGreater(hist[0], hist[2])  # C > D


class TestAnalyzeKeyEndToEnd(unittest.TestCase):

    def test_returns_required_fields(self):
        notes = notes_from_pitches([50, 52, 53, 55, 57, 58, 60])  # D natural minor
        result = analyze_key(notes, save=False)
        for field in ["root", "root_midi", "mode", "scale", "scale_pcs", "key_str", "confidence", "histogram"]:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_key_str_is_string(self):
        notes = notes_from_pitches([60, 62, 64, 65, 67, 69, 71])
        result = analyze_key(notes, save=False)
        self.assertIsInstance(result["key_str"], str)
        self.assertGreater(len(result["key_str"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
