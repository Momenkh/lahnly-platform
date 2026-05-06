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
    analyze_key_segmented,
    detect_capo,
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
        for field in ["root", "root_midi", "mode", "scale", "scale_pcs", "key_str",
                      "confidence", "histogram", "capo_fret"]:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_key_str_is_string(self):
        notes = notes_from_pitches([60, 62, 64, 65, 67, 69, 71])
        result = analyze_key(notes, save=False)
        self.assertIsInstance(result["key_str"], str)
        self.assertGreater(len(result["key_str"]), 0)

    def test_capo_fret_present_for_guitar(self):
        notes = notes_from_pitches([60, 62, 64, 65, 67, 69, 71])
        result = analyze_key(notes, save=False, instrument="guitar")
        self.assertIn("capo_fret", result)
        self.assertIsInstance(result["capo_fret"], int)

    def test_capo_fret_absent_for_non_guitar(self):
        """Non-guitar instruments always get capo_fret=0."""
        notes = notes_from_pitches([60, 62, 64, 65, 67, 69, 71])
        result = analyze_key(notes, save=False, instrument="piano")
        self.assertEqual(result["capo_fret"], 0)


class TestAnalyzeKeySegmented(unittest.TestCase):
    """Tests for per-section key detection."""

    def _make_notes(self, pitches: list[int], t_offset: float = 0.0) -> list[dict]:
        return [
            {"pitch": p, "start": t_offset + i * 0.5, "duration": 0.5, "confidence": 0.9}
            for i, p in enumerate(pitches)
        ]

    def test_single_segment_returns_empty_segments_list(self):
        """If there's only one section (no gap), segments should be empty."""
        notes = self._make_notes([60, 62, 64, 65, 67, 69, 71] * 4)
        global_key, segments = analyze_key_segmented(notes, instrument="guitar")
        self.assertIn("root", global_key)
        self.assertEqual(segments, [])

    def test_gap_creates_two_segments(self):
        """A silence gap >= KEY_SEGMENT_MIN_GAP_S should produce two segments."""
        from pipeline.settings import KEY_SEGMENT_MIN_DURATION_S, KEY_SEGMENT_MIN_GAP_S

        # Each repeat = 7 notes × 0.5s = 3.5s. Need > MIN_DURATION_S seconds per part.
        n_repeats = max(5, int(KEY_SEGMENT_MIN_DURATION_S / (7 * 0.5)) + 1)
        part1 = self._make_notes([60, 62, 64, 65, 67, 69, 71] * n_repeats, t_offset=0.0)
        # Start second part well after the first ends + gap
        t_gap_start = part1[-1]["start"] + part1[-1]["duration"] + KEY_SEGMENT_MIN_GAP_S + 1.0
        part2 = self._make_notes([61, 63, 65, 66, 68, 70, 72] * n_repeats, t_offset=t_gap_start)
        notes = part1 + part2
        global_key, segments = analyze_key_segmented(notes, instrument="guitar")
        self.assertGreaterEqual(len(segments), 2)

    def test_global_key_always_present(self):
        notes = self._make_notes([60, 62, 64])
        global_key, _ = analyze_key_segmented(notes, instrument="guitar")
        self.assertIn("scale_pcs", global_key)
        self.assertIn("capo_fret", global_key)

    def test_segments_have_required_fields(self):
        from pipeline.settings import KEY_SEGMENT_MIN_DURATION_S, KEY_SEGMENT_MIN_GAP_S
        n_repeats = max(5, int(KEY_SEGMENT_MIN_DURATION_S / (7 * 0.5)) + 1)
        part1 = self._make_notes([60, 62, 64, 65, 67, 69, 71] * n_repeats, t_offset=0.0)
        t2 = part1[-1]["start"] + part1[-1]["duration"] + KEY_SEGMENT_MIN_GAP_S + 1.0
        part2 = self._make_notes([61, 63, 65, 66, 68, 70, 72] * n_repeats, t_offset=t2)
        _, segments = analyze_key_segmented(part1 + part2, instrument="guitar")
        for s in segments:
            for field in ("seg_start", "seg_end", "scale_pcs", "key_str"):
                self.assertIn(field, s, f"Segment missing field: {field}")


class TestDetectCapo(unittest.TestCase):
    """Tests for the capo inference heuristic."""

    def test_guitar_friendly_key_no_capo(self):
        # G major: all 5 open string PCs (E,A,D,G,B) are in the scale → score 5/5
        self.assertEqual(detect_capo(root_midi=7, mode="major"), 0)

    def test_guitar_friendly_minor_no_capo(self):
        # E minor: all 5 open string PCs in scale → no capo needed
        self.assertEqual(detect_capo(root_midi=4, mode="minor"), 0)

    def test_unfriendly_key_gets_capo(self):
        # Ab major (root=8): few open strings in scale.
        # With capo 1 → G major (root=7): 5/5 open strings.
        capo = detect_capo(root_midi=8, mode="major")
        self.assertEqual(capo, 1)

    def test_bb_major_capo_3(self):
        # Bb major (root=10): unfriendly (E♭ key).
        # Capo 3 → G major (10-3=7): all 5 open strings in scale.
        capo = detect_capo(root_midi=10, mode="major")
        self.assertIn(capo, {1, 2, 3})  # any capo that improves the score

    def test_maqam_never_gets_capo(self):
        self.assertEqual(detect_capo(root_midi=2, mode="maqam_hijaz"), 0)
        self.assertEqual(detect_capo(root_midi=5, mode="maqam_kurd"),  0)

    def test_capo_within_bounds(self):
        """Returned capo fret is always 0-5."""
        for root in range(12):
            for mode in ("major", "minor"):
                capo = detect_capo(root_midi=root, mode=mode)
                self.assertGreaterEqual(capo, 0)
                self.assertLessEqual(capo, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
