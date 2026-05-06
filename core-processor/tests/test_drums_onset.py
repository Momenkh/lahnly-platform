"""
Tests for pipeline/instruments/drums/onset._classify_hit

Patches librosa.feature.spectral_centroid and spectral_flatness so tests
run without audio I/O and produce deterministic centroid/flatness values.

Decision tree from _classify_hit:
  centroid <= KICK_MAX                                            → "kick"
  centroid >= HIHAT_MIN and flatness >= FLATNESS_HIHAT_MIN
    + spectral flux > HIHAT_OPEN_FLUX_THRESH                     → "hihat_open"
    + spectral flux <= HIHAT_OPEN_FLUX_THRESH                    → "hihat_closed"
  centroid >= HIHAT_MIN (flatness too low)                       → "cymbal"
  flatness >= FLATNESS_SNARE_MIN                                 → "snare"
  centroid <= TOM_MAX                                            → "tom"
  else                                                           → "snare" (default)

Note: spectral flux is computed directly from the window (np.diff), not via librosa,
so it is controlled by the window array passed to _classify_hit rather than mocking.
_WIN is all-zeros → flux=0 → "hihat_closed".  _WIN_OPEN alternates sign → high flux.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import patch
import numpy as np

from pipeline.instruments.drums.onset import _classify_hit
from pipeline.settings import (
    DRUMS_CENTROID_KICK_MAX,
    DRUMS_CENTROID_TOM_MAX,
    DRUMS_CENTROID_HIHAT_MIN,
    DRUMS_FLATNESS_SNARE_MIN,
    DRUMS_FLATNESS_HIHAT_MIN,
)

_SR      = 22050
_WIN     = np.zeros(512, dtype=np.float32)           # all-zeros → flux=0 → "hihat_closed"
_WIN_OPEN = np.tile([1.0, -1.0], 256).astype(np.float32)  # alternating → high flux → "hihat_open"


def _mock_classify(centroid_hz: float, flatness: float, window=None) -> tuple[str, float]:
    """Call _classify_hit with mocked librosa spectral features.

    librosa is imported inside _classify_hit (local import), so we patch
    the canonical module attributes directly.  The spectral flux branch uses
    np.diff on the window directly (no librosa), so pass window=_WIN_OPEN to
    trigger the open hi-hat path.
    """
    if window is None:
        window = _WIN
    centroid_arr = np.array([[centroid_hz]])
    flatness_arr = np.array([[flatness]])
    with patch("librosa.feature.spectral_centroid", return_value=centroid_arr), \
         patch("librosa.feature.spectral_flatness", return_value=flatness_arr):
        return _classify_hit(window, _SR)


class TestClassifyHit(unittest.TestCase):

    def test_low_centroid_is_kick(self):
        """A centroid well below KICK_MAX should classify as 'kick'."""
        hit_class, conf = _mock_classify(centroid_hz=200.0, flatness=0.02)
        self.assertEqual(hit_class, "kick")
        self.assertGreater(conf, 0.0)

    def test_at_kick_threshold_is_kick(self):
        """A centroid exactly at KICK_MAX should still be 'kick' (<= comparison)."""
        hit_class, _ = _mock_classify(centroid_hz=DRUMS_CENTROID_KICK_MAX, flatness=0.01)
        self.assertEqual(hit_class, "kick")

    def test_high_centroid_high_flatness_is_hihat_closed(self):
        """High centroid + high flatness + zero flux (silent window) → 'hihat_closed'."""
        hit_class, conf = _mock_classify(
            centroid_hz=DRUMS_CENTROID_HIHAT_MIN + 1000.0,
            flatness=DRUMS_FLATNESS_HIHAT_MIN + 0.05,
            window=_WIN,   # all-zeros → flux=0 → closed
        )
        self.assertEqual(hit_class, "hihat_closed")
        self.assertGreater(conf, 0.0)

    def test_high_centroid_high_flatness_high_flux_is_hihat_open(self):
        """High centroid + high flatness + high flux (alternating window) → 'hihat_open'."""
        hit_class, conf = _mock_classify(
            centroid_hz=DRUMS_CENTROID_HIHAT_MIN + 1000.0,
            flatness=DRUMS_FLATNESS_HIHAT_MIN + 0.05,
            window=_WIN_OPEN,   # alternating ±1 → max flux → open
        )
        self.assertEqual(hit_class, "hihat_open")
        self.assertGreater(conf, 0.0)

    def test_high_centroid_low_flatness_is_cymbal(self):
        """High centroid + flatness below hi-hat threshold → 'cymbal'."""
        hit_class, conf = _mock_classify(
            centroid_hz=DRUMS_CENTROID_HIHAT_MIN + 1000.0,
            flatness=DRUMS_FLATNESS_HIHAT_MIN - 0.01,
        )
        self.assertEqual(hit_class, "cymbal")
        self.assertGreater(conf, 0.0)

    def test_mid_centroid_high_flatness_is_snare(self):
        """Mid centroid (between kick max and hi-hat min) + high flatness → 'snare'."""
        mid_centroid = (DRUMS_CENTROID_KICK_MAX + DRUMS_CENTROID_HIHAT_MIN) / 2
        hit_class, conf = _mock_classify(
            centroid_hz=mid_centroid,
            flatness=DRUMS_FLATNESS_SNARE_MIN + 0.05,
        )
        self.assertEqual(hit_class, "snare")
        self.assertGreater(conf, 0.0)

    def test_low_mid_centroid_low_flatness_is_tom(self):
        """Mid centroid <= TOM_MAX + flatness below snare threshold → 'tom'."""
        hit_class, conf = _mock_classify(
            centroid_hz=DRUMS_CENTROID_TOM_MAX - 100.0,
            flatness=DRUMS_FLATNESS_SNARE_MIN - 0.02,
        )
        self.assertEqual(hit_class, "tom")
        self.assertGreater(conf, 0.0)

    def test_confidence_in_unit_range(self):
        """All classifier paths should return confidence in [0, 1]."""
        scenarios = [
            (200.0,  0.02, _WIN),       # kick
            (5000.0, 0.15, _WIN),       # hihat_closed
            (5000.0, 0.15, _WIN_OPEN),  # hihat_open
            (5000.0, 0.05, _WIN),       # cymbal
            (800.0,  0.10, _WIN),       # snare
            (1000.0, 0.02, _WIN),       # tom
        ]
        for centroid, flatness, window in scenarios:
            _, conf = _mock_classify(centroid, flatness, window)
            self.assertGreaterEqual(conf, 0.0,
                                    f"confidence < 0 for centroid={centroid}, flatness={flatness}")
            self.assertLessEqual(conf, 1.0,
                                 f"confidence > 1 for centroid={centroid}, flatness={flatness}")

    def test_returned_class_is_known(self):
        """Every classification result should be a known hit class."""
        from pipeline.settings import DRUMS_HIT_CLASSES
        scenarios = [
            (200.0,  0.02, _WIN),
            (5000.0, 0.15, _WIN),
            (5000.0, 0.15, _WIN_OPEN),
            (5000.0, 0.05, _WIN),
            (800.0,  0.10, _WIN),
            (1000.0, 0.02, _WIN),
        ]
        for centroid, flatness, window in scenarios:
            hit_class, _ = _mock_classify(centroid, flatness, window)
            self.assertIn(hit_class, DRUMS_HIT_CLASSES,
                          f"'{hit_class}' is not a known drum hit class")


if __name__ == "__main__":
    unittest.main()
