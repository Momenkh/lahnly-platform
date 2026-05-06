"""
Tests for pipeline/tab_generation.py

Block format (each block = time marker + 6 string lines):
  [0:00]
  e |--0---3--|
  B |---------|
  G |---------|
  D |---------|
  A |---------|
  E |---------|
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from pipeline.instruments.guitar.tab import generate_tabs, STRING_NAMES
from pipeline.settings import MAPPING_MAX_FRET as MAX_FRET


def make_mapped(string, fret, start, duration=0.5):
    return {"string": string, "fret": fret, "pitch": 50, "start": start, "duration": duration}


def get_string_lines(tab: str) -> list[str]:
    """Return only the 6 string lines from the first block (skip time markers)."""
    lines = []
    for line in tab.strip().split("\n"):
        stripped = line.strip()
        if len(stripped) >= 2 and stripped[1] == " " and stripped[0] in "eBGDAeE":
            lines.append(stripped)
    return lines


class TestTabStructure(unittest.TestCase):

    def test_always_six_string_lines(self):
        notes = [make_mapped(1, 0, 0.0), make_mapped(6, 5, 1.0)]
        tab = generate_tabs(notes, save=False)
        string_lines = get_string_lines(tab)
        self.assertEqual(len(string_lines), 6, "Tab must always have exactly 6 string lines")

    def test_correct_string_labels(self):
        notes = [make_mapped(1, 0, 0.0)]
        tab = generate_tabs(notes, save=False)
        string_lines = get_string_lines(tab)
        expected_labels = ["e", "B", "G", "D", "A", "E"]
        for line, label in zip(string_lines, expected_labels):
            self.assertTrue(line.startswith(label),
                            f"Line should start with '{label}', got: {line[:5]}")

    def test_fret_number_present_in_output(self):
        notes = [make_mapped(string=6, fret=7, start=0.0)]
        tab = generate_tabs(notes, save=False)
        self.assertIn("7", tab)

    def test_note_on_high_e_string(self):
        notes = [make_mapped(string=1, fret=3, start=0.0)]
        tab = generate_tabs(notes, save=False)
        string_lines = get_string_lines(tab)
        high_e_line = string_lines[0]
        self.assertTrue(high_e_line.startswith("e"), f"First string line should be 'e', got: {high_e_line[:3]}")
        self.assertIn("3", high_e_line)

    def test_empty_input_returns_fallback(self):
        tab = generate_tabs([], save=False)
        self.assertIsInstance(tab, str)
        self.assertGreater(len(tab), 0)

    def test_string_lines_contain_pipes(self):
        notes = [make_mapped(1, 0, 0.0)]
        tab = generate_tabs(notes, save=False)
        for line in get_string_lines(tab):
            self.assertIn("|", line)

    def test_time_marker_present(self):
        """Each block should start with a [M:SS] time marker."""
        notes = [make_mapped(1, 0, 0.0)]
        tab = generate_tabs(notes, save=False)
        self.assertRegex(tab, r'\[\d+:\d{2}\]')

    def test_two_notes_different_strings(self):
        notes = [
            make_mapped(string=1, fret=0, start=0.0),
            make_mapped(string=6, fret=5, start=1.0),
        ]
        tab = generate_tabs(notes, save=False)
        self.assertIn("0", tab)
        self.assertIn("5", tab)

    def test_no_fret_exceeds_max(self):
        notes = [make_mapped(string=1, fret=MAX_FRET, start=0.0)]
        tab = generate_tabs(notes, save=False)
        self.assertIn(str(MAX_FRET), tab)


if __name__ == "__main__":
    unittest.main(verbosity=2)
