"""
Stage 6 (Piano): Piano Mapping
Input:  cleaned piano notes [{pitch, start, duration, confidence}]
Output: mapped notes [{hand, key_index, pitch, start, duration, confidence}]
        saved to outputs/06_piano_mapped_notes.json

Piano has no fretboard — mapping is trivial:
  - key_index : 0-based index from A0 (MIDI 21), matching the physical piano key
  - hand      : "left" if pitch < PIANO_LH_RH_SPLIT_MIDI, else "right"

There is no Viterbi DP needed — every MIDI pitch maps to exactly one piano key.
Left/right hand split is done at middle C (MIDI 60) by default.
"""

import json
import os

from pipeline.config import get_instrument_dir
from pipeline.settings import PIANO_MIDI_MIN, PIANO_MIDI_MAX, PIANO_LH_RH_SPLIT_MIDI


def map_to_piano(
    cleaned_notes: list[dict],
    key_info: dict | None = None,
    save: bool = True,
) -> list[dict]:
    if key_info:
        print(f"[Stage 6P] Mapping {len(cleaned_notes)} notes  "
              f"(key: {key_info['key_str']})")
    else:
        print(f"[Stage 6P] Mapping {len(cleaned_notes)} notes")

    mapped  = []
    skipped = 0

    for note in sorted(cleaned_notes, key=lambda n: n["start"]):
        pitch = note["pitch"]
        if not (PIANO_MIDI_MIN <= pitch <= PIANO_MIDI_MAX):
            skipped += 1
            continue

        key_index = pitch - PIANO_MIDI_MIN   # 0-based from A0
        hand      = "right" if pitch >= PIANO_LH_RH_SPLIT_MIDI else "left"

        mapped.append({
            "hand":       hand,
            "key_index":  key_index,
            "pitch":      pitch,
            "start":      note["start"],
            "duration":   note["duration"],
            "confidence": note.get("confidence", 0.5),
        })

    lh = sum(1 for n in mapped if n["hand"] == "left")
    rh = sum(1 for n in mapped if n["hand"] == "right")
    print(f"[Stage 6P] Mapped {len(mapped)} notes  "
          f"(left hand: {lh}  right hand: {rh})"
          + (f"  skipped {skipped} out-of-range" if skipped else ""))

    if save:
        out_path = os.path.join(get_instrument_dir("piano"), "06_mapped_notes.json")
        with open(out_path, "w") as f:
            json.dump(mapped, f, indent=2)
        print(f"[Stage 6P] Saved -> {out_path}")

    return mapped


def load_mapped_notes_piano() -> list[dict]:
    path = os.path.join(get_instrument_dir("piano"), "06_mapped_notes.json")
    with open(path) as f:
        return json.load(f)
