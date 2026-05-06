"""
Migrate existing flat-format outputs to the new per-instrument subdirectory format.
Renames files in-place within each song's output folder.

Old (flat):  outputs/{song}/01_guitar_stem.wav, 04_tempo.json, ...
New (subdir): outputs/{song}/guitar/01_stem.wav, shared/04_tempo.json, ...

Run once: python migrate_outputs.py
"""

import os
import shutil

OUTPUTS_DIR = "outputs"

# Files that go into the 'shared' subdirectory (cross-instrument)
SHARED_FILES = {
    "04_tempo.json",
    "04_quantized_notes.json",
    "05_key_analysis.json",
}

# Files that go into the 'guitar' subdirectory, with optional rename
GUITAR_FILES = {
    "01_guitar_stem.wav":         "01_stem.wav",
    "01_stem_meta.json":          "01_stem_meta.json",
    "02_raw_notes.json":          "02_raw_notes.json",
    "02_pass1_preview.wav":       "02_pass1_preview.wav",
    "02_pass2_preview.wav":       "02_pass2_preview.wav",
    "02_merged_raw_preview.wav":  "02_merged_raw_preview.wav",
    "03_cleaned_notes.json":      "03_cleaned_notes.json",
    "03_clean_meta.json":         "03_clean_meta.json",
    "06_mapped_notes.json":       "06_mapped_notes.json",
    "07_chords.json":             "07_chords.json",
    "08_tabs.txt":                "08_tabs.txt",
    "09_preview.wav":             "09_preview.wav",
    "10_fretboard.png":           "10_fretboard.png",
    "11_chord_sheet.png":         "11_chord_sheet.png",
}

SKIP_DIRS = {"guitar", "bass", "piano", "vocals", "drums", "shared"}


def migrate_song(song_dir: str) -> None:
    entries = set(os.listdir(song_dir))
    # Skip already-migrated directories
    if "guitar" in entries or "shared" in entries:
        print(f"  [skip] already migrated or empty: {song_dir}")
        return

    guitar_dir = os.path.join(song_dir, "guitar")
    shared_dir = os.path.join(song_dir, "shared")
    os.makedirs(guitar_dir, exist_ok=True)
    os.makedirs(shared_dir, exist_ok=True)

    moved = 0
    for filename in sorted(entries):
        src = os.path.join(song_dir, filename)
        if os.path.isdir(src):
            continue

        if filename in SHARED_FILES:
            dst = os.path.join(shared_dir, filename)
            shutil.move(src, dst)
            print(f"    shared/ {filename}")
            moved += 1
        elif filename in GUITAR_FILES:
            dst = os.path.join(guitar_dir, GUITAR_FILES[filename])
            shutil.move(src, dst)
            new_name = GUITAR_FILES[filename]
            label = f"{filename} -> {new_name}" if new_name != filename else filename
            print(f"    guitar/ {label}")
            moved += 1
        else:
            print(f"    [unknown] {filename} — left in place")

    print(f"  Migrated {moved} files.")


def main():
    if not os.path.isdir(OUTPUTS_DIR):
        print(f"No '{OUTPUTS_DIR}' directory found. Run from core-processor/.")
        return

    songs = sorted(
        d for d in os.listdir(OUTPUTS_DIR)
        if os.path.isdir(os.path.join(OUTPUTS_DIR, d)) and d not in SKIP_DIRS
    )
    print(f"Found {len(songs)} song directories to migrate.\n")

    for song in songs:
        song_dir = os.path.join(OUTPUTS_DIR, song)
        print(f"{song}/")
        migrate_song(song_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
