"""One-shot migration script: update all output paths to use instrument subdirectories."""
import re, os

BASE = os.path.join(os.path.dirname(__file__), "pipeline")

def do(path, replacements):
    full = os.path.join(BASE, path)
    with open(full, encoding="utf-8") as f:
        text = f.read()
    changed = False
    for pat, repl in replacements:
        new = re.sub(pat, repl, text)
        if new != text:
            text = new
            changed = True
    if changed:
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Updated: {path}")
    else:
        print(f"  No change: {path}")

# ── Guitar ────────────────────────────────────────────────────────────────────
print("== Guitar ==")

do("instruments/guitar/pitch.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    # inner local import inside _save_pass_preview
    (r'    from pipeline\.config import get_outputs_dir\n',
     ''),
    # _save_pass_preview makedirs+path
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), filename\)',
     r'out_path = os.path.join(get_instrument_dir("guitar"), filename)'),
    # _save makedirs+path
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "02_raw_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("guitar"), "02_raw_notes.json")'),
    # load_raw_notes
    (r'path = os\.path\.join\(get_outputs_dir\(\), "02_raw_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("guitar"), "02_raw_notes.json")'),
])

do("instruments/guitar/cleaning.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir, get_shared_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "03_cleaned_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("guitar"), "03_cleaned_notes.json")'),
    (r'meta_path = os\.path\.join\(get_outputs_dir\(\), "03_clean_meta\.json"\)',
     'meta_path = os.path.join(get_instrument_dir("guitar"), "03_clean_meta.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "03_clean_meta\.json"\)',
     'path = os.path.join(get_instrument_dir("guitar"), "03_clean_meta.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "04_tempo\.json"\)',
     'path = os.path.join(get_shared_dir(), "04_tempo.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "03_cleaned_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("guitar"), "03_cleaned_notes.json")'),
])

do("instruments/guitar/mapping.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "06_mapped_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("guitar"), "06_mapped_notes.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "06_mapped_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("guitar"), "06_mapped_notes.json")'),
])

do("instruments/guitar/chords.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "07_chords\.json"\)',
     'path = os.path.join(get_instrument_dir("guitar"), "07_chords.json")'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "07_chords\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("guitar"), "07_chords.json")'),
])

do("instruments/guitar/tab.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "08_tabs\.txt"\)',
     r'out_path = os.path.join(get_instrument_dir("guitar"), "08_tabs.txt")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "08_tabs\.txt"\)',
     'path = os.path.join(get_instrument_dir("guitar"), "08_tabs.txt")'),
])

do("instruments/guitar/visualization.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "10_fretboard\.png"\)',
     r'out_path = os.path.join(get_instrument_dir("guitar"), "10_fretboard.png")'),
])

do("instruments/guitar/chord_sheet.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "11_chord_sheet\.png"\)',
     r'out_path = os.path.join(get_instrument_dir("guitar"), "11_chord_sheet.png")'),
])

do("instruments/guitar/pipeline.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'preview_path = os\.path\.join\(get_outputs_dir\(\), "09_preview\.wav"\)',
     'preview_path = os.path.join(get_instrument_dir("guitar"), "09_preview.wav")'),
])

# ── Bass ──────────────────────────────────────────────────────────────────────
print("== Bass ==")

do("instruments/bass/pitch.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "02_bass_raw_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("bass"), "02_raw_notes.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "02_bass_raw_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("bass"), "02_raw_notes.json")'),
])

do("instruments/bass/cleaning.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir, get_shared_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "03_bass_cleaned_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("bass"), "03_cleaned_notes.json")'),
    (r'meta_path = os\.path\.join\(get_outputs_dir\(\), "03_bass_clean_meta\.json"\)',
     'meta_path = os.path.join(get_instrument_dir("bass"), "03_clean_meta.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "04_tempo\.json"\)',
     'path = os.path.join(get_shared_dir(), "04_tempo.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "03_bass_cleaned_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("bass"), "03_cleaned_notes.json")'),
])

do("instruments/bass/mapping.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "06_bass_mapped_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("bass"), "06_mapped_notes.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "06_bass_mapped_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("bass"), "06_mapped_notes.json")'),
])

do("instruments/bass/tab.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "08_bass_tabs\.txt"\)',
     r'out_path = os.path.join(get_instrument_dir("bass"), "08_tabs.txt")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "08_bass_tabs\.txt"\)',
     'path = os.path.join(get_instrument_dir("bass"), "08_tabs.txt")'),
])

do("instruments/bass/visualization.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "10_bass_fretboard\.png"\)',
     r'out_path = os.path.join(get_instrument_dir("bass"), "10_fretboard.png")'),
])

do("instruments/bass/pipeline.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'out_path\s*= os\.path\.join\(get_outputs_dir\(\), "09_bass_preview\.wav"\)',
     'out_path  = os.path.join(get_instrument_dir("bass"), "09_preview.wav")'),
])

# ── Piano ─────────────────────────────────────────────────────────────────────
print("== Piano ==")

do("instruments/piano/pitch.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "02_piano_raw_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("piano"), "02_raw_notes.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "02_piano_raw_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("piano"), "02_raw_notes.json")'),
])

do("instruments/piano/cleaning.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir, get_shared_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "03_piano_cleaned_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("piano"), "03_cleaned_notes.json")'),
    (r'meta_path = os\.path\.join\(get_outputs_dir\(\), "03_piano_clean_meta\.json"\)',
     'meta_path = os.path.join(get_instrument_dir("piano"), "03_clean_meta.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "04_tempo\.json"\)',
     'path = os.path.join(get_shared_dir(), "04_tempo.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "03_piano_cleaned_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("piano"), "03_cleaned_notes.json")'),
])

do("instruments/piano/mapping.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "06_piano_mapped_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("piano"), "06_mapped_notes.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "06_piano_mapped_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("piano"), "06_mapped_notes.json")'),
])

do("instruments/piano/roll.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "08_piano_roll\.png"\)',
     r'out_path = os.path.join(get_instrument_dir("piano"), "08_piano_roll.png")'),
])

do("instruments/piano/visualization.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "10_piano_keyboard\.png"\)',
     r'out_path = os.path.join(get_instrument_dir("piano"), "10_keyboard.png")'),
])

do("instruments/piano/pipeline.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'out_path = os\.path\.join\(get_outputs_dir\(\), "09_piano_preview\.wav"\)',
     'out_path = os.path.join(get_instrument_dir("piano"), "09_preview.wav")'),
])

# ── Vocals ────────────────────────────────────────────────────────────────────
print("== Vocals ==")

do("instruments/vocals/pitch.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "02_vocals_raw_notes\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("vocals"), "02_raw_notes.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "02_vocals_raw_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("vocals"), "02_raw_notes.json")'),
])

do("instruments/vocals/cleaning.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir, get_shared_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)path = os\.path\.join\(get_outputs_dir\(\), "03_vocals_cleaned_notes\.json"\)',
     r'path = os.path.join(get_instrument_dir("vocals"), "03_cleaned_notes.json")'),
    (r'meta = os\.path\.join\(get_outputs_dir\(\), "03_vocals_clean_meta\.json"\)',
     'meta = os.path.join(get_instrument_dir("vocals"), "03_clean_meta.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "04_tempo\.json"\)',
     'path = os.path.join(get_shared_dir(), "04_tempo.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "03_vocals_cleaned_notes\.json"\)',
     'path = os.path.join(get_instrument_dir("vocals"), "03_cleaned_notes.json")'),
])

do("instruments/vocals/visualization.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "10_vocals_contour\.png"\)',
     r'out_path = os.path.join(get_instrument_dir("vocals"), "10_contour.png")'),
])

do("instruments/vocals/pipeline.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'out_path = os\.path\.join\(get_outputs_dir\(\), "09_vocals_preview\.wav"\)',
     'out_path = os.path.join(get_instrument_dir("vocals"), "09_preview.wav")'),
    (r'prev\s*= os\.path\.join\(get_outputs_dir\(\), "09_vocals_preview\.wav"\)',
     'prev  = os.path.join(get_instrument_dir("vocals"), "09_preview.wav")'),
])

# ── Drums ─────────────────────────────────────────────────────────────────────
print("== Drums ==")

do("instruments/drums/onset.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "02_drums_hits\.json"\)',
     r'out_path = os.path.join(get_instrument_dir("drums"), "02_hits.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "02_drums_hits\.json"\)',
     'path = os.path.join(get_instrument_dir("drums"), "02_hits.json")'),
])

do("instruments/drums/notation.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir, get_shared_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "08_drums_grid\.txt"\)',
     r'out_path = os.path.join(get_instrument_dir("drums"), "08_grid.txt")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "04_tempo\.json"\)',
     'path = os.path.join(get_shared_dir(), "04_tempo.json")'),
    (r'path = os\.path\.join\(get_outputs_dir\(\), "08_drums_grid\.txt"\)',
     'path = os.path.join(get_instrument_dir("drums"), "08_grid.txt")'),
])

do("instruments/drums/visualization.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_instrument_dir'),
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)out_path = os\.path\.join\(get_outputs_dir\(\), "10_drums_pattern\.png"\)',
     r'out_path = os.path.join(get_instrument_dir("drums"), "10_pattern.png")'),
])

do("instruments/drums/pipeline.py", [
    (r'from pipeline\.config import get_outputs_dir',
     'from pipeline.config import get_shared_dir'),
    # Two occurrences of tempo.json write in drums/pipeline.py
    (r'os\.makedirs\(get_outputs_dir\(\), exist_ok=True\)\n(\s*)with open\(os\.path\.join\(get_outputs_dir\(\), "04_tempo\.json"\)',
     r'with open(os.path.join(get_shared_dir(), "04_tempo.json")'),
    # Remaining bare get_outputs_dir() for tempo if not caught above
    (r'os\.path\.join\(get_outputs_dir\(\), "04_tempo\.json"\)',
     'os.path.join(get_shared_dir(), "04_tempo.json")'),
])

print("\nAll done!")
