"""
Pipeline Diagnostic Tool
Runs the full pipeline on an audio file and produces a detailed report
showing exactly what happens at every stage — what's kept, what's lost, and why.

Usage:
    python diagnose.py <audio_file>
"""

import sys
import os
import json
import numpy as np

from pipeline.config import set_outputs_dir


def separator(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'=' * pad} {title} {'=' * pad}")
    else:
        print("=" * width)


def run(audio_path: str, no_separate: bool = False):
    out_dir = set_outputs_dir(audio_path)
    separator("PIPELINE DIAGNOSTIC REPORT")
    print(f"File    : {audio_path}")
    print(f"Size    : {os.path.getsize(audio_path) / 1024:.1f} KB")
    print(f"Outputs : {out_dir}")

    # ── Stage 1: Pitch Extraction ─────────────────────────────────────────────
    # ── Stage 0: Instrument Separation ───────────────────────────────────────
    separator("STAGE 0 — Instrument Separation")
    if no_separate:
        pitch_input = audio_path
        print("  Skipped — using raw mix")
    else:
        from pipeline.shared.separation import separate_guitar, get_stem_path
        stem = get_stem_path()
        if os.path.isfile(stem):
            print("  Using saved guitar stem (delete outputs/00_guitar_stem.wav to re-run)")
            pitch_input = stem
        else:
            pitch_input = separate_guitar(audio_path)

    separator("STAGE 1 — Pitch Extraction")
    from pipeline.instruments.guitar.pitch import extract_pitches
    raw_notes = extract_pitches(pitch_input, save=True)

    if raw_notes:
        durations  = [n["duration"]   for n in raw_notes]
        confs      = [n["confidence"] for n in raw_notes]
        pitches    = [n["pitch"]      for n in raw_notes]
        total_dur  = max(n["start"] + n["duration"] for n in raw_notes)

        print(f"  Raw notes detected : {len(raw_notes)}")
        print(f"  Audio duration     : {total_dur:.1f}s")
        print(f"  Note density       : {len(raw_notes)/total_dur:.1f} notes/sec")
        print(f"  Pitch range        : MIDI {min(pitches)} – {max(pitches)}")
        print(f"  Duration  — min {min(durations):.3f}s  avg {np.mean(durations):.3f}s  max {max(durations):.3f}s")
        print(f"  Confidence— min {min(confs):.2f}   avg {np.mean(confs):.2f}   max {max(confs):.2f}")

        # Confidence distribution
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        print("  Confidence distribution:")
        for i, label in enumerate(labels):
            count = sum(1 for c in confs if bins[i] <= c < bins[i+1])
            bar   = "#" * count
            print(f"    {label} | {bar} ({count})")
    else:
        print("  No notes detected.")

    # ── Stage 2: Note Cleaning ────────────────────────────────────────────────
    separator("STAGE 2 — Note Cleaning")
    from pipeline.instruments.guitar.cleaning import clean_notes, MIN_DURATION_S, CONFIDENCE_THRESHOLD, MERGE_GAP_S

    print(f"  Filters applied:")
    print(f"    Min duration  : {MIN_DURATION_S*1000:.0f}ms")
    print(f"    Min confidence: {CONFIDENCE_THRESHOLD}")
    print(f"    Merge gap     : {MERGE_GAP_S*1000:.0f}ms")

    # Track losses at each filter step
    after_duration = [n for n in raw_notes if n["duration"] >= MIN_DURATION_S]
    after_conf     = [n for n in after_duration if n["confidence"] >= CONFIDENCE_THRESHOLD]

    cleaned_notes = clean_notes(raw_notes, save=True)

    print(f"\n  Notes in           : {len(raw_notes)}")
    print(f"  After dur filter   : {len(after_duration)}  (lost {len(raw_notes)-len(after_duration)})")
    print(f"  After conf filter  : {len(after_conf)}  (lost {len(after_duration)-len(after_conf)})")
    print(f"  After merge        : {len(cleaned_notes)}  (merged {len(after_conf)-len(cleaned_notes)})")
    print(f"  Final kept         : {len(cleaned_notes)}  ({100*len(cleaned_notes)/max(len(raw_notes),1):.1f}% of raw)")

    if not cleaned_notes:
        print("\n  WARNING: All notes were filtered out.")
        print("  Consider lowering MIN_DURATION_S or CONFIDENCE_THRESHOLD in note_cleaning.py")
        return

    # ── Stage 2.5: Key Analysis ───────────────────────────────────────────────
    separator("STAGE 2.5 — Key / Scale Analysis")
    from pipeline.shared.music_theory import analyze_key, CHROMATIC, SCALES

    key_info = analyze_key(cleaned_notes, save=True)

    print(f"  Detected key  : {key_info['key_str']}")
    print(f"  Confidence    : {key_info['confidence']:.2f}  {'(strong)' if key_info['confidence']>0.8 else '(moderate)' if key_info['confidence']>0.6 else '(weak — noisy audio?)'}")
    print(f"  Scale notes   : {[CHROMATIC[pc] for pc in key_info['scale_pcs']]}")

    # Show which detected notes fit the scale
    scale_pcs = set(key_info["scale_pcs"])
    in_scale  = [n for n in cleaned_notes if n["pitch"] % 12 in scale_pcs]
    out_scale = [n for n in cleaned_notes if n["pitch"] % 12 not in scale_pcs]
    print(f"  Notes in scale    : {len(in_scale)} / {len(cleaned_notes)} ({100*len(in_scale)/max(len(cleaned_notes),1):.0f}%)")
    if out_scale:
        out_names = list({CHROMATIC[n["pitch"] % 12] for n in out_scale})
        print(f"  Notes outside scale: {out_names}  (likely detection noise or chromatic passing tones)")

    # Top pitch classes by duration
    from collections import defaultdict
    pc_dur = defaultdict(float)
    for n in cleaned_notes:
        pc_dur[n["pitch"] % 12] += n["duration"]
    top = sorted(pc_dur.items(), key=lambda x: -x[1])[:5]
    print(f"  Top pitch classes (by duration): {[(CHROMATIC[pc], round(d,2)) for pc,d in top]}")

    # ── Stage 3: Guitar Mapping ───────────────────────────────────────────────
    separator("STAGE 3 — Guitar Mapping")
    from pipeline.instruments.guitar.mapping import map_to_guitar, HAND_SPAN, MAX_FRET

    mapped_notes = map_to_guitar(cleaned_notes, key_info=key_info, save=True)

    frets   = [n["fret"]   for n in mapped_notes]
    strings = [n["string"] for n in mapped_notes]
    string_names = {1:"e",2:"B",3:"G",4:"D",5:"A",6:"E"}

    print(f"  Mapped notes  : {len(mapped_notes)}")
    print(f"  Fret range    : {min(frets)} – {max(frets)}  (span: {max(frets)-min(frets)} frets)")
    print(f"  Hand span     : {HAND_SPAN} frets (window size)")
    print(f"  Max fret      : {MAX_FRET}")

    # String usage
    print(f"  String usage:")
    for s in range(1, 7):
        count = strings.count(s)
        bar   = "#" * count
        print(f"    String {s} ({string_names[s]}) | {bar} ({count})")

    # Detect large fret jumps (hand shifts)
    shifts = 0
    for i in range(1, len(mapped_notes)):
        if abs(mapped_notes[i]["fret"] - mapped_notes[i-1]["fret"]) > HAND_SPAN:
            shifts += 1
    print(f"  Hand shifts   : {shifts}  (jumps > {HAND_SPAN} frets)")

    # ── Stage 4: Tab Generation ───────────────────────────────────────────────
    separator("STAGE 4 — Tab Generation")
    from pipeline.instruments.guitar.tab import generate_tabs, SECONDS_PER_COLUMN

    tab_str = generate_tabs(mapped_notes, save=True)
    lines   = tab_str.strip().split("\n")

    print(f"  Tab width     : {len(lines[0])} chars")
    print(f"  Time res.     : {SECONDS_PER_COLUMN*1000:.0f}ms per column")
    print(f"  Non-empty strings: {sum(1 for l in lines if l.count('-') < len(l)-4)}/6")

    # ── Summary ───────────────────────────────────────────────────────────────
    separator("SUMMARY")

    print(f"  Audio         : {os.path.basename(audio_path)}")
    print(f"  Key           : {key_info['key_str']}  (confidence {key_info['confidence']:.2f})")
    print(f"  Notes pipeline: {len(raw_notes)} raw -> {len(cleaned_notes)} clean -> {len(mapped_notes)} mapped")
    survival = 100 * len(mapped_notes) / max(len(raw_notes), 1)
    print(f"  Note survival : {survival:.1f}%")

    print(f"\n  Known limitations at this stage:")
    if survival < 20:
        print(f"  [!] Low note survival ({survival:.0f}%) — full mix audio confuses pyin.")
        print(f"      Fix in Phase 2: add Demucs instrument separation before pitch detection.")
    if key_info["confidence"] < 0.7:
        print(f"  [!] Weak key confidence ({key_info['confidence']:.2f}) — may be wrong key.")
        print(f"      Will improve once instrument separation gives a cleaner signal.")
    if len(in_scale) < len(cleaned_notes):
        print(f"  [!] {len(out_scale)} notes outside detected scale — chromatic tones or detection errors.")
    if shifts > len(mapped_notes) * 0.3:
        print(f"  [!] Many hand shifts ({shifts}) — mapping could be smoother with position optimization (Phase 3).")
    if survival >= 20 and key_info["confidence"] >= 0.7:
        print(f"  [OK] Pipeline performing well for this recording type.")

    separator()
    print(f"  Full outputs in: outputs/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose.py <audio_file> [--no-separate]")
        sys.exit(1)
    run(sys.argv[1], no_separate="--no-separate" in sys.argv)
