"""
AI Music Transcription Platform
End-to-end pipeline: audio file -> guitar tabs + playback + visualization

Usage:
    python main.py <audio_file> [options]

Pipeline stages:
    1  Instrument separation  (Demucs — extracts guitar stem)
    2  Pitch extraction       (basic-pitch polyphonic ML model)
    3  Note cleaning          (filters, merges, polyphony limit)
    4  Quantization           (tempo detection, snap to 16th-note grid)
    5  Key / scale analysis   (Krumhansl-Schmuckler)
    5b Key-context octave correction
    6  Guitar mapping         (string + fret assignment)
    7  Chord detection        (groups simultaneous notes, names chords)
    8  Tab generation         (ASCII guitar tab — solo notes only)
    9  Audio preview          (synthesized WAV + optional playback)
   10  Fretboard diagram      (PNG visualization)
   11  Chord sheet            (PNG chord box diagrams + progression)

Options:
    --type T      acoustic | clean | distorted  (default: clean)
                  Tonal character of the guitar.
                  acoustic   = nylon/steel string — middle-ground thresholds
                  clean      = clean electric — default electric settings
                  distorted  = distorted electric — HPSS pre-filter + tighter thresholds
    --role R      lead | rhythm  (default: rhythm)
                  Playing style.
                  lead       = single-note solo — lower polyphony, melody isolation
                  rhythm     = chords / rhythm parts — full polyphony, all notes used
    --start MM:SS     Only process notes from this time onward
    --end   MM:SS     Only process notes up to this time
    --bpm-override N  Force BPM to N instead of auto-detecting
    --force-tempo     Re-detect BPM even if a saved tempo exists
    --score           Compute chroma similarity score (stem vs preview) and print it
    --no-play         Skip audio playback (still saves 09_preview.wav)
    --no-viz          Skip fretboard and chord sheet images
    --no-separate     Skip separation (use raw mix for pitch detection)
    --no-quantize     Skip tempo detection and note quantization
    --from-stage N    Resume from stage N (2-11); requires prior run outputs
"""

import argparse
import os
import sys

from pipeline.config import set_outputs_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Guitar tab transcription pipeline")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (WAV, FLAC, M4A, MP3)")
    parser.add_argument("--score",         action="store_true", help="Compute chroma similarity score (stem vs preview) after processing")
    parser.add_argument("--no-play",      action="store_true", help="Skip audio playback")
    parser.add_argument("--no-viz",       action="store_true", help="Skip visualization")
    parser.add_argument("--no-separate",  action="store_true", help="Skip separation (use raw mix)")
    parser.add_argument("--no-quantize",  action="store_true", help="Skip tempo detection and quantization")
    parser.add_argument("--force-tempo",  action="store_true", help="Re-detect BPM even if saved tempo exists")
    parser.add_argument(
        "--type",
        choices=["acoustic", "clean", "distorted"],
        default=None,
        dest="guitar_type",
        help="Guitar type: acoustic, clean, distorted (default: auto-detected from stem)",
    )
    parser.add_argument(
        "--role",
        choices=["lead", "rhythm"],
        default=None,
        dest="guitar_role",
        help="Guitar role: lead or rhythm (default: auto-detected from stem)",
    )
    parser.add_argument(
        "--bpm-override",
        type=float,
        default=None,
        metavar="BPM",
        help="Force a specific BPM instead of auto-detecting",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="MM:SS",
        help="Only process notes starting from this timestamp (e.g. 5:25 or 325)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="MM:SS",
        help="Only process notes up to this timestamp (e.g. 6:10 or 370)",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=2,
        metavar="N",
        help="Resume from stage N (2-11). Requires prior run to have saved outputs.",
    )
    return parser.parse_args()


def _parse_time(s: str) -> float:
    """Parse MM:SS or plain seconds string to float seconds."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)


def _load_saved_tempo() -> dict | None:
    """Load 04_tempo.json if it exists, return None otherwise."""
    from pipeline.quantization import load_quantization
    try:
        _, tempo = load_quantization()
        return tempo
    except FileNotFoundError:
        return None


def main():
    args = parse_args()

    if args.audio_file:
        out_dir = set_outputs_dir(args.audio_file)
        print(f"Outputs -> {out_dir}")

    if args.from_stage <= 1 and not args.audio_file:
        print("Error: audio_file is required unless --from-stage > 1")
        sys.exit(1)

    if args.audio_file and not os.path.isfile(args.audio_file):
        print(f"Error: file not found: {args.audio_file}")
        sys.exit(1)

    # ── Stage 1: Instrument Separation ───────────────────────────────────────
    if not args.no_separate and args.from_stage <= 1:
        from pipeline.separation import separate_guitar
        pitch_input = separate_guitar(args.audio_file)
    elif not args.no_separate and args.from_stage <= 2:
        from pipeline.separation import get_stem_path
        stem = get_stem_path()
        if os.path.isfile(stem):
            print("[Stage 1] Skipped — using saved guitar stem")
            pitch_input = stem
        else:
            from pipeline.separation import separate_guitar
            print("[Stage 1] No saved stem found — running separation")
            pitch_input = separate_guitar(args.audio_file)
    else:
        if args.no_separate:
            print("[Stage 1] Skipped — using raw mix for pitch detection")
        pitch_input = args.audio_file

    # ── Auto-detect guitar type + role (when not explicitly specified) ───────
    if args.guitar_type is None or args.guitar_role is None:
        from pipeline.separation import get_stem_path
        from pipeline.auto_detect import detect_guitar_mode

        # Prefer the separated stem for detection (more reliable than raw mix).
        # Fall back to the raw audio only if the stem doesn't exist.
        stem_path  = get_stem_path()
        detect_src = stem_path if os.path.isfile(stem_path) else None
        if detect_src is None and pitch_input and os.path.isfile(pitch_input):
            detect_src = pitch_input

        if detect_src and os.path.isfile(detect_src):
            print(f"[Auto] Detecting guitar type + role from: {os.path.basename(detect_src)}")
            detected = detect_guitar_mode(detect_src)
            f = detected["features"]
            print(f"[Auto] flatness={f['spectral_flatness']}  "
                  f"contrast={f['spectral_contrast']}  "
                  f"centroid={f['spectral_centroid_hz']:.0f}Hz  "
                  f"polyphony={f['median_polyphony']}  "
                  f"onsets/s={f['onset_density_per_s']}")

            if args.guitar_type is None:
                args.guitar_type = detected["guitar_type"]
                conf = detected["type_confidence"]
                flag = "" if conf >= 0.70 else "  [low confidence — use --type to override]"
                print(f"[Auto] Type  : {args.guitar_type:<10} ({conf:.0%} confidence){flag}")

            if args.guitar_role is None:
                args.guitar_role = detected["guitar_role"]
                conf = detected["role_confidence"]
                flag = "" if conf >= 0.70 else "  [low confidence — use --role to override]"
                print(f"[Auto] Role  : {args.guitar_role:<10} ({conf:.0%} confidence){flag}")
        else:
            # No audio source for detection — try saved clean meta, then defaults
            if args.from_stage >= 3:
                from pipeline.note_cleaning import load_clean_meta
                meta = load_clean_meta()
                if args.guitar_type is None:
                    args.guitar_type = meta.get("guitar_type", "clean")
                if args.guitar_role is None:
                    args.guitar_role = meta.get("guitar_role", "rhythm")
                print(f"[Auto] Using saved mode: type={args.guitar_type} role={args.guitar_role}")
            else:
                if args.guitar_type is None:
                    args.guitar_type = "clean"
                if args.guitar_role is None:
                    args.guitar_role = "rhythm"
                print(f"[Auto] No audio source for detection — defaulting to "
                      f"type={args.guitar_type} role={args.guitar_role}")

    # ── Mode summary ─────────────────────────────────────────────────────────
    _TYPE_DESC = {
        "acoustic":  "acoustic guitar (nylon/steel string)",
        "clean":     "clean electric guitar",
        "distorted": "distorted electric guitar",
    }
    _ROLE_DESC = {
        "lead":   "lead / solo (single-note, melody isolation on)",
        "rhythm": "rhythm / chords (full polyphony, all notes used)",
    }
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Mode : {args.guitar_type}_{args.guitar_role}")
    print(f"  Type : {_TYPE_DESC.get(args.guitar_type, args.guitar_type)}")
    print(f"  Role : {_ROLE_DESC.get(args.guitar_role, args.guitar_role)}")
    print(f"{sep}\n")

    # ── Stage 2: Pitch Extraction ─────────────────────────────────────────────
    if args.from_stage <= 2:
        from pipeline.pitch_extraction import extract_pitches
        raw_notes = extract_pitches(pitch_input, guitar_type=args.guitar_type, guitar_role=args.guitar_role)
    elif args.from_stage <= 3:
        from pipeline.pitch_extraction import load_raw_notes
        print("[Stage 2] Skipped — loading saved raw notes")
        raw_notes = load_raw_notes()
    # else: raw_notes not needed — Stage 3 loads its own saved output

    # ── Stage 3: Note Cleaning ────────────────────────────────────────────────
    if args.from_stage <= 3:
        from pipeline.note_cleaning import clean_notes
        cleaned_notes = clean_notes(raw_notes, guitar_type=args.guitar_type, guitar_role=args.guitar_role)
    else:
        from pipeline.note_cleaning import load_cleaned_notes, load_clean_meta
        print("[Stage 3] Skipped — loading saved cleaned notes")
        cleaned_notes = load_cleaned_notes()

        # Mode mismatch warning
        meta = load_clean_meta()
        saved_mode = meta.get("mode")
        cur_mode   = f"{args.guitar_type}_{args.guitar_role}"
        if saved_mode and saved_mode != cur_mode:
            print(f"[Warning] Saved cleaned notes used mode={saved_mode} "
                  f"but current run specifies '{cur_mode}'.")
            print(f"[Warning] Results may be inconsistent — re-run with --from-stage 3 "
                  f"to re-clean with '{cur_mode}' settings.")

    # ── Stage 4: Tempo Detection & Quantization ───────────────────────────────
    # Save pre-quantization notes for synthesis — quantization snaps timing to a
    # BPM grid which can drift when the original recording has rubato or unstable
    # tempo.  The synthesized preview uses the original timing so its chroma
    # matches the stem more accurately; tabs use the quantized timing.
    pre_quant_notes = list(cleaned_notes)

    if not args.no_quantize:
        from pipeline.quantization import quantize_notes

        if args.from_stage <= 4:
            # Decide whether to reuse a saved BPM
            reuse_tempo = None
            if args.from_stage >= 3 and not args.force_tempo and args.bpm_override is None:
                reuse_tempo = _load_saved_tempo()
                if reuse_tempo:
                    print(f"[Stage 4] Saved BPM found ({reuse_tempo['bpm']:.1f}) — "
                          f"reusing (--force-tempo to override)")

            cleaned_notes, tempo_info = quantize_notes(
                cleaned_notes, pitch_input,
                bpm_override=args.bpm_override,
                reuse_tempo=reuse_tempo,
            )
        else:
            from pipeline.quantization import load_quantization
            try:
                print("[Stage 4] Skipped — loading saved quantization")
                cleaned_notes, tempo_info = load_quantization()
            except FileNotFoundError:
                print("[Stage 4] No saved quantization — running now")
                cleaned_notes, tempo_info = quantize_notes(
                    cleaned_notes, pitch_input,
                    bpm_override=args.bpm_override,
                )
    else:
        print("[Stage 4] Skipped — quantization disabled")
        tempo_info = None

    # ── Stage 5: Key / Scale Analysis ────────────────────────────────────────
    if args.from_stage <= 5:
        from pipeline.music_theory import analyze_key
        key_info = analyze_key(cleaned_notes)
    else:
        from pipeline.music_theory import load_key_analysis
        try:
            print("[Stage 5] Skipped — loading saved key analysis")
            key_info = load_key_analysis()
        except FileNotFoundError:
            from pipeline.music_theory import analyze_key
            print("[Stage 5] No saved key analysis — running now")
            key_info = analyze_key(cleaned_notes)

    # ── Stage 5b: Key-context feedback (octave correction + confidence filter) ──
    from pipeline.note_cleaning import apply_key_octave_correction, apply_key_confidence_filter
    from pipeline.settings import CLEANING_KEY_CONFIDENCE_CUTOFF
    cleaned_notes = apply_key_octave_correction(cleaned_notes, key_info)
    cleaned_notes = apply_key_confidence_filter(cleaned_notes, key_info,
                                                conf_cutoff=CLEANING_KEY_CONFIDENCE_CUTOFF)
    # Apply the same corrections to the pre-quantization copy so the audio
    # synthesis path gets accurate pitch and the same note set.
    pre_quant_notes = apply_key_octave_correction(pre_quant_notes, key_info)
    pre_quant_notes = apply_key_confidence_filter(pre_quant_notes, key_info,
                                                  conf_cutoff=CLEANING_KEY_CONFIDENCE_CUTOFF)

    # ── Stage 6: Guitar Mapping ───────────────────────────────────────────────
    if args.from_stage <= 6:
        from pipeline.guitar_mapping import map_to_guitar
        mapped_notes = map_to_guitar(cleaned_notes, key_info=key_info,
                                     guitar_type=args.guitar_type, guitar_role=args.guitar_role)
    else:
        from pipeline.guitar_mapping import load_mapped_notes
        print("[Stage 6] Skipped — loading saved mapped notes")
        mapped_notes = load_mapped_notes()

    # ── Time range filter (applied to mapped notes so all downstream is scoped) ─
    t_start_s = _parse_time(args.start) if args.start else None
    t_end_s   = _parse_time(args.end)   if args.end   else None

    if t_start_s is not None or t_end_s is not None:
        before = len(mapped_notes)
        lo = t_start_s or 0.0
        hi = t_end_s or float("inf")
        mapped_notes = [n for n in mapped_notes if lo <= n["start"] <= hi]
        label = f"{args.start or '0:00'} - {args.end or 'end'}"
        print(f"[Filter] Time range {label}: {len(mapped_notes)} notes "
              f"(removed {before - len(mapped_notes)})")

    # ── Melody isolation ──────────────────────────────────────────────────────
    # For lead role: isolate the top voice for tabs and audio preview.
    # For rhythm role: use all mapped notes so chord columns are rendered.
    from pipeline.guitar_mapping import isolate_melody, map_to_guitar as _map
    from pipeline.settings import MELODY_MIN_PITCH
    melody_notes, harmony_notes = isolate_melody(
        mapped_notes,
        min_pitch=MELODY_MIN_PITCH.get(args.guitar_role, 0),
    )

    tab_notes = melody_notes if args.guitar_role == "lead" else mapped_notes

    # Audio synthesis uses pre-quantization timing so the chroma matches the stem
    # without grid-snapping drift.  Key context (key_info) is still applied via
    # the 5b filters already run on pre_quant_notes.
    audio_mapped = _map(pre_quant_notes, key_info=key_info,
                        guitar_type=args.guitar_type, guitar_role=args.guitar_role,
                        save=False)
    audio_melody, _ = isolate_melody(audio_mapped,
                                     min_pitch=MELODY_MIN_PITCH.get(args.guitar_role, 0))
    audio_notes = audio_melody if args.guitar_role == "lead" else audio_mapped

    # ── Stage 7: Chord Detection ──────────────────────────────────────────────
    # harmony_notes excludes fast single-voice runs so they don't trigger
    # false chord groups. For lead role this is the melody complement;
    # for rhythm role pass all mapped notes (harmony_notes ≈ mapped_notes).
    chord_input = harmony_notes if args.guitar_role == "lead" else mapped_notes
    if args.from_stage <= 7:
        from pipeline.chord_detection import detect_chords
        _, chord_groups = detect_chords(
            chord_input, tempo_info=tempo_info, key_info=key_info
        )
    else:
        from pipeline.chord_detection import load_chord_detection
        try:
            print("[Stage 7] Skipped — loading saved chord detection")
            _, chord_groups = load_chord_detection()
        except FileNotFoundError:
            from pipeline.chord_detection import detect_chords
            print("[Stage 7] No saved chord data — running now")
            _, chord_groups = detect_chords(
                chord_input, tempo_info=tempo_info, key_info=key_info
            )

    # ── Stage 8: Tab Generation ───────────────────────────────────────────────
    # lead: melody only (single-voice); rhythm/acoustic: all mapped notes (chord columns)
    if args.from_stage <= 8:
        from pipeline.tab_generation import generate_tabs
        tab_str = generate_tabs(tab_notes, chord_groups=chord_groups, tempo_info=tempo_info)
    else:
        from pipeline.tab_generation import load_tabs
        print("[Stage 8] Skipped — loading saved tabs")
        tab_str = load_tabs()

    # ── Stage 9: Audio Export + Playback ─────────────────────────────────────
    if args.from_stage <= 9:
        from pipeline.audio_playback import save_audio, play_notes
        save_audio(audio_notes)
        if not args.no_play:
            play_notes(audio_notes)
    else:
        print("[Stage 9] Skipped")

    # ── Score: Chroma Similarity ──────────────────────────────────────────────
    if args.score:
        from pipeline.evaluation import score_transcription
        from pipeline.separation import get_stem_path
        from pipeline.config import get_outputs_dir

        stem_path    = get_stem_path()
        preview_path = os.path.join(get_outputs_dir(), "09_preview.wav")

        if os.path.isfile(stem_path) and os.path.isfile(preview_path):
            sim = score_transcription(stem_path, preview_path)
            print(f"\n[Score] Chroma similarity (stem vs preview): {sim:.3f}"
                  f"  (0.55 = poor  |  0.70 = usable  |  0.85 = very good)")
        else:
            missing = stem_path if not os.path.isfile(stem_path) else preview_path
            print(f"[Score] Skipped — file not found: {missing}")

    # ── Stage 10: Fretboard Visualization ────────────────────────────────────
    if not args.no_viz and args.from_stage <= 10:
        from pipeline.visualization import plot_fretboard
        img_path = plot_fretboard(
            mapped_notes, key_info=key_info,
            save=True, show=False,
        )
        if img_path:
            print(f"[Stage 10] Fretboard diagram saved to: {img_path}")
    else:
        print("[Stage 10] Skipped — visualization disabled")

    # ── Stage 11: Chord Sheet ─────────────────────────────────────────────────
    if not args.no_viz and args.from_stage <= 11:
        from pipeline.chord_sheet import plot_chord_sheet
        chord_path = plot_chord_sheet(
            chord_groups, tempo_info=tempo_info, save=True, show=False
        )
        if chord_path:
            print(f"[Stage 11] Chord sheet saved to: {chord_path}")
    else:
        print("[Stage 11] Skipped — visualization disabled")

    print("\nDone. All outputs saved to: outputs/")


if __name__ == "__main__":
    main()
