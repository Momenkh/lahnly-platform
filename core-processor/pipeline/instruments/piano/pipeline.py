"""
Piano / Keyboard Pipeline
==========================
Full end-to-end pipeline for piano/keyboard: stem separation → pitch extraction →
note cleaning → quantization → key analysis → key mapping → piano roll → audio → viz.

Called from main.py when --instrument piano is passed.
"""

import os

from pipeline.shared.stage_executor import StageExecutor, parse_time_range


def run_piano_pipeline(args) -> None:
    from pipeline.config import get_instrument_dir
    executor = StageExecutor(args.from_stage)

    _PIANO_CASCADE = [("htdemucs_6s", "piano"), ("htdemucs_6s", "other")]

    # ── Stage 1: Stem Separation ──────────────────────────────────────────────
    if not args.no_separate and args.from_stage <= 1:
        from pipeline.shared.separation import separate_stem
        pitch_input = separate_stem(args.audio_file, "piano", _PIANO_CASCADE,
                                    instrument_label="piano")
    elif not args.no_separate and args.from_stage <= 2:
        from pipeline.shared.separation import get_stem_path_for, separate_stem
        stem = get_stem_path_for("piano")
        if os.path.isfile(stem):
            print("[Stage 1P] Skipped — using saved piano stem")
            pitch_input = stem
        else:
            print("[Stage 1P] No saved stem found — running separation")
            pitch_input = separate_stem(args.audio_file, "piano", _PIANO_CASCADE,
                                        instrument_label="piano")
    else:
        if args.no_separate:
            print("[Stage 1P] Skipped — using raw mix")
        pitch_input = args.audio_file

    # ── Stage 1.5: Auto-detect piano mode ────────────────────────────────────
    piano_mode = getattr(args, "piano_mode", None)
    if piano_mode is None:
        from pipeline.shared.separation import get_stem_path_for
        from pipeline.shared.auto_detect import detect_piano_mode

        stem_path  = get_stem_path_for("piano")
        detect_src = stem_path if os.path.isfile(stem_path) else pitch_input

        if detect_src and os.path.isfile(detect_src):
            print(f"[Auto P] Detecting piano mode from: {os.path.basename(detect_src)}")
            detected   = detect_piano_mode(detect_src)
            piano_mode = detected["piano_mode"]
            conf       = detected["mode_confidence"]
            f          = detected["features"]
            print(f"[Auto P] polyphony={f['median_polyphony']}  "
                  f"onsets/s={f['onset_density_per_s']}")
            flag = "" if conf >= 0.65 else "  [low confidence — use --piano-mode to override]"
            print(f"[Auto P] Mode: {piano_mode:<15} ({conf:.0%} confidence){flag}")
        else:
            piano_mode = "piano_chord"
            print("[Auto P] No audio source — defaulting to piano_chord")

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Instrument : piano")
    print(f"  Mode       : {piano_mode}")
    print(f"{sep}\n")

    # ── Stage 2: Pitch Extraction ─────────────────────────────────────────────
    if executor.should_run(2):
        from pipeline.instruments.piano.pitch import extract_pitches_piano
        raw_notes = extract_pitches_piano(pitch_input, piano_mode=piano_mode)
    elif executor.should_run(3):
        from pipeline.instruments.piano.pitch import load_raw_notes_piano
        print("[Stage 2P] Skipped — loading saved raw piano notes")
        raw_notes = load_raw_notes_piano()
    # else: raw_notes not needed — stage 3 loads its own saved output

    # ── Stage 3: Note Cleaning ────────────────────────────────────────────────
    from pipeline.instruments.piano.cleaning import clean_notes_piano, load_cleaned_notes_piano
    cleaned_notes = executor.run_or_load(
        3,
        lambda: clean_notes_piano(raw_notes, piano_mode=piano_mode),
        load_cleaned_notes_piano,
        skip_msg="[Stage 3P] Skipped — loading saved cleaned piano notes",
    )

    # ── Stage 4: Tempo Detection & Quantization ───────────────────────────────
    pre_quant_notes = list(cleaned_notes)

    if not args.no_quantize:
        from pipeline.shared.quantization import quantize_notes

        if args.from_stage <= 4:
            bpm_override = getattr(args, "bpm_override", None)
            force_tempo  = getattr(args, "force_tempo",  False)
            reuse_tempo  = None
            if args.from_stage >= 3 and not force_tempo and bpm_override is None:
                try:
                    from pipeline.shared.quantization import load_quantization
                    _, reuse_tempo = load_quantization()
                    if reuse_tempo:
                        print(f"[Stage 4P] Saved BPM ({reuse_tempo['bpm']:.1f}) — reusing")
                except FileNotFoundError:
                    pass

            cleaned_notes, tempo_info = quantize_notes(
                cleaned_notes, pitch_input,
                bpm_override=bpm_override,
                reuse_tempo=reuse_tempo,
            )
        else:
            from pipeline.shared.quantization import load_quantization
            try:
                print("[Stage 4P] Skipped — loading saved quantization")
                cleaned_notes, tempo_info = load_quantization()
            except FileNotFoundError:
                from pipeline.shared.quantization import quantize_notes
                cleaned_notes, tempo_info = quantize_notes(cleaned_notes, pitch_input)
    else:
        print("[Stage 4P] Skipped — quantization disabled")
        tempo_info = None

    # ── Stage 5: Key / Scale Analysis ────────────────────────────────────────
    from pipeline.shared.music_theory import analyze_key, load_key_analysis
    key_info = executor.run_or_load(
        5,
        lambda: analyze_key(cleaned_notes, instrument="piano"),
        load_key_analysis,
        skip_msg="[Stage 5P] Skipped — loading saved key analysis",
    )

    # ── Stage 5b: Key-context feedback ───────────────────────────────────────
    from pipeline.instruments.piano.cleaning import (
        apply_key_octave_correction_piano,
        apply_key_confidence_filter_piano,
    )
    cleaned_notes   = apply_key_octave_correction_piano(cleaned_notes, key_info)
    cleaned_notes   = apply_key_confidence_filter_piano(cleaned_notes, key_info)
    pre_quant_notes = apply_key_octave_correction_piano(pre_quant_notes, key_info)
    pre_quant_notes = apply_key_confidence_filter_piano(pre_quant_notes, key_info)

    # ── Stage 6: Piano Mapping (MIDI → key index + LH/RH split) ──────────────
    from pipeline.instruments.piano.mapping import map_to_piano, load_mapped_notes_piano
    mapped_notes = executor.run_or_load(
        6,
        lambda: map_to_piano(cleaned_notes, key_info=key_info),
        load_mapped_notes_piano,
        skip_msg="[Stage 6P] Skipped — loading saved piano mapped notes",
    )

    # ── Time range filter ─────────────────────────────────────────────────────
    t_start_s, t_end_s = parse_time_range(args)
    if t_start_s is not None or t_end_s is not None:
        before = len(mapped_notes)
        lo = t_start_s or 0.0
        hi = t_end_s or float("inf")
        mapped_notes = [n for n in mapped_notes if lo <= n["start"] <= hi]
        print(f"[Filter P] Time range: {len(mapped_notes)} notes (removed {before - len(mapped_notes)})")

    # ── Stage 8: Piano Roll ───────────────────────────────────────────────────
    if args.from_stage <= 8:
        from pipeline.instruments.piano.roll import generate_piano_roll
        generate_piano_roll(mapped_notes, key_info=key_info)
    else:
        print("[Stage 8P] Skipped")

    # ── Stage 9: Audio Export + Playback ─────────────────────────────────────
    if args.from_stage <= 9:
        import soundfile as sf
        from pipeline.shared.audio import synthesize_notes, play_notes
        from pipeline.settings import AUDIO_SAMPLE_RATE
        from pipeline.instruments.piano.mapping import map_to_piano as _pmap

        audio_mapped = _pmap(pre_quant_notes, key_info=key_info, save=False)
        print(f"[Stage 9P] Synthesizing {len(audio_mapped)} notes...")
        buffer   = synthesize_notes(audio_mapped)
        out_path = os.path.join(get_instrument_dir("piano"), "09_preview.wav")
        sf.write(out_path, buffer, AUDIO_SAMPLE_RATE, subtype="PCM_16")
        print(f"[Stage 9P] Saved -> {out_path}  ({len(buffer)/AUDIO_SAMPLE_RATE:.1f}s)")

        if not getattr(args, "no_play", False):
            play_notes(audio_mapped)
    else:
        print("[Stage 9P] Skipped")

    # ── Stage 10: Piano Keyboard Visualization ────────────────────────────────
    if not getattr(args, "no_viz", False) and args.from_stage <= 10:
        from pipeline.instruments.piano.visualization import generate_piano_keyboard
        img_path = generate_piano_keyboard(mapped_notes, key_info=key_info,
                                            save=True, show=False)
        if img_path:
            print(f"[Stage 10P] Keyboard diagram saved to: {img_path}")
    else:
        print("[Stage 10P] Skipped — visualization disabled")

    print("\nDone. All piano outputs saved to: outputs/")
