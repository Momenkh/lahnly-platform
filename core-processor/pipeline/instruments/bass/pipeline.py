"""
Bass Guitar Pipeline
=====================
Full end-to-end pipeline for bass guitar: stem separation → pitch extraction →
note cleaning → quantization → key analysis → fretboard mapping → tab → audio → viz.

Called from main.py when --instrument bass is passed.
All guitar code in main.py is left completely untouched.
"""

import os

from pipeline.shared.stage_executor import StageExecutor, parse_time_range


def run_bass_pipeline(args) -> None:
    """
    Execute all pipeline stages for bass guitar.
    args is the parsed argparse namespace from main.py (audio_file, flags, etc.).
    """
    from pipeline.config import get_instrument_dir
    executor = StageExecutor(args.from_stage)

    # Bass model cascade: htdemucs_6s has a dedicated bass stem.
    _BASS_CASCADE = [("htdemucs_6s", "bass"), ("htdemucs", "bass")]

    # ── Stage 1: Stem Separation ──────────────────────────────────────────────
    if not args.no_separate and args.from_stage <= 1:
        from pipeline.shared.separation import separate_stem
        pitch_input = separate_stem(args.audio_file, "bass", _BASS_CASCADE,
                                    instrument_label="bass")
    elif not args.no_separate and args.from_stage <= 2:
        from pipeline.shared.separation import get_stem_path_for, separate_stem
        stem = get_stem_path_for("bass")
        if os.path.isfile(stem):
            print("[Stage 1B] Skipped — using saved bass stem")
            pitch_input = stem
        else:
            print("[Stage 1B] No saved stem found — running separation")
            pitch_input = separate_stem(args.audio_file, "bass", _BASS_CASCADE,
                                        instrument_label="bass")
    else:
        if args.no_separate:
            print("[Stage 1B] Skipped — using raw mix for pitch detection")
        pitch_input = args.audio_file

    # ── Stage 1.5: Auto-detect bass style ────────────────────────────────────
    bass_style = getattr(args, "bass_style", None)
    if bass_style is None:
        from pipeline.shared.separation import get_stem_path_for
        from pipeline.shared.auto_detect import detect_bass_mode

        stem_path  = get_stem_path_for("bass")
        detect_src = stem_path if os.path.isfile(stem_path) else pitch_input

        if detect_src and os.path.isfile(detect_src):
            print(f"[Auto B] Detecting bass style from: {os.path.basename(detect_src)}")
            detected   = detect_bass_mode(detect_src)
            bass_style = detected["bass_style"]
            conf       = detected["style_confidence"]
            f          = detected["features"]
            print(f"[Auto B] flatness={f['spectral_flatness']}  "
                  f"onset_str={f['mean_onset_strength']}")
            flag = "" if conf >= 0.65 else "  [low confidence — use --bass-style to override]"
            print(f"[Auto B] Style: {bass_style:<15} ({conf:.0%} confidence){flag}")
        else:
            bass_style = "bass_fingered"
            print("[Auto B] No audio source — defaulting to bass_fingered")

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Instrument : bass")
    print(f"  Style      : {bass_style}")
    print(f"{sep}\n")

    # ── Stage 2: Pitch Extraction ─────────────────────────────────────────────
    if executor.should_run(2):
        from pipeline.instruments.bass.pitch import extract_pitches_bass
        raw_notes = extract_pitches_bass(pitch_input, bass_style=bass_style)
    elif executor.should_run(3):
        from pipeline.instruments.bass.pitch import load_raw_notes_bass
        print("[Stage 2B] Skipped — loading saved raw bass notes")
        raw_notes = load_raw_notes_bass()
    # else: raw_notes not needed — stage 3 loads its own saved output

    # ── Stage 3: Note Cleaning ────────────────────────────────────────────────
    from pipeline.instruments.bass.cleaning import clean_notes_bass, load_cleaned_notes_bass
    cleaned_notes = executor.run_or_load(
        3,
        lambda: clean_notes_bass(raw_notes, bass_style=bass_style),
        load_cleaned_notes_bass,
        skip_msg="[Stage 3B] Skipped — loading saved cleaned bass notes",
    )

    # ── Stage 4: Tempo Detection & Quantization ───────────────────────────────
    pre_quant_notes = list(cleaned_notes)

    if not args.no_quantize:
        from pipeline.shared.quantization import quantize_notes

        if args.from_stage <= 4:
            bpm_override = getattr(args, "bpm_override", None)
            force_tempo  = getattr(args, "force_tempo",  False)

            reuse_tempo = None
            if args.from_stage >= 3 and not force_tempo and bpm_override is None:
                try:
                    from pipeline.shared.quantization import load_quantization
                    _, reuse_tempo = load_quantization()
                    if reuse_tempo:
                        print(f"[Stage 4B] Saved BPM ({reuse_tempo['bpm']:.1f}) — reusing")
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
                print("[Stage 4B] Skipped — loading saved quantization")
                cleaned_notes, tempo_info = load_quantization()
            except FileNotFoundError:
                from pipeline.shared.quantization import quantize_notes
                print("[Stage 4B] No saved quantization — running now")
                cleaned_notes, tempo_info = quantize_notes(cleaned_notes, pitch_input)
    else:
        print("[Stage 4B] Skipped — quantization disabled")
        tempo_info = None

    # ── Stage 5: Key / Scale Analysis ────────────────────────────────────────
    from pipeline.shared.music_theory import analyze_key, load_key_analysis
    key_info = executor.run_or_load(
        5,
        lambda: analyze_key(cleaned_notes, instrument="bass"),
        load_key_analysis,
        skip_msg="[Stage 5B] Skipped — loading saved key analysis",
    )

    # ── Stage 5b: Key-context feedback ───────────────────────────────────────
    from pipeline.instruments.bass.cleaning import (
        apply_key_octave_correction_bass,
        apply_key_confidence_filter_bass,
    )
    cleaned_notes   = apply_key_octave_correction_bass(cleaned_notes, key_info)
    cleaned_notes   = apply_key_confidence_filter_bass(cleaned_notes, key_info)
    pre_quant_notes = apply_key_octave_correction_bass(pre_quant_notes, key_info)
    pre_quant_notes = apply_key_confidence_filter_bass(pre_quant_notes, key_info)

    # ── Stage 6: Bass Mapping ─────────────────────────────────────────────────
    from pipeline.instruments.bass.mapping import map_to_bass, load_mapped_notes_bass
    mapped_notes = executor.run_or_load(
        6,
        lambda: map_to_bass(cleaned_notes, key_info=key_info),
        load_mapped_notes_bass,
        skip_msg="[Stage 6B] Skipped — loading saved bass mapped notes",
    )

    # ── Time range filter ─────────────────────────────────────────────────────
    t_start_s, t_end_s = parse_time_range(args)
    if t_start_s is not None or t_end_s is not None:
        before = len(mapped_notes)
        lo = t_start_s or 0.0
        hi = t_end_s or float("inf")
        mapped_notes = [n for n in mapped_notes if lo <= n["start"] <= hi]
        print(f"[Filter B] Time range: {len(mapped_notes)} notes "
              f"(removed {before - len(mapped_notes)})")

    # ── Stage 8: Tab Generation ───────────────────────────────────────────────
    if args.from_stage <= 8:
        from pipeline.instruments.bass.tab import generate_tabs_bass
        generate_tabs_bass(mapped_notes, tempo_info=tempo_info)
    else:
        print("[Stage 8B] Skipped")

    # ── Stage 9: Audio Export + Playback ─────────────────────────────────────
    if args.from_stage <= 9:
        import soundfile as sf
        from pipeline.shared.audio import synthesize_notes, play_notes
        from pipeline.settings import AUDIO_SAMPLE_RATE
        from pipeline.instruments.bass.mapping import map_to_bass as _bmap

        audio_mapped = _bmap(pre_quant_notes, key_info=key_info, save=False)
        print(f"[Stage 9B] Synthesizing {len(audio_mapped)} notes...")
        buffer    = synthesize_notes(audio_mapped)
        out_path  = os.path.join(get_instrument_dir("bass"), "09_preview.wav")
        sf.write(out_path, buffer, AUDIO_SAMPLE_RATE, subtype="PCM_16")
        print(f"[Stage 9B] Saved -> {out_path}  ({len(buffer)/AUDIO_SAMPLE_RATE:.1f}s)")

        if not getattr(args, "no_play", False):
            play_notes(audio_mapped)
    else:
        print("[Stage 9B] Skipped")

    # ── Stage 10: Fretboard Visualization ────────────────────────────────────
    if not getattr(args, "no_viz", False) and args.from_stage <= 10:
        from pipeline.instruments.bass.visualization import plot_fretboard_bass
        img_path = plot_fretboard_bass(mapped_notes, key_info=key_info,
                                        save=True, show=False)
        if img_path:
            print(f"[Stage 10B] Fretboard diagram saved to: {img_path}")
    else:
        print("[Stage 10B] Skipped — visualization disabled")

    print("\nDone. All bass outputs saved to: outputs/")
