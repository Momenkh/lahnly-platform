"""
Vocals Pipeline
================
Full end-to-end pipeline for vocals: stem separation → pitch extraction (CREPE) →
note cleaning → quantization → key analysis → lead-sheet visualization.

No fretboard mapping stage (vocals have no fingering).
Called from main.py when --instrument vocals is passed.
"""

import os

from pipeline.shared.stage_executor import StageExecutor, parse_time_range


def run_vocals_pipeline(args) -> None:
    from pipeline.config import get_instrument_dir
    executor = StageExecutor(args.from_stage)

    _VOCALS_CASCADE = [("htdemucs_6s", "vocals"), ("htdemucs", "vocals")]

    # ── Stage 1: Stem Separation ──────────────────────────────────────────────
    if not args.no_separate and args.from_stage <= 1:
        from pipeline.shared.separation import separate_stem
        pitch_input = separate_stem(args.audio_file, "vocals", _VOCALS_CASCADE,
                                    instrument_label="vocals")
    elif not args.no_separate and args.from_stage <= 2:
        from pipeline.shared.separation import get_stem_path_for, separate_stem
        stem = get_stem_path_for("vocals")
        if os.path.isfile(stem):
            print("[Stage 1V] Skipped — using saved vocals stem")
            pitch_input = stem
        else:
            print("[Stage 1V] No saved stem found — running separation")
            pitch_input = separate_stem(args.audio_file, "vocals", _VOCALS_CASCADE,
                                        instrument_label="vocals")
    else:
        if args.no_separate:
            print("[Stage 1V] Skipped — using raw mix for pitch detection")
        pitch_input = args.audio_file

    # ── Stage 1.5: Auto-detect vocals mode ───────────────────────────────────
    vocals_mode = getattr(args, "vocals_mode", None)
    if vocals_mode is None:
        from pipeline.shared.separation import get_stem_path_for
        from pipeline.shared.auto_detect import detect_vocals_mode

        stem_path  = get_stem_path_for("vocals")
        detect_src = stem_path if os.path.isfile(stem_path) else pitch_input

        if detect_src and os.path.isfile(detect_src):
            print(f"[Auto V] Detecting vocals mode from: {os.path.basename(detect_src)}")
            detected    = detect_vocals_mode(detect_src)
            vocals_mode = detected["vocals_mode"]
            conf        = detected["mode_confidence"]
            f           = detected["features"]
            print(f"[Auto V] rms_ratio={f['rms_ratio']}  "
                  f"flatness={f['spectral_flatness']}")
            flag = "" if conf >= 0.65 else "  [low confidence]"
            print(f"[Auto V] Mode: {vocals_mode:<20} ({conf:.0%} confidence){flag}")
        else:
            vocals_mode = "vocals_lead"
            print("[Auto V] No audio source — defaulting to vocals_lead")

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Instrument : vocals")
    print(f"  Mode       : {vocals_mode}")
    print(f"{sep}\n")

    # ── Stage 2: Pitch Extraction (CREPE primary) ─────────────────────────────
    use_repet = getattr(args, "vocals_repet", False)
    from pipeline.instruments.vocals.pitch import extract_pitches_vocals, load_raw_notes_vocals
    raw_notes = executor.run_or_load(
        2,
        lambda: extract_pitches_vocals(pitch_input, vocals_mode=vocals_mode,
                                       use_repet=use_repet),
        load_raw_notes_vocals,
        skip_msg="[Stage 2V] Skipped — loading saved raw vocals notes",
    )

    # ── Stage 3: Note Cleaning ────────────────────────────────────────────────
    from pipeline.instruments.vocals.cleaning import clean_notes_vocals, load_cleaned_notes_vocals
    cleaned_notes = executor.run_or_load(
        3,
        lambda: clean_notes_vocals(raw_notes, vocals_mode=vocals_mode),
        load_cleaned_notes_vocals,
        skip_msg="[Stage 3V] Skipped — loading saved cleaned vocals notes",
    )

    # ── Velocity Estimation (between Stage 3 and Stage 4) ────────────────────
    if os.path.isfile(pitch_input):
        from pipeline.shared.spectral import compute_velocity
        cleaned_notes = compute_velocity(cleaned_notes, pitch_input)
        print(f"[Velocity] Estimated velocity for {len(cleaned_notes)} vocals notes")

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
                        print(f"[Stage 4V] Saved BPM ({reuse_tempo['bpm']:.1f}) — reusing")
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
                print("[Stage 4V] Skipped — loading saved quantization")
                cleaned_notes, tempo_info = load_quantization()
            except FileNotFoundError:
                from pipeline.shared.quantization import quantize_notes
                print("[Stage 4V] No saved quantization — running now")
                cleaned_notes, tempo_info = quantize_notes(cleaned_notes, pitch_input)
    else:
        print("[Stage 4V] Skipped — quantization disabled")
        tempo_info = None

    # ── Stage 5: Key / Scale Analysis ────────────────────────────────────────
    from pipeline.shared.music_theory import analyze_key, load_key_analysis
    key_info = executor.run_or_load(
        5,
        lambda: analyze_key(cleaned_notes, instrument="vocals"),
        load_key_analysis,
        skip_msg="[Stage 5V] Skipped — loading saved key analysis",
    )

    # ── Stage 5b: Key-context feedback ───────────────────────────────────────
    from pipeline.instruments.vocals.cleaning import (
        apply_key_octave_correction_vocals,
        apply_key_confidence_filter_vocals,
    )
    cleaned_notes   = apply_key_octave_correction_vocals(cleaned_notes, key_info)
    cleaned_notes   = apply_key_confidence_filter_vocals(cleaned_notes, key_info)
    pre_quant_notes = apply_key_octave_correction_vocals(pre_quant_notes, key_info)
    pre_quant_notes = apply_key_confidence_filter_vocals(pre_quant_notes, key_info)

    # ── Time range filter ─────────────────────────────────────────────────────
    t_start_s, t_end_s = parse_time_range(args)
    if t_start_s is not None or t_end_s is not None:
        before = len(cleaned_notes)
        lo = t_start_s or 0.0
        hi = t_end_s or float("inf")
        cleaned_notes   = [n for n in cleaned_notes   if lo <= n["start"] <= hi]
        pre_quant_notes = [n for n in pre_quant_notes if lo <= n["start"] <= hi]
        print(f"[Filter V] Time range: {len(cleaned_notes)} notes "
              f"(removed {before - len(cleaned_notes)})")

    # ── Stage 9: Audio Export + Playback ─────────────────────────────────────
    stem_path_for_viz = None
    if args.from_stage <= 9:
        import soundfile as sf
        from pipeline.shared.audio import synthesize_notes, play_notes
        from pipeline.settings import AUDIO_SAMPLE_RATE
        from pipeline.shared.separation import get_stem_path_for

        print(f"[Stage 9V] Synthesizing {len(pre_quant_notes)} notes...")
        buffer   = synthesize_notes(pre_quant_notes)
        out_path = os.path.join(get_instrument_dir("vocals"), "09_preview.wav")
        sf.write(out_path, buffer, AUDIO_SAMPLE_RATE, subtype="PCM_16")
        print(f"[Stage 9V] Saved -> {out_path}  ({len(buffer)/AUDIO_SAMPLE_RATE:.1f}s)")

        sp = get_stem_path_for("vocals")
        stem_path_for_viz = sp if os.path.isfile(sp) else None

        if not getattr(args, "no_play", False):
            play_notes(pre_quant_notes)
    else:
        print("[Stage 9V] Skipped")

    # ── Stage 10: Vocals Contour Visualization ────────────────────────────────
    if not getattr(args, "no_viz", False) and args.from_stage <= 10:
        from pipeline.instruments.vocals.visualization import plot_vocals_contour
        from pipeline.shared.separation import get_stem_path_for

        sp = stem_path_for_viz or get_stem_path_for("vocals")
        img_path = plot_vocals_contour(
            cleaned_notes,
            stem_path=sp if os.path.isfile(sp) else None,
            key_info=key_info,
            save=True,
            show=False,
        )
        if img_path:
            print(f"[Stage 10V] Contour saved to: {img_path}")
    else:
        print("[Stage 10V] Skipped — visualization disabled")

    # ── Scoring ───────────────────────────────────────────────────────────────
    if getattr(args, "score", False):
        from pipeline.evaluation import score_transcription
        from pipeline.shared.separation import get_stem_path_for

        stem  = get_stem_path_for("vocals")
        prev  = os.path.join(get_instrument_dir("vocals"), "09_preview.wav")
        if os.path.isfile(stem) and os.path.isfile(prev):
            s = score_transcription(stem, prev)
            print(f"\n[Score V] Chroma similarity: {s:.3f}  ({s*100:.1f}%)")
        else:
            print("[Score V] Could not compute score — stem or preview missing")

    executor.print_summary("vocals")
    print("\nDone. All vocals outputs saved to: outputs/")
