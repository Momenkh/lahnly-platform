"""
Guitar pipeline — runs all stages (1–11) for guitar transcription.
Extracted from main.py to allow independent testing and registry dispatch.
"""

import os
import time

from pipeline.shared.stage_executor import StageExecutor


def _parse_time(s: str) -> float:
    """Parse MM:SS or plain seconds string to float seconds."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)


def _load_saved_tempo() -> dict | None:
    """Load 04_tempo.json if it exists, return None otherwise."""
    from pipeline.shared.quantization import load_quantization
    try:
        _, tempo = load_quantization()
        return tempo
    except FileNotFoundError:
        return None


def run_guitar_pipeline(args) -> None:
    executor = StageExecutor(args.from_stage)

    # ── Stage 1: Instrument Separation ───────────────────────────────────────
    if not args.no_separate and args.from_stage <= 1:
        from pipeline.shared.separation import separate_guitar
        pitch_input = separate_guitar(args.audio_file)
    elif not args.no_separate and args.from_stage <= 2:
        from pipeline.shared.separation import get_stem_path
        stem = get_stem_path()
        if os.path.isfile(stem):
            print("[Stage 1] Skipped — using saved guitar stem")
            pitch_input = stem
        else:
            from pipeline.shared.separation import separate_guitar
            print("[Stage 1] No saved stem found — running separation")
            pitch_input = separate_guitar(args.audio_file)
    else:
        if args.no_separate:
            print("[Stage 1] Skipped — using raw mix for pitch detection")
        pitch_input = args.audio_file

    # ── Auto-detect guitar type + role (when not explicitly specified) ───────
    if args.guitar_type is None or args.guitar_role is None:
        from pipeline.shared.separation import get_stem_path
        from pipeline.shared.auto_detect import detect_guitar_mode

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
                from pipeline.instruments.guitar.cleaning import load_clean_meta
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
        from pipeline.instruments.guitar.pitch import extract_pitches
        _t0 = time.perf_counter()
        raw_notes = extract_pitches(pitch_input, guitar_type=args.guitar_type, guitar_role=args.guitar_role)
        executor.timings["2"] = time.perf_counter() - _t0
        print(f"[Timing] Stage 2: {executor.timings['2']:.1f}s")
    elif args.from_stage <= 3:
        from pipeline.instruments.guitar.pitch import load_raw_notes
        print("[Stage 2] Skipped — loading saved raw notes")
        raw_notes = load_raw_notes()
    # else: raw_notes not needed — Stage 3 loads its own saved output

    # ── Stage 3: Note Cleaning ────────────────────────────────────────────────
    if args.from_stage <= 3:
        from pipeline.instruments.guitar.cleaning import clean_notes
        _t0 = time.perf_counter()
        cleaned_notes = clean_notes(raw_notes, guitar_type=args.guitar_type, guitar_role=args.guitar_role)
        executor.timings["3"] = time.perf_counter() - _t0
        print(f"[Timing] Stage 3: {executor.timings['3']:.1f}s")
    else:
        from pipeline.instruments.guitar.cleaning import load_cleaned_notes, load_clean_meta
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

    # ── Velocity Estimation (between Stage 3 and Stage 4) ────────────────────
    # Compute per-note MIDI velocity from stem RMS at the attack window.
    # This adds a "velocity" field to each note that Stage 9 synthesis uses for
    # realistic dynamics, improving the preview's amplitude match to the stem.
    from pipeline.shared.separation import get_stem_path
    _stem_for_vel = get_stem_path()
    if os.path.isfile(_stem_for_vel):
        from pipeline.shared.spectral import compute_velocity
        cleaned_notes = compute_velocity(cleaned_notes, _stem_for_vel)
        print(f"[Velocity] Estimated velocity for {len(cleaned_notes)} notes")

    # ── Stage 4: Tempo Detection & Quantization ───────────────────────────────
    # Save pre-quantization notes for synthesis — quantization snaps timing to a
    # BPM grid which can drift when the original recording has rubato or unstable
    # tempo.  The synthesized preview uses the original timing so its chroma
    # matches the stem more accurately; tabs use the quantized timing.
    pre_quant_notes = list(cleaned_notes)

    if not args.no_quantize:
        from pipeline.shared.quantization import quantize_notes

        if args.from_stage <= 4:
            reuse_tempo = None
            if args.from_stage >= 3 and not args.force_tempo and args.bpm_override is None:
                reuse_tempo = _load_saved_tempo()
                if reuse_tempo:
                    print(f"[Stage 4] Saved BPM found ({reuse_tempo['bpm']:.1f}) — "
                          f"reusing (--force-tempo to override)")
            _t0 = time.perf_counter()
            cleaned_notes, tempo_info = quantize_notes(
                cleaned_notes, pitch_input,
                bpm_override=args.bpm_override,
                reuse_tempo=reuse_tempo,
            )
            executor.timings["4"] = time.perf_counter() - _t0
            print(f"[Timing] Stage 4: {executor.timings['4']:.1f}s")
        else:
            from pipeline.shared.quantization import load_quantization
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

    # ── Stage 5: Key / Scale Analysis (global + per-segment) ─────────────────
    if args.from_stage <= 5:
        from pipeline.shared.music_theory import analyze_key_segmented
        _t0 = time.perf_counter()
        key_info, key_segments = analyze_key_segmented(cleaned_notes, instrument="guitar")
        executor.timings["5"] = time.perf_counter() - _t0
        print(f"[Timing] Stage 5: {executor.timings['5']:.1f}s")
    else:
        from pipeline.shared.music_theory import load_key_analysis
        try:
            print("[Stage 5] Skipped — loading saved key analysis")
            key_info = load_key_analysis()
        except FileNotFoundError:
            from pipeline.shared.music_theory import analyze_key_segmented
            print("[Stage 5] No saved key analysis — running now")
            key_info, key_segments = analyze_key_segmented(cleaned_notes, instrument="guitar")
        key_segments = []  # segments not persisted; fall back to global key for stage 5b

    # ── Stage 5b: Key-context feedback (octave correction + confidence filter) ──
    from pipeline.instruments.guitar.cleaning import (
        apply_key_octave_correction,
        apply_key_confidence_filter,
        apply_key_confidence_filter_segmented,
    )
    from pipeline.instruments.guitar.chords import get_chord_tone_pcs
    from pipeline.settings import CLEANING_KEY_CONFIDENCE_CUTOFF

    # Pre-chord pass: collect pitch classes that appear in any chord group so the
    # key-confidence filter does not delete legitimate chromatic chord tones
    # (e.g. the b7 of a V7 chord in C major, or the b3 of a borrowed iv chord).
    protected_pcs = get_chord_tone_pcs(cleaned_notes, tempo_info=tempo_info)
    if protected_pcs:
        print(f"[Stage 5b] Chord-tone protection active for PCs: "
              f"{sorted(protected_pcs)}")

    cleaned_notes = apply_key_octave_correction(cleaned_notes, key_info)
    cleaned_notes = apply_key_confidence_filter_segmented(
        cleaned_notes, key_segments, key_info,
        conf_cutoff=CLEANING_KEY_CONFIDENCE_CUTOFF,
        protected_pcs=protected_pcs,
    )
    # Apply the same corrections to the pre-quantization copy so the audio
    # synthesis path gets accurate pitch and the same note set.
    pre_quant_notes = apply_key_octave_correction(pre_quant_notes, key_info)
    pre_quant_notes = apply_key_confidence_filter_segmented(
        pre_quant_notes, key_segments, key_info,
        conf_cutoff=CLEANING_KEY_CONFIDENCE_CUTOFF,
        protected_pcs=protected_pcs,
    )

    # ── Stage 6: Guitar Mapping ───────────────────────────────────────────────
    capo_fret = key_info.get("capo_fret", 0)
    if args.from_stage <= 6:
        from pipeline.instruments.guitar.mapping import map_to_guitar
        _t0 = time.perf_counter()
        mapped_notes = map_to_guitar(cleaned_notes, key_info=key_info,
                                     guitar_type=args.guitar_type, guitar_role=args.guitar_role,
                                     capo_fret=capo_fret)
        executor.timings["6"] = time.perf_counter() - _t0
        print(f"[Timing] Stage 6: {executor.timings['6']:.1f}s")
    else:
        from pipeline.instruments.guitar.mapping import load_mapped_notes
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
    from pipeline.instruments.guitar.mapping import isolate_melody, map_to_guitar as _map
    from pipeline.settings import MELODY_MIN_PITCH
    melody_notes, harmony_notes = isolate_melody(
        mapped_notes,
        min_pitch=MELODY_MIN_PITCH.get(args.guitar_role, 0),
    )

    tab_notes = melody_notes if args.guitar_role == "lead" else mapped_notes

    # Audio synthesis uses pre-quantization timing so the chroma matches the stem
    # without grid-snapping drift.  Key context (key_info) is still applied via
    # the 5b filters already run on pre_quant_notes.
    # Always use ALL mapped notes for the audio preview regardless of role — the
    # preview should sound like the song, not just the top voice. Tab display is
    # the only output that restricts to melody for lead role.
    audio_mapped = _map(pre_quant_notes, key_info=key_info,
                        guitar_type=args.guitar_type, guitar_role=args.guitar_role,
                        capo_fret=capo_fret, save=False)
    audio_notes = audio_mapped

    # ── Stage 7: Chord Detection ──────────────────────────────────────────────
    # harmony_notes excludes fast single-voice runs so they don't trigger
    # false chord groups. For lead role this is the melody complement;
    # for rhythm role pass all mapped notes (harmony_notes ≈ mapped_notes).
    chord_input = harmony_notes if args.guitar_role == "lead" else mapped_notes
    if args.from_stage <= 7:
        from pipeline.instruments.guitar.chords import detect_chords
        _t0 = time.perf_counter()
        _, chord_groups = detect_chords(
            chord_input, tempo_info=tempo_info, key_info=key_info
        )
        executor.timings["7"] = time.perf_counter() - _t0
        print(f"[Timing] Stage 7: {executor.timings['7']:.1f}s")
    else:
        from pipeline.instruments.guitar.chords import load_chord_detection
        try:
            print("[Stage 7] Skipped — loading saved chord detection")
            _, chord_groups = load_chord_detection()
        except FileNotFoundError:
            from pipeline.instruments.guitar.chords import detect_chords
            print("[Stage 7] No saved chord data — running now")
            _, chord_groups = detect_chords(
                chord_input, tempo_info=tempo_info, key_info=key_info
            )

    # ── Stage 8: Tab Generation ───────────────────────────────────────────────
    # lead: melody only (single-voice); rhythm/acoustic: all mapped notes (chord columns)
    if args.from_stage <= 8:
        from pipeline.instruments.guitar.tab import generate_tabs
        _t0 = time.perf_counter()
        tab_str = generate_tabs(tab_notes, chord_groups=chord_groups, tempo_info=tempo_info, capo_fret=capo_fret)
        executor.timings["8"] = time.perf_counter() - _t0
        print(f"[Timing] Stage 8: {executor.timings['8']:.1f}s")
    else:
        from pipeline.instruments.guitar.tab import load_tabs
        print("[Stage 8] Skipped — loading saved tabs")
        tab_str = load_tabs()

    # ── Stage 9: Audio Export + Playback ─────────────────────────────────────
    if args.from_stage <= 9:
        from pipeline.shared.audio import save_audio, play_notes
        _t0 = time.perf_counter()
        save_audio(audio_notes)
        executor.timings["9"] = time.perf_counter() - _t0
        print(f"[Timing] Stage 9: {executor.timings['9']:.1f}s")
        if not args.no_play:
            play_notes(audio_notes)
    else:
        print("[Stage 9] Skipped")

    # ── Score: Chroma Similarity ──────────────────────────────────────────────
    if args.score:
        from pipeline.evaluation import score_transcription
        from pipeline.shared.separation import get_stem_path
        from pipeline.config import get_instrument_dir

        stem_path    = get_stem_path()
        preview_path = os.path.join(get_instrument_dir("guitar"), "09_preview.wav")

        if os.path.isfile(stem_path) and os.path.isfile(preview_path):
            sim = score_transcription(stem_path, preview_path)
            print(f"\n[Score] Chroma similarity (stem vs preview): {sim:.3f}"
                  f"  (0.55 = poor  |  0.70 = usable  |  0.85 = very good)")
        else:
            missing = stem_path if not os.path.isfile(stem_path) else preview_path
            print(f"[Score] Skipped — file not found: {missing}")

    # ── Stage 10: Fretboard Visualization ────────────────────────────────────
    if not args.no_viz and args.from_stage <= 10:
        from pipeline.instruments.guitar.visualization import plot_fretboard
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
        from pipeline.instruments.guitar.chord_sheet import plot_chord_sheet
        chord_path = plot_chord_sheet(
            chord_groups, tempo_info=tempo_info, save=True, show=False
        )
        if chord_path:
            print(f"[Stage 11] Chord sheet saved to: {chord_path}")
    else:
        print("[Stage 11] Skipped — visualization disabled")

    executor.print_summary("guitar")
    print("\nDone. All outputs saved to: outputs/")
