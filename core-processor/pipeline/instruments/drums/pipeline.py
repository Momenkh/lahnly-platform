"""
Drums Pipeline
===============
Drums bypass the pitch pipeline entirely.

Stages 2–7 (pitch extraction, cleaning, quantization, key, mapping, chord) are
all replaced by a single onset detection + hit classification step.

Stage flow:
  1  → Stem separation  (Demucs htdemucs_6s → drums stem)
  1.5 → Auto-detect drums character (sparse / dense) — logging only
  2D → Onset detection + hit classification  (drums_onset.py)
  4  → Tempo detection (BPM only — no quantization of hits needed)
  8D → ASCII drum grid notation  (drums_notation.py)
  9D → (no pitched audio synthesis; stem-only playback option)
  10D → Drum pattern visualization  (drums_visualization.py)
"""

import os

from pipeline.shared.stage_executor import StageExecutor, parse_time_range


def run_drums_pipeline(args) -> None:
    from pipeline.config import get_shared_dir
    executor = StageExecutor(args.from_stage)

    _DRUMS_CASCADE = [("htdemucs_6s", "drums"), ("htdemucs", "drums")]

    # ── Stage 1: Stem Separation ──────────────────────────────────────────────
    if not args.no_separate and args.from_stage <= 1:
        from pipeline.shared.separation import separate_stem
        drum_stem = separate_stem(args.audio_file, "drums", _DRUMS_CASCADE,
                                  instrument_label="drums")
    elif not args.no_separate and args.from_stage <= 2:
        from pipeline.shared.separation import get_stem_path_for, separate_stem
        stem = get_stem_path_for("drums")
        if os.path.isfile(stem):
            print("[Stage 1D] Skipped — using saved drums stem")
            drum_stem = stem
        else:
            print("[Stage 1D] No saved stem found — running separation")
            drum_stem = separate_stem(args.audio_file, "drums", _DRUMS_CASCADE,
                                      instrument_label="drums")
    else:
        if args.no_separate:
            print("[Stage 1D] Skipped — using raw mix")
        drum_stem = args.audio_file

    # ── Stage 1.5: Auto-detect character ─────────────────────────────────────
    from pipeline.shared.separation import get_stem_path_for
    from pipeline.shared.auto_detect import detect_drums_mode

    sp = get_stem_path_for("drums")
    detect_src = sp if os.path.isfile(sp) else drum_stem
    if detect_src and os.path.isfile(detect_src):
        print(f"[Auto D] Detecting drum character from: {os.path.basename(detect_src)}")
        detected   = detect_drums_mode(detect_src)
        drums_char = detected["drums_character"]
        conf       = detected["mode_confidence"]
        f          = detected["features"]
        print(f"[Auto D] onset_density={f['onset_density_per_s']:.1f}/s  "
              f"centroid={f['mean_centroid_hz']:.0f} Hz")
        print(f"[Auto D] Character: {drums_char:<10} ({conf:.0%} confidence)")
    else:
        drums_char = "unknown"

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Instrument : drums")
    print(f"  Character  : {drums_char}")
    print(f"{sep}\n")

    # ── Stage 2D: Onset Detection + Hit Classification ────────────────────────
    from pipeline.instruments.drums.onset import detect_drum_hits, load_drum_hits
    hits = executor.run_or_load(
        2,
        lambda: detect_drum_hits(drum_stem, save=True),
        load_drum_hits,
        skip_msg="[Stage 2D] Skipped — loading saved drum hits",
    )

    # ── Time range filter ─────────────────────────────────────────────────────
    t_start_s, t_end_s = parse_time_range(args)
    if t_start_s is not None or t_end_s is not None:
        before = len(hits)
        lo = t_start_s or 0.0
        hi = t_end_s or float("inf")
        hits = [h for h in hits if lo <= h["start"] <= hi]
        print(f"[Filter D] Time range: {len(hits)} hits "
              f"(removed {before - len(hits)})")

    # ── Stage 4 (BPM only): Tempo Detection ──────────────────────────────────
    tempo_info = None
    if not args.no_quantize:
        bpm_override = getattr(args, "bpm_override", None)
        if bpm_override:
            import json
            tempo_info = {"bpm": bpm_override}
            with open(os.path.join(get_shared_dir(), "04_tempo.json"), "w") as f:
                json.dump(tempo_info, f)
            print(f"[Stage 4D] BPM override: {bpm_override}")
        else:
            try:
                from pipeline.shared.quantization import load_quantization
                _, tempo_info = load_quantization()
                if tempo_info:
                    print(f"[Stage 4D] Saved BPM ({tempo_info['bpm']:.1f}) — reusing")
            except FileNotFoundError:
                try:
                    import librosa
                    y, sr = librosa.load(drum_stem, sr=22050, mono=True, duration=60.0)
                    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
                    bpm = float(bpm)
                    tempo_info = {"bpm": bpm}
                    import json
                    with open(os.path.join(get_shared_dir(), "04_tempo.json"), "w") as f:
                        json.dump(tempo_info, f)
                    print(f"[Stage 4D] Detected BPM: {bpm:.1f}")
                except Exception as e:
                    print(f"[Stage 4D] BPM detection failed ({e}) — using 120")
                    tempo_info = {"bpm": 120.0}

    bpm = tempo_info["bpm"] if tempo_info else None

    # ── Stage 8D: ASCII Drum Grid Notation ───────────────────────────────────
    if args.from_stage <= 8:
        from pipeline.instruments.drums.notation import generate_drum_grid
        generate_drum_grid(hits, tempo_bpm=bpm, save=True)
    else:
        print("[Stage 8D] Skipped")

    # ── Stage 9D: Playback — play stem directly (no synthesis) ───────────────
    if args.from_stage <= 9:
        if not getattr(args, "no_play", False):
            try:
                import sounddevice as sd
                import soundfile as sf
                print(f"[Stage 9D] Playing drum stem: {os.path.basename(drum_stem)}")
                data, srate = sf.read(drum_stem, dtype="float32")
                sd.play(data, srate)
                sd.wait()
            except Exception as e:
                print(f"[Stage 9D] Playback skipped ({e})")
        else:
            print("[Stage 9D] Playback skipped (--no-play)")
    else:
        print("[Stage 9D] Skipped")

    # ── Stage 10D: Drum Pattern Visualization ────────────────────────────────
    if not getattr(args, "no_viz", False) and args.from_stage <= 10:
        from pipeline.instruments.drums.visualization import plot_drum_pattern
        img_path = plot_drum_pattern(hits, save=True, show=False)
        if img_path:
            print(f"[Stage 10D] Pattern saved to: {img_path}")
    else:
        print("[Stage 10D] Skipped — visualization disabled")

    # ── Scoring — drums use onset density similarity ──────────────────────────
    if getattr(args, "score", False):
        from pipeline.evaluation import score_drums
        sp = get_stem_path_for("drums")
        if os.path.isfile(sp):
            s = score_drums(sp, hits)
            print(f"\n[Score D] Onset F1: {s:.3f}  ({s*100:.1f}%)")
        else:
            print("[Score D] Could not compute score — stem missing")

    executor.print_summary("drums")
    print("\nDone. All drums outputs saved to: outputs/")
