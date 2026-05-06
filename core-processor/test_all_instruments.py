"""
All-Instruments Integration Test
==================================
Runs every instrument pipeline on a single audio file and reports:
  - Detected mode / style / character
  - Note counts at each stage
  - Chroma similarity score (or onset F1 for drums)
  - Wall-clock time per instrument

Usage:
    cd core-processor
    python test_all_instruments.py "path/to/song.mp3"
    python test_all_instruments.py "path/to/song.mp3" --no-play --from-stage 2
    python test_all_instruments.py "path/to/song.mp3" --instruments guitar bass piano
"""

import argparse
import os
import sys
import time
import json
import subprocess
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON = sys.executable

_ALL_INSTRUMENTS = ["guitar", "bass", "piano", "vocals", "drums"]


def parse_args():
    p = argparse.ArgumentParser(description="Test all instrument pipelines on one audio file")
    p.add_argument("audio_file", help="Path to audio file (WAV, FLAC, M4A, MP3)")
    p.add_argument(
        "--instruments",
        nargs="+",
        choices=_ALL_INSTRUMENTS,
        default=_ALL_INSTRUMENTS,
        metavar="INST",
        help="Instruments to test (default: all)",
    )
    p.add_argument("--no-play",      action="store_true", help="Skip audio playback")
    p.add_argument("--no-viz",       action="store_true", help="Skip visualization")
    p.add_argument("--no-separate",  action="store_true", help="Skip separation (use saved stems)")
    p.add_argument("--from-stage",   type=int, default=1,
                   help="Resume from this stage for every instrument (saves time on re-runs)")
    p.add_argument("--score",        action="store_true", help="Compute accuracy scores")
    p.add_argument("--bpm-override", type=float, default=None)
    return p.parse_args()


def run_instrument(audio_file: str, instrument: str, flags: list[str]) -> dict:
    """Run main.py for one instrument, capture output, return result dict."""
    cmd = [
        _PYTHON, os.path.join(_HERE, "main.py"),
        audio_file,
        "--instrument", instrument,
    ] + flags

    print(f"\n{'='*70}")
    print(f"  Running: {instrument.upper()}")
    print(f"  Command: {' '.join(os.path.basename(c) if i < 2 else c for i, c in enumerate(cmd))}")
    print(f"{'='*70}")

    t0      = time.time()
    result  = subprocess.run(cmd, capture_output=False, text=True, cwd=_HERE)
    elapsed = time.time() - t0

    return {
        "instrument": instrument,
        "returncode": result.returncode,
        "elapsed_s":  round(elapsed, 1),
        "success":    result.returncode == 0,
    }


def collect_output_stats(audio_file: str, instruments: list[str]) -> dict[str, dict]:
    """Read pipeline output JSON files and compute summary stats."""
    from pipeline.config import set_outputs_dir, get_outputs_dir
    outputs_dir = set_outputs_dir(audio_file)

    stats = {}
    for inst in instruments:
        d = {"instrument": inst}

        if inst == "guitar":
            _read_json_count(d, outputs_dir, "02_raw_notes.json",     "raw_notes")
            _read_json_count(d, outputs_dir, "03_cleaned_notes.json", "cleaned_notes")
            _read_json_count(d, outputs_dir, "06_mapped_notes.json",  "mapped_notes")
            _read_key(d, outputs_dir, "05_key_analysis.json")

        elif inst == "bass":
            _read_json_count(d, outputs_dir, "02_bass_raw_notes.json",     "raw_notes")
            _read_json_count(d, outputs_dir, "03_bass_cleaned_notes.json", "cleaned_notes")
            _read_json_count(d, outputs_dir, "06_bass_mapped_notes.json",  "mapped_notes")
            _read_key(d, outputs_dir, "05_key_analysis.json")

        elif inst == "piano":
            _read_json_count(d, outputs_dir, "02_piano_raw_notes.json",     "raw_notes")
            _read_json_count(d, outputs_dir, "03_piano_cleaned_notes.json", "cleaned_notes")
            _read_json_count(d, outputs_dir, "06_piano_mapped_notes.json",  "mapped_notes")
            _read_key(d, outputs_dir, "05_key_analysis.json")

        elif inst == "vocals":
            _read_json_count(d, outputs_dir, "02_vocals_raw_notes.json",     "raw_notes")
            _read_json_count(d, outputs_dir, "03_vocals_cleaned_notes.json", "cleaned_notes")
            _read_vocals_mode(d, outputs_dir)
            _read_key(d, outputs_dir, "05_key_analysis.json")

        elif inst == "drums":
            _read_drum_hits(d, outputs_dir)

        stats[inst] = d

    return stats


def score_outputs(audio_file: str, instruments: list[str]) -> dict[str, dict]:
    """Compute evaluation scores for each instrument."""
    from pipeline.config import set_outputs_dir
    from pipeline.evaluation import score_instrument

    outputs_dir = set_outputs_dir(audio_file)
    results = {}
    for inst in instruments:
        print(f"[Score] Evaluating {inst}...", end=" ", flush=True)
        try:
            r = score_instrument(inst, outputs_dir)
            results[inst] = r
            if r["score"] is not None:
                print(f"{r['score']:.3f} ({r['metric']})")
            else:
                print(f"N/A — {r['details']}")
        except Exception as e:
            results[inst] = {"instrument": inst, "score": None,
                             "metric": "error", "details": str(e)}
            print(f"ERROR: {e}")
    return results


def print_summary(run_results: list[dict], stats: dict[str, dict],
                  scores: dict[str, dict] | None) -> None:
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    headers = ["Instrument", "Status", "Time(s)",
               "Raw", "Cleaned", "Mapped/Hits", "Key", "Score"]
    col_w   = [12, 8, 8, 6, 8, 12, 14, 8]

    def row(*cells):
        return "  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    print(row(*headers))
    print("  " + "-" * (sum(col_w) + 2 * len(col_w)))

    for r in run_results:
        inst   = r["instrument"]
        status = "OK" if r["success"] else f"ERR({r['returncode']})"
        t      = r["elapsed_s"]
        s      = stats.get(inst, {})
        sc     = scores.get(inst, {}) if scores else {}

        raw      = s.get("raw_notes",     s.get("hit_count", "—"))
        cleaned  = s.get("cleaned_notes", "—")
        mapped   = s.get("mapped_notes",  s.get("hit_classes", "—"))
        key      = s.get("key", "—")
        score_v  = f"{sc['score']:.3f}" if sc.get("score") is not None else "—"

        print(row(inst, status, t, raw, cleaned, mapped, key, score_v))

    print()

    # Per-instrument detail
    for r in run_results:
        inst = r["instrument"]
        s    = stats.get(inst, {})
        sc   = scores.get(inst, {}) if scores else {}
        if not r["success"]:
            print(f"  [{inst}] FAILED — exit code {r['returncode']}")
            continue
        detail_lines = [f"  [{inst}]"]
        for k, v in s.items():
            if k == "instrument":
                continue
            detail_lines.append(f"    {k}: {v}")
        if sc.get("score") is not None:
            detail_lines.append(f"    accuracy ({sc['metric']}): {sc['score']:.4f} = {sc['score']*100:.1f}%")
            detail_lines.append(f"    note: {sc.get('details', '')}")
        print("\n".join(detail_lines))
        print()


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _read_json_count(d, out_dir, filename, key):
    path = os.path.join(out_dir, filename)
    if os.path.isfile(path):
        try:
            with open(path) as f:
                d[key] = len(json.load(f))
        except Exception:
            d[key] = "?"
    else:
        d[key] = "—"


def _read_key(d, out_dir, filename):
    path = os.path.join(out_dir, filename)
    if os.path.isfile(path):
        try:
            with open(path) as f:
                info = json.load(f)
            d["key"] = info.get("key_str", info.get("key", "?"))
        except Exception:
            d["key"] = "?"
    else:
        d["key"] = "—"


def _read_drum_hits(d, out_dir):
    path = os.path.join(out_dir, "02_drums_hits.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                hits = json.load(f)
            from collections import Counter
            counts = Counter(h["hit_class"] for h in hits)
            d["hit_count"]   = len(hits)
            d["hit_classes"] = "  ".join(f"{c}={n}" for c, n in sorted(counts.items()))
        except Exception:
            d["hit_count"]   = "?"
            d["hit_classes"] = "?"
    else:
        d["hit_count"]   = "—"
        d["hit_classes"] = "—"


def _read_vocals_mode(d, out_dir):
    path = os.path.join(out_dir, "03_vocals_clean_meta.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                meta = json.load(f)
            d["vocals_mode"]  = meta.get("vocals_mode", "?")
            d["conf_thresh"]  = meta.get("conf_thresh", "?")
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.isfile(args.audio_file):
        print(f"Error: file not found: {args.audio_file}")
        sys.exit(1)

    # Build common flags to pass to main.py for each instrument
    flags = []
    if args.no_play:      flags += ["--no-play"]
    if args.no_viz:       flags += ["--no-viz"]
    if args.no_separate:  flags += ["--no-separate"]
    if args.score:        flags += ["--score"]
    if args.from_stage > 1:
        flags += ["--from-stage", str(args.from_stage)]
    if args.bpm_override:
        flags += ["--bpm-override", str(args.bpm_override)]

    print(f"\nAll-instruments test")
    print(f"  File       : {args.audio_file}")
    print(f"  Instruments: {', '.join(args.instruments)}")
    print(f"  Flags      : {' '.join(flags) or '(none)'}")

    # Run each instrument
    run_results = []
    for inst in args.instruments:
        r = run_instrument(args.audio_file, inst, flags)
        run_results.append(r)

    # Collect stats from output files
    print(f"\n[Stats] Reading output files...")
    try:
        sys.path.insert(0, _HERE)
        stats = collect_output_stats(args.audio_file, args.instruments)
    except Exception as e:
        print(f"[Stats] Failed to collect stats: {e}")
        stats = {}

    # Score if requested
    scores = None
    if args.score:
        print(f"\n[Score] Computing accuracy scores...")
        try:
            scores = score_outputs(args.audio_file, args.instruments)
        except Exception as e:
            print(f"[Score] Failed: {e}")

    # Print summary
    print_summary(run_results, stats, scores)

    # Write JSON report
    report_path = os.path.join(
        _HERE, "outputs",
        os.path.splitext(os.path.basename(args.audio_file))[0],
        "test_report.json",
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        "audio_file":   args.audio_file,
        "instruments":  args.instruments,
        "run_results":  run_results,
        "stats":        stats,
        "scores":       scores,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()
