"""
Batch accuracy benchmark — auto-discovers all audio files in the current directory,
infers guitar type+role from the filename, runs the pipeline, and appends scores to
benchmark_results.json.

Type+role inference rules (case-insensitive filename match):
  contains "acoustic"/"accoustic"  → type=acoustic
  contains "distorted"/"dist"      → type=distorted
  otherwise                        → type=clean
  contains "lead"/"solo"           → role=lead
  contains "rhythm"                → role=rhythm
  no role match (full/mixed song)  → TWO runs: role=rhythm + role=lead

Usage:
  python batch_score.py                        # run all discovered songs
  python batch_score.py --type lead            # run only lead-role songs
  python batch_score.py --type rhythm          # run only rhythm-role songs
  python batch_score.py --type acoustic        # run only acoustic-type songs
  python batch_score.py --from-stage 2        # skip separation (reuse existing stems)
  python batch_score.py --no-save             # print results only, don't write to JSON
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
RESULTS_FILE     = "benchmark_results.json"
SCORE_RE         = re.compile(r"\[Score\] Chroma similarity \(stem vs preview\): ([0-9.]+)")


def infer_type_and_role(filename: str) -> list[tuple[str, str]]:
    """
    Returns list of (guitar_type, guitar_role) pairs to run for this file.
    Labelled files → single pair.
    Unlabelled (mixed/full song) → two runs: rhythm + lead (both clean type).
    """
    name = filename.lower()

    if any(k in name for k in ("acoustic", "accoustic")):
        guitar_type = "acoustic"
    elif any(k in name for k in ("distorted", "dist")):
        guitar_type = "distorted"
    else:
        guitar_type = "clean"

    if any(k in name for k in ("lead", "solo")):
        return [(guitar_type, "lead")]
    if "rhythm" in name:
        return [(guitar_type, "rhythm")]
    # No role label → run twice
    return [(guitar_type, "rhythm"), (guitar_type, "lead")]


def discover_songs(type_filter: str | None = None) -> list[tuple[str, str, str]]:
    """Returns list of (filename, guitar_type, guitar_role) triples to run.

    type_filter may be a guitar type ("acoustic", "clean", "distorted") or
    a role ("lead", "rhythm") — matched against either field.
    """
    songs = []
    seen = set()
    for ext in AUDIO_EXTENSIONS:
        for f in glob.glob(f"*{ext}") + glob.glob(f"*{ext.upper()}"):
            if f in seen:
                continue
            seen.add(f)
            for gtype, grole in infer_type_and_role(f):
                if type_filter is None or type_filter in (gtype, grole):
                    songs.append((f, gtype, grole))
    return sorted(songs)


def load_results() -> dict:
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "description": (
            "Chroma similarity benchmark results (stem vs synthesized preview). "
            "Scale: 0.55=poor, 0.70=usable, 0.85=very good."
        ),
        "runs": [],
    }


def save_results(data: dict) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def run_song(filename: str, guitar_type: str, guitar_role: str, extra_args: list[str]) -> float | None:
    cmd = [
        sys.executable, "main.py", filename,
        "--type", guitar_type,
        "--role", guitar_role,
        "--score", "--no-play", "--no-viz",
    ] + extra_args

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = proc.stdout + proc.stderr

    keywords = ["[Stage", "[Score", "[Warning", "Error", "Traceback"]
    for line in output.splitlines():
        if any(k in line for k in keywords):
            print("  " + line)

    m = SCORE_RE.search(output)
    return float(m.group(1)) if m else None


def print_summary(runs: list[dict]) -> None:
    if not runs:
        return

    print("\n" + "=" * 80)
    print(f"  {'Song':<40} {'Mode':<18} {'Score':>6}  {'Timestamp'}")
    print("=" * 80)
    for r in runs:
        ts   = r.get("timestamp", r.get("date", "—"))
        mode = r.get("mode", f"{r.get('guitar_type','?')}_{r.get('guitar_role','?')}")
        print(f"  {r['song'][:40]:<40} {mode:<18} {r['score']:>6.3f}  {ts}")

    scores = [r["score"] for r in runs]
    print("-" * 80)
    print(f"  Total runs: {len(scores)}   Overall avg: {sum(scores)/len(scores):.3f}")

    by_mode: dict[str, list[float]] = {}
    for r in runs:
        mode = r.get("mode", f"{r.get('guitar_type','?')}_{r.get('guitar_role','?')}")
        by_mode.setdefault(mode, []).append(r["score"])
    for mode, sc in sorted(by_mode.items()):
        print(f"    {mode:<18}: avg {sum(sc)/len(sc):.3f}  ({len(sc)} runs)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Batch chroma-similarity benchmark")
    parser.add_argument(
        "--type", choices=["acoustic", "clean", "distorted", "lead", "rhythm"], default=None,
        help="Filter: only run songs matching this type (acoustic/clean/distorted) or role (lead/rhythm)",
    )
    parser.add_argument(
        "--from-stage", type=int, default=1, metavar="N",
        help="Resume each song from stage N (e.g. 2 to skip separation)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print results only — do not append to benchmark_results.json",
    )
    args = parser.parse_args()

    songs = discover_songs(type_filter=args.type)
    if not songs:
        print("No matching audio files found.")
        return

    extra_args = ["--from-stage", str(args.from_stage)] if args.from_stage > 1 else []

    results_data = load_results()
    new_runs: list[dict] = []

    print("=" * 80)
    label = f"filter={args.type}" if args.type else "all"
    print(f"  Discovered {len(songs)} run(s) — {label}")
    print("=" * 80)

    for filename, guitar_type, guitar_role in songs:
        mode = f"{guitar_type}_{guitar_role}"
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"\n[{mode}] {filename}", flush=True)
        score = run_song(filename, guitar_type, guitar_role, extra_args)

        if score is not None:
            print(f"  → Score: {score:.3f}")
            entry = {
                "song":        filename,
                "guitar_type": guitar_type,
                "guitar_role": guitar_role,
                "mode":        mode,
                "score":       score,
                "timestamp":   ts,
            }
            new_runs.append(entry)
            if not args.no_save:
                results_data["runs"].append(entry)
                save_results(results_data)
        else:
            print("  → ERROR: no score extracted")

    print_summary(new_runs if new_runs else results_data.get("runs", []))


if __name__ == "__main__":
    main()
