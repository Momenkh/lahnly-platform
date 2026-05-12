"""
Batch accuracy benchmark — auto-discovers all audio files in the current directory,
infers guitar role from the filename, lets the pipeline auto-detect the type from
the separated stem, and appends scores to benchmark_results.json.

Role inference rules (case-insensitive filename match):
  contains "lead"/"solo"  → role=lead   (single run)
  contains "rhythm"       → role=rhythm (single run)
  no match (full song)    → TWO runs: role=rhythm + role=lead

Usage:
  python batch_score.py                     run all discovered songs (legacy score)
  python batch_score.py --v2                run with full v2 metric bundle
  python batch_score.py --analyze           analyze accumulated benchmark_results.json
  python batch_score.py --compare           diff the last two batches in the results
  python batch_score.py --role lead         run only lead-role songs
  python batch_score.py --from-stage 2      skip separation (reuse existing stems)
  python batch_score.py --no-save           print results only, don't write to JSON
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
MODE_RE          = re.compile(r"Mode\s*:\s*(\w+)_(\w+)")
_CHROMA_NAMES    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def infer_roles(filename: str) -> list[str]:
    name = filename.lower()
    if any(k in name for k in ("lead", "solo")):
        return ["lead"]
    if "rhythm" in name:
        return ["rhythm"]
    return ["rhythm", "lead"]


def discover_songs(role_filter: str | None = None) -> list[tuple[str, str]]:
    songs = []
    seen  = set()
    for ext in AUDIO_EXTENSIONS:
        for f in glob.glob(f"*{ext}") + glob.glob(f"*{ext.upper()}"):
            if f in seen:
                continue
            seen.add(f)
            for role in infer_roles(f):
                if role_filter is None or role == role_filter:
                    songs.append((f, role))
    return sorted(songs)


def load_results() -> dict:
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "description": (
            "Benchmark results. "
            "Primary metric: cqt_sim (octave-aware). "
            "Legacy: chroma_sim. "
            "Scale: 0.55=poor, 0.70=usable, 0.85=very good."
        ),
        "runs": [],
    }


def save_results(data: dict) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def run_song(filename: str, guitar_role: str, extra_args: list[str], use_v2: bool) -> dict | None:
    """
    Run the pipeline for one song+role.
    With use_v2=True, calls score_transcription_v2 after the pipeline completes
    using the saved stem and preview WAVs.
    Returns a run-entry dict or None on error.
    """
    cmd = [
        sys.executable, "main.py", filename,
        "--role", guitar_role,
        "--score", "--no-play", "--no-viz",
    ] + extra_args

    proc   = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = proc.stdout + proc.stderr

    keywords = ["[Auto]", "Mode :", "[Stage", "[Score", "[Warning", "Error", "Traceback"]
    for line in output.splitlines():
        if any(k in line for k in keywords):
            print("  " + line)

    score_m = SCORE_RE.search(output)
    mode_m  = MODE_RE.search(output)

    score         = float(score_m.group(1)) if score_m else None
    detected_type = mode_m.group(1) if mode_m else None
    detected_role = mode_m.group(2) if mode_m else None

    if score is None:
        return None

    guitar_type = detected_type or "unknown"
    actual_role = detected_role or guitar_role
    mode        = f"{guitar_type}_{actual_role}"

    entry: dict = {
        "song":        filename,
        "guitar_type": guitar_type,
        "guitar_role": actual_role,
        "mode":        mode,
        "score":       score,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    if use_v2:
        v2 = _run_v2_score(filename, instrument="guitar")
        if v2:
            entry["v2"] = v2
            print(f"  -> cqt_sim={v2.get('cqt_sim'):.3f}  "
                  f"prec={v2.get('spectral_precision')}  "
                  f"recall={v2.get('spectral_recall')}  "
                  f"f1={v2.get('pseudo_f1')}")

    return entry


def _run_v2_score(audio_filename: str, instrument: str = "guitar") -> dict | None:
    """Load saved outputs and call score_transcription_v2."""
    try:
        from pipeline.evaluation import score_transcription_v2
        import json as _json

        song_name = os.path.splitext(audio_filename)[0]
        outputs_dir  = os.path.join("outputs", song_name)
        instr_dir    = os.path.join(outputs_dir, instrument)
        stem_path    = os.path.join(instr_dir, "01_stem.wav")
        preview_path = os.path.join(instr_dir, "09_preview.wav")
        notes_path   = os.path.join(instr_dir, "03_cleaned_notes.json")

        if not os.path.isfile(stem_path) or not os.path.isfile(preview_path):
            return None

        notes = None
        if os.path.isfile(notes_path):
            with open(notes_path) as f:
                notes = _json.load(f)

        return score_transcription_v2(stem_path, preview_path, cleaned_notes=notes)
    except Exception as e:
        print(f"  [v2 warning] {e}")
        return None


def print_summary(runs: list[dict]) -> None:
    if not runs:
        return

    print("\n" + "=" * 90)
    print(f"  {'Song':<38} {'Mode':<18} {'Score':>6}  {'CQT':>6}  {'F1':>6}  {'Timestamp'}")
    print("=" * 90)
    for r in runs:
        ts    = r.get("timestamp", r.get("date", "—"))
        mode  = r.get("mode", f"{r.get('guitar_type','?')}_{r.get('guitar_role','?')}")
        cqt   = r.get("v2", {}).get("cqt_sim") if isinstance(r.get("v2"), dict) else None
        f1    = r.get("v2", {}).get("pseudo_f1") if isinstance(r.get("v2"), dict) else None
        cqt_s = f"{cqt:.3f}" if cqt is not None else "  —  "
        f1_s  = f"{f1:.3f}"  if f1  is not None else "  —  "
        print(f"  {r['song'][:38]:<38} {mode:<18} {r['score']:>6.3f}  {cqt_s:>6}  {f1_s:>6}  {ts}")

    scores = [r["score"] for r in runs]
    print("-" * 90)
    print(f"  Total runs: {len(scores)}   Chroma avg: {sum(scores)/len(scores):.3f}")

    by_mode: dict[str, list[float]] = {}
    for r in runs:
        mode = r.get("mode", f"{r.get('guitar_type','?')}_{r.get('guitar_role','?')}")
        by_mode.setdefault(mode, []).append(r["score"])
    for mode, sc in sorted(by_mode.items()):
        print(f"    {mode:<20}: avg {sum(sc)/len(sc):.3f}  ({len(sc)} runs)")
    print("=" * 90)


def cmd_analyze(results_data: dict) -> None:
    """Analyze accumulated benchmark_results.json and print insights."""
    runs = results_data.get("runs", [])
    if not runs:
        print("No runs in benchmark_results.json")
        return

    print("\n" + "=" * 80)
    print("  BENCHMARK ANALYSIS")
    print("=" * 80)

    # Overall stats
    chroma_scores = [r["score"] for r in runs if r.get("score") is not None]
    cqt_scores    = [r["v2"]["cqt_sim"] for r in runs
                     if isinstance(r.get("v2"), dict) and r["v2"].get("cqt_sim") is not None]
    f1_scores     = [r["v2"]["pseudo_f1"] for r in runs
                     if isinstance(r.get("v2"), dict) and r["v2"].get("pseudo_f1") is not None]

    print(f"\n  Runs analyzed: {len(runs)}")
    if chroma_scores:
        print(f"  Chroma sim  — avg: {sum(chroma_scores)/len(chroma_scores):.3f}"
              f"  min: {min(chroma_scores):.3f}  max: {max(chroma_scores):.3f}")
    if cqt_scores:
        print(f"  CQT sim     — avg: {sum(cqt_scores)/len(cqt_scores):.3f}"
              f"  min: {min(cqt_scores):.3f}  max: {max(cqt_scores):.3f}")
    if f1_scores:
        print(f"  Pseudo-F1   — avg: {sum(f1_scores)/len(f1_scores):.3f}"
              f"  min: {min(f1_scores):.3f}  max: {max(f1_scores):.3f}")

    # Per-pitch-class breakdown (v2 runs only)
    v2_runs = [r for r in runs if isinstance(r.get("v2"), dict)
               and isinstance(r["v2"].get("per_pitch_class"), dict)]
    if v2_runs:
        print("\n  Per-pitch-class chroma similarity (averaged across all v2 runs):")
        pc_avgs = {}
        for name in _CHROMA_NAMES:
            vals = [r["v2"]["per_pitch_class"].get(name) for r in v2_runs
                    if r["v2"]["per_pitch_class"].get(name) is not None]
            pc_avgs[name] = sum(vals) / len(vals) if vals else None
        sorted_pcs = sorted(
            [(k, v) for k, v in pc_avgs.items() if v is not None],
            key=lambda x: x[1],
        )
        for name, score in sorted_pcs:
            bar = "#" * int(score * 20)
            flag = "  ← LOW" if score < 0.60 else ""
            print(f"    {name:>3}: {score:.3f}  {bar}{flag}")

    # Worst songs
    if chroma_scores:
        print("\n  5 worst-scoring songs:")
        sorted_runs = sorted(runs, key=lambda r: r.get("score", 1.0))
        for r in sorted_runs[:5]:
            mode = r.get("mode", "?")
            cqt  = r.get("v2", {}).get("cqt_sim") if isinstance(r.get("v2"), dict) else None
            print(f"    {r['song']:<45}  chroma={r['score']:.3f}"
                  + (f"  cqt={cqt:.3f}" if cqt else ""))

    # Worst temporal clusters
    all_clusters = []
    for r in v2_runs:
        for cl in r["v2"].get("temporal_clusters", []):
            if cl.get("severity") == "high":
                all_clusters.append((r["song"], cl))
    if all_clusters:
        print(f"\n  High-severity temporal error clusters ({len(all_clusters)} total):")
        for song, cl in all_clusters[:8]:
            print(f"    {song[:40]:<40}  t=[{cl['start']:.1f}–{cl['end']:.1f}s]"
                  f"  local_score={cl['score']:.3f}")

    print("=" * 80)


def cmd_compare(results_data: dict) -> None:
    """Compare the last two batches (by timestamp) and print metric deltas."""
    runs = results_data.get("runs", [])
    if len(runs) < 2:
        print("Need at least 2 runs to compare.")
        return

    # Group by song+mode, take last 2 entries per group
    groups: dict[str, list[dict]] = {}
    for r in runs:
        key = f"{r['song']}_{r.get('mode','?')}"
        groups.setdefault(key, []).append(r)

    metrics = ["score", "cqt_sim", "spectral_precision", "spectral_recall", "pseudo_f1"]
    deltas  = {m: [] for m in metrics}

    print("\n" + "=" * 90)
    print("  BEFORE → AFTER COMPARISON (last 2 entries per song+mode)")
    print(f"  {'Song+Mode':<42}  {'chroma':>7}  {'cqt':>7}  {'prec':>7}  {'recall':>7}  {'f1':>7}")
    print("=" * 90)

    for key, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        before, after = group[-2], group[-1]

        def _get(r, metric):
            if metric == "score":
                return r.get("score")
            return r.get("v2", {}).get(metric) if isinstance(r.get("v2"), dict) else None

        def _delta_str(m):
            b, a = _get(before, m), _get(after, m)
            if b is None or a is None:
                return "   —  "
            d = a - b
            deltas[m].append(d)
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:+.3f}"

        parts = [_delta_str(m) for m in ["score", "cqt_sim", "spectral_precision",
                                          "spectral_recall", "pseudo_f1"]]
        print(f"  {key[:42]:<42}  " + "  ".join(f"{p:>7}" for p in parts))

    print("-" * 90)
    print("  Averages:")
    for m in metrics:
        if deltas[m]:
            avg = sum(deltas[m]) / len(deltas[m])
            sign = "+" if avg >= 0 else ""
            print(f"    {m:<22}: {sign}{avg:+.4f}")
    print("=" * 90)


def cmd_sweep_conf(instrument: str, mode: str, sweep_step: float = 0.01) -> None:
    """
    Sweep conf_floor over raw notes from the instrument's output directory.
    For each threshold value, compute spectral_precision and spectral_recall
    on the filtered note set.  Reports best conf_floor for pseudo_f1.
    Does not re-run the pipeline — works on saved outputs only.
    """
    import json as _json

    try:
        from pipeline.shared.spectral import compute_stem_cqt
        from pipeline.evaluation import _compute_spectral_precision, _compute_spectral_recall
    except ImportError as e:
        print(f"[sweep-conf] Import error: {e}")
        return

    # Find most recent song's outputs for this instrument
    song_dirs = sorted(
        (d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))),
        key=lambda d: os.path.getmtime(os.path.join("outputs", d)),
        reverse=True,
    )
    if not song_dirs:
        print("[sweep-conf] No output directories found in outputs/")
        return

    results = []
    songs_checked = 0

    for song_dir in song_dirs[:5]:  # check up to 5 most recent songs
        instr_dir    = os.path.join("outputs", song_dir, instrument)
        raw_path     = os.path.join(instr_dir, "02_raw_notes.json")
        stem_path    = os.path.join(instr_dir, "01_stem.wav")

        if not os.path.isfile(raw_path) or not os.path.isfile(stem_path):
            continue

        try:
            import librosa as _librosa
            with open(raw_path) as f:
                raw_notes = _json.load(f)
            stem_cqt, sr, hop = compute_stem_cqt(stem_path)
            stem_audio, _sr   = _librosa.load(stem_path, sr=sr, mono=True)
        except Exception as e:
            print(f"  [sweep-conf] Skipping {song_dir}: {e}")
            continue

        songs_checked += 1
        print(f"\n  Sweeping conf_floor on: {song_dir}  ({len(raw_notes)} raw notes)")

        # Find current conf range from notes
        confs = [n["confidence"] for n in raw_notes]
        if not confs:
            continue
        c_min = max(0.05, min(confs) - 0.01)
        c_max = min(0.60, max(confs) + 0.01)

        sweep_vals = []
        v = round(c_min, 3)
        while v <= c_max + 0.001:
            sweep_vals.append(round(v, 3))
            v = round(v + sweep_step, 3)

        song_results = []
        for thresh in sweep_vals:
            filtered = [n for n in raw_notes if n["confidence"] >= thresh]
            if not filtered:
                continue
            prec, _  = _compute_spectral_precision(filtered, stem_cqt, sr, hop)
            recall   = _compute_spectral_recall(stem_audio, sr, hop, filtered)
            f1       = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
            song_results.append((thresh, prec, recall, f1))

        results.append((song_dir, song_results))

    if not results:
        print("[sweep-conf] No valid song outputs found for this instrument.")
        return

    # Aggregate across songs
    print(f"\n{'=' * 70}")
    print(f"  CONF SWEEP — instrument={instrument}  mode={mode}  ({songs_checked} songs)")
    print(f"  {'conf':>7}  {'prec':>7}  {'recall':>7}  {'f1':>7}")
    print(f"{'=' * 70}")

    # Collect all threshold values used
    all_thresholds = sorted(set(
        thresh for _, song_results in results for thresh, *_ in song_results
    ))
    best_thresh = None
    best_f1     = -1.0

    for thresh in all_thresholds:
        prec_vals   = []
        recall_vals = []
        f1_vals     = []
        for _, song_results in results:
            row = next((r for r in song_results if r[0] == thresh), None)
            if row:
                prec_vals.append(row[1])
                recall_vals.append(row[2])
                f1_vals.append(row[3])
        if not f1_vals:
            continue
        avg_prec   = sum(prec_vals)   / len(prec_vals)
        avg_recall = sum(recall_vals) / len(recall_vals)
        avg_f1     = sum(f1_vals)     / len(f1_vals)
        marker = " ← BEST" if avg_f1 > best_f1 else ""
        if avg_f1 > best_f1:
            best_f1     = avg_f1
            best_thresh = thresh
        print(f"  {thresh:>7.3f}  {avg_prec:>7.3f}  {avg_recall:>7.3f}  {avg_f1:>7.3f}{marker}")

    print(f"{'=' * 70}")
    print(f"  Best conf_floor: {best_thresh}  (pseudo_f1={best_f1:.3f})")
    print(f"  → Set in pipeline/settings/{instrument}/cleaning.py")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Batch chroma/CQT benchmark")
    parser.add_argument("--role", choices=["lead", "rhythm"], default=None)
    parser.add_argument("--from-stage", type=int, default=1, metavar="N")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--v2", action="store_true",
                        help="Compute full v2 metric bundle (CQT, precision, recall, F1)")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze benchmark_results.json and exit (no new runs)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare last two runs per song and exit (no new runs)")
    parser.add_argument("--sweep-conf", action="store_true",
                        help="Sweep conf_floor on saved raw notes to find optimal threshold")
    parser.add_argument("--instrument", default="guitar",
                        help="Instrument for --sweep-conf (default: guitar)")
    parser.add_argument("--mode", default="clean_lead",
                        help="Mode label for --sweep-conf display (default: clean_lead)")
    args = parser.parse_args()

    results_data = load_results()

    if args.analyze:
        cmd_analyze(results_data)
        return

    if args.compare:
        cmd_compare(results_data)
        return

    if args.sweep_conf:
        cmd_sweep_conf(instrument=args.instrument, mode=args.mode)
        return

    songs = discover_songs(role_filter=args.role)
    if not songs:
        print("No matching audio files found.")
        return

    extra_args = ["--from-stage", str(args.from_stage)] if args.from_stage > 1 else []

    new_runs: list[dict] = []

    print("=" * 80)
    label = f"role={args.role}" if args.role else "all roles"
    print(f"  Discovered {len(songs)} run(s) — {label}  (type: auto-detected)")
    print("=" * 80)

    for filename, guitar_role in songs:
        print(f"\n[role={guitar_role}] {filename}", flush=True)
        entry = run_song(filename, guitar_role, extra_args, use_v2=args.v2)

        if entry is not None:
            print(f"  -> Chroma score: {entry['score']:.3f}  (detected: {entry['mode']})")
            new_runs.append(entry)
            if not args.no_save:
                results_data["runs"].append(entry)
                save_results(results_data)
        else:
            print("  -> ERROR: no score extracted")

    print_summary(new_runs if new_runs else results_data.get("runs", []))


if __name__ == "__main__":
    main()
