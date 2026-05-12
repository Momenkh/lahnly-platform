"""
run_all.py  — run every song × every instrument and report timings.

Each instrument runs in its OWN subprocess so GPU/CPU models are fully
unloaded between instruments — prevents OOM on long sessions.

Guitar always runs first per song: it triggers htdemucs_6s which caches
all 6 stems on disk.  Bass/piano/vocals/drums find their stems already
saved and skip Stage 1 entirely.

Usage:
    python run_all.py                   # full pipeline (sep + pitch + clean + ...)
    python run_all.py --from-stage 2    # skip separation (reuse saved stems)
    python run_all.py --from-stage 3    # skip sep + pitch (reuse raw notes)
    python run_all.py --no-viz --no-play
    python run_all.py --songs song1.mp3 song2.mp3

Output: per-stage timings parsed from [Timing] lines, per-instrument wall-clock,
        per-song total, and a final summary table.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}
INSTRUMENTS      = ["guitar", "bass", "piano", "vocals", "drums"]

RE_TIMING_STAGE = re.compile(r"\[Timing\]\s+Stage\s+(\S+):\s+([\d.]+)s")
RE_TIMING_TOTAL = re.compile(r"\[Timing\]\s+TOTAL\s*:\s*([\d.]+)s")
RE_NOTES_FINAL  = re.compile(r"\[Stage \w+\] Final: (\d+) notes")


def discover_songs() -> list[str]:
    seen, songs = set(), []
    for ext in AUDIO_EXTENSIONS:
        for f in glob.glob(f"*{ext}") + glob.glob(f"*{ext.upper()}"):
            if f not in seen:
                seen.add(f)
                songs.append(f)
    return sorted(songs)


def run_instrument(audio_file: str, instrument: str, extra_args: list[str]) -> dict:
    """
    Run one instrument pipeline in its own subprocess.
    Returns per-stage timing dict, total wall-clock, exit code, and note count.
    """
    cmd = [sys.executable, "main.py", audio_file, "--instrument", instrument] + extra_args

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stage_timings: dict[str, float] = {}
    total_s_reported = None
    notes = None
    lines_collected = []

    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        print(f"  [{instrument.upper():<6}] {line}")
        lines_collected.append(line)

        m = RE_TIMING_STAGE.search(line)
        if m:
            stage_timings[m.group(1)] = float(m.group(2))

        m = RE_TIMING_TOTAL.search(line)
        if m:
            total_s_reported = float(m.group(1))

        m = RE_NOTES_FINAL.search(line)
        if m:
            notes = int(m.group(1))

    proc.wait()
    wall_s = time.perf_counter() - t0

    return {
        "instrument":      instrument,
        "exit_code":       proc.returncode,
        "wall_s":          wall_s,
        "total_s":         total_s_reported or wall_s,
        "stage_timings":   stage_timings,
        "notes":           notes,
    }


def run_song(audio_file: str, extra_args: list[str], instruments: list[str] | None = None) -> dict:
    """
    Run all instruments for one song, each in its own subprocess.
    Guitar goes first so htdemucs_6s caches all 6 stems; the rest skip Stage 1.
    """
    print(f"\n{'='*70}")
    print(f"  SONG: {audio_file}")
    print(f"  FLAGS: {' '.join(extra_args) or '(none)'}")
    print(f"{'='*70}")

    t_song_start = time.perf_counter()
    instrument_data: dict[str, dict] = {}
    run_list = instruments if instruments is not None else INSTRUMENTS

    for inst in run_list:
        print(f"\n  ── {inst.upper()} ──")
        result = run_instrument(audio_file, inst, extra_args)
        instrument_data[inst] = result
        status = "OK" if result["exit_code"] == 0 else f"FAIL({result['exit_code']})"
        print(f"  ── {inst.upper()} done: {result['wall_s']:.0f}s  [{status}]")

    song_total_s = time.perf_counter() - t_song_start

    return {
        "song":         audio_file,
        "song_total_s": song_total_s,
        "instruments":  instrument_data,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  TIMING REPORT")
    print("=" * 80)

    for r in results:
        any_fail = any(
            r["instruments"].get(i, {}).get("exit_code", 0) != 0
            for i in INSTRUMENTS
        )
        status = "PARTIAL FAIL" if any_fail else "OK"
        print(f"\n  {r['song']}  [{status}]  total={r['song_total_s']:.0f}s")

        for inst in INSTRUMENTS:
            data = r["instruments"].get(inst)
            if not data:
                continue
            ec      = data.get("exit_code", 0)
            total   = data.get("total_s")
            notes   = data.get("notes")
            stages  = data.get("stage_timings", {})
            inst_ok = "OK" if ec == 0 else f"FAIL({ec})"
            total_s = f"{total:.1f}s" if total else "—"
            notes_s = f"{notes} notes" if notes is not None else "—"
            print(f"    {inst:<8} {total_s:>7}  ({notes_s})  [{inst_ok}]")
            if stages and total:
                for stage_id, secs in sorted(stages.items(), key=lambda x: (len(x[0]), x[0])):
                    pct = f"{secs/total*100:.0f}%"
                    print(f"             Stage {stage_id:>3}: {secs:5.1f}s  {pct}")

    # Summary table
    print("\n" + "-" * 80)
    print(f"  {'Song':<45} {'Total':>7}  {'Status'}")
    print("-" * 80)
    for r in results:
        any_fail = any(
            r["instruments"].get(i, {}).get("exit_code", 0) != 0
            for i in INSTRUMENTS
        )
        status = "FAIL" if any_fail else "OK"
        print(f"  {r['song'][:45]:<45} {r['song_total_s']:>6.0f}s  {status}")

    all_times = [r["song_total_s"] for r in results]
    if all_times:
        print("-" * 80)
        print(f"  Total wall-clock: {sum(all_times):.0f}s  "
              f"({sum(all_times)/60:.1f} min)  |  avg per song: {sum(all_times)/len(all_times):.0f}s")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Run all songs × all instruments with timing")
    ap.add_argument("--from-stage", type=int, default=1, metavar="N",
                    help="Resume from stage N (default: 1 = full run)")
    ap.add_argument("--no-play",  action="store_true")
    ap.add_argument("--no-viz",   action="store_true")
    ap.add_argument("--score",    action="store_true")
    ap.add_argument("--songs",    nargs="*", metavar="SONG",
                    help="Specific audio files to run (default: all discovered)")
    ap.add_argument("--instruments", nargs="*", metavar="INST",
                    choices=INSTRUMENTS + [None],
                    help="Instruments to run (default: all). E.g. --instruments guitar bass")
    args = ap.parse_args()

    songs = args.songs or discover_songs()
    if not songs:
        print("No audio files found in current directory.")
        sys.exit(1)

    extra = []
    if args.from_stage > 1:
        extra += ["--from-stage", str(args.from_stage)]
    if args.no_play:  extra.append("--no-play")
    if args.no_viz:   extra.append("--no-viz")
    if args.score:    extra.append("--score")

    instruments_to_run = args.instruments or INSTRUMENTS

    print(f"\nFound {len(songs)} song(s). Each instrument runs in its own subprocess.")
    print(f"Instruments: {instruments_to_run}")
    print(f"Extra flags: {extra or '(none)'}")

    results = []
    for song in songs:
        if not os.path.isfile(song):
            print(f"[skip] {song} not found")
            continue
        result = run_song(song, extra, instruments_to_run)
        results.append(result)

    print_report(results)


if __name__ == "__main__":
    main()
