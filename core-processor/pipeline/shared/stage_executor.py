"""
StageExecutor — eliminates per-instrument boilerplate in pipeline runners.

Each instrument pipeline repeats the same pattern for every stage:
    if from_stage <= N:
        result = run_fn(...)
    else:
        try:
            result = load_fn()
        except FileNotFoundError:
            result = run_fn(...)

StageExecutor encapsulates that pattern and records per-stage wall-clock time.
"""

import time


class StageExecutor:
    def __init__(self, from_stage: int):
        self.from_stage = from_stage
        self.timings: dict[str, float] = {}   # stage_label -> seconds

    def should_run(self, stage_n: int) -> bool:
        return self.from_stage <= stage_n

    def run_or_load(self, stage_n: int, run_fn, load_fn, skip_msg: str = ""):
        """
        Run run_fn() if from_stage <= stage_n, otherwise try load_fn().
        Falls back to run_fn() if load_fn raises FileNotFoundError.
        Records elapsed wall-clock time and prints it after every executed stage.
        """
        label = str(stage_n)
        if self.should_run(stage_n):
            t0 = time.perf_counter()
            result = run_fn()
            elapsed = time.perf_counter() - t0
            self.timings[label] = elapsed
            print(f"[Timing] Stage {stage_n}: {elapsed:.1f}s")
            return result
        if skip_msg:
            print(skip_msg)
        try:
            return load_fn()
        except FileNotFoundError:
            print(f"[Stage {stage_n}] Saved output not found — re-running.")
            t0 = time.perf_counter()
            result = run_fn()
            elapsed = time.perf_counter() - t0
            self.timings[label] = elapsed
            print(f"[Timing] Stage {stage_n}: {elapsed:.1f}s")
            return result

    def print_summary(self, instrument: str = "") -> None:
        if not self.timings:
            return
        total = sum(self.timings.values())
        tag = f" ({instrument})" if instrument else ""
        print(f"[Timing] Stage summary{tag}:")
        for label, secs in sorted(self.timings.items(), key=lambda x: (len(x[0]), x[0])):
            bar = "#" * max(1, int(secs / total * 30))
            print(f"[Timing]   Stage {label:>3}: {secs:6.1f}s  {bar}")
        print(f"[Timing]   TOTAL  : {total:6.1f}s")


def parse_time_range(args) -> tuple[float | None, float | None]:
    """
    Parse --start / --end args to float seconds.
    Returns (start_s, end_s), either of which may be None.
    """
    def _parse(s: str) -> float:
        s = s.strip()
        if ":" in s:
            parts = s.split(":")
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)

    start_s = _parse(args.start) if getattr(args, "start", None) else None
    end_s   = _parse(args.end)   if getattr(args, "end",   None) else None
    return start_s, end_s
