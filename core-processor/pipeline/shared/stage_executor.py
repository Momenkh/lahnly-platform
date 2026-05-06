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

StageExecutor encapsulates that pattern.
"""


class StageExecutor:
    def __init__(self, from_stage: int):
        self.from_stage = from_stage

    def should_run(self, stage_n: int) -> bool:
        return self.from_stage <= stage_n

    def run_or_load(self, stage_n: int, run_fn, load_fn, skip_msg: str = ""):
        """
        Run run_fn() if from_stage <= stage_n, otherwise try load_fn().
        Falls back to run_fn() if load_fn raises FileNotFoundError.
        """
        if self.should_run(stage_n):
            return run_fn()
        if skip_msg:
            print(skip_msg)
        try:
            return load_fn()
        except FileNotFoundError:
            print(f"[Stage {stage_n}] Saved output not found — re-running.")
            return run_fn()


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
