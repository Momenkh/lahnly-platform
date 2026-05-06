"""
Stage 8 (Bass): Tab Generation
Input:  mapped bass notes [{string, fret, pitch, start, duration}]
Output: ASCII 4-string bass tab, saved to outputs/08_bass_tabs.txt

Format — 4-line blocks, each covering COLS_PER_BLOCK 16th notes:

  [0:00]
  G |---------------------|
  D |--0-------3---5---3--|
  A |--2---5--------------|
  E |---------------------|

String names (low to high): E A D G  (4-string standard tuning)
"""

import os
from pipeline.config import get_instrument_dir
from pipeline.settings import TAB_COLS_PER_BLOCK, TAB_DEFAULT_SECONDS_PER_COL

STRING_NAMES = {4: "E", 3: "A", 2: "D", 1: "G"}


def generate_tabs_bass(
    notes: list[dict],
    tempo_info: dict | None = None,
    save: bool = True,
) -> str:
    if not notes:
        return "# No notes to display"

    spc     = tempo_info["subdivision_s"] if tempo_info else TAB_DEFAULT_SECONDS_PER_COL
    bpm_str = f"  BPM: {tempo_info['bpm']:.1f}" if tempo_info else ""
    print(f"[Stage 8B] Generating bass tabs for {len(notes)} notes{bpm_str}...")

    max_start  = max(n["start"] for n in notes)
    total_cols = int(max_start / spc) + TAB_COLS_PER_BLOCK

    # grid[string][col] = (fret_str, duration_cols)
    grid: dict[int, dict[int, tuple[str, int]]] = {s: {} for s in range(1, 5)}
    collisions = 0

    for note in sorted(notes, key=lambda n: n["start"]):
        col      = int(note["start"] / spc)
        s        = note["string"]
        fret_str = str(note["fret"])
        dur_cols = max(1, round(note["duration"] / spc))
        while col in grid[s]:
            col += 1
            collisions += 1
        grid[s][col] = (fret_str, dur_cols)

    if collisions:
        print(f"[Stage 8B] Resolved {collisions} column collisions (nudged right)")

    blocks = []
    col_start = 0
    while col_start < total_cols:
        col_end = min(col_start + TAB_COLS_PER_BLOCK, total_cols)

        has_content = any(
            any(col in grid[s] for col in range(col_start, col_end))
            for s in range(1, 5)
        )
        if not has_content:
            col_start = col_end
            continue

        has_high_fret = any(
            int(grid[s][col][0]) >= 10
            for s in range(1, 5)
            for col in range(col_start, col_end)
            if col in grid[s]
        )
        col_w = 3 if has_high_fret else 2

        t_start = col_start * spc
        minutes = int(t_start // 60)
        seconds = int(t_start % 60)

        lines = [f"[{minutes}:{seconds:02d}]"]
        # Render strings high-to-low (G at top, E at bottom)
        for string_num in range(1, 5):
            cells: list[str] = []
            col = col_start
            while col < col_end:
                if col in grid[string_num]:
                    fret_str, dur_cols = grid[string_num][col]
                    label     = fret_str.ljust(col_w, "-")
                    trail_len = max(0, dur_cols - 1) * col_w
                    trail     = "-" * trail_len
                    combined  = label + trail
                    remaining = (col_end - col) * col_w
                    combined  = combined[:remaining].ljust(remaining, "-")
                    cells.append(combined)
                    col += max(1, dur_cols)
                else:
                    cells.append("-" * col_w)
                    col += 1
            lines.append(f"{STRING_NAMES[string_num]} |{''.join(cells)}|")

        blocks.append("\n".join(lines))
        col_start = col_end

    tab_str = "\n\n".join(blocks)

    if save:
        out_path = os.path.join(get_instrument_dir("bass"), "08_tabs.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Bass Tabs\n")
            f.write(f"# {len(notes)} notes  |  {len(blocks)} blocks\n\n")
            f.write(tab_str)
        print(f"[Stage 8B] Saved -> {out_path}  ({len(blocks)} blocks)")

    return tab_str


def load_tabs_bass() -> str:
    path = os.path.join(get_instrument_dir("bass"), "08_tabs.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()
