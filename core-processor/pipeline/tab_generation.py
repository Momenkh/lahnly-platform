"""
Stage 8: Tab Generation
Input:  solo notes only (chord notes are excluded — shown separately on chord sheet)
Output: ASCII guitar tab, saved to outputs/08_tabs.txt

Format — 6-line blocks, each covering COLS_PER_BLOCK 16th notes:

  [0:00]
  e |--0-------3---5---3--|
  B |---------------------|
  G |---------------------|
  D |---------------------|
  A |--2---5--------------|
  E |---------------------|

Rules:
  - Each column = subdivision_s seconds (from tempo detection) or fallback
  - Note duration shown as trailing dashes extending across occupied columns
  - Column width is 2 chars for frets 0–9, 3 chars for frets 10+ in that block
  - Notes on the same string in the same column: earlier note wins,
    later note shifts one column right
  - Empty blocks are skipped
"""

import os
from pipeline.config import get_outputs_dir
from pipeline.settings import TAB_COLS_PER_BLOCK, TAB_DEFAULT_SECONDS_PER_COL

STRING_NAMES = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}


def generate_tabs(
    solo_notes: list[dict],
    chord_groups: list[dict] | None = None,   # kept for signature compat, not used
    tempo_info: dict | None = None,
    save: bool = True,
) -> str:
    if not solo_notes:
        return "# No notes to display"

    spc     = tempo_info["subdivision_s"] if tempo_info else TAB_DEFAULT_SECONDS_PER_COL
    bpm_str = f"  BPM: {tempo_info['bpm']:.1f}" if tempo_info else ""
    print(f"[Stage 8] Generating tabs for {len(solo_notes)} solo notes{bpm_str}...")

    max_start  = max(n["start"] for n in solo_notes)
    total_cols = int(max_start / spc) + TAB_COLS_PER_BLOCK

    # ── Build note grid ───────────────────────────────────────────────────────
    # grid[string][col] = (fret_str, duration_cols)
    grid: dict[int, dict[int, tuple[str, int]]] = {s: {} for s in range(1, 7)}
    collisions = 0

    for note in sorted(solo_notes, key=lambda n: n["start"]):
        col      = int(note["start"] / spc)
        s        = note["string"]
        fret_str = str(note["fret"])
        dur_cols = max(1, round(note["duration"] / spc))
        while col in grid[s]:
            col += 1
            collisions += 1
        grid[s][col] = (fret_str, dur_cols)

    if collisions:
        print(f"[Stage 8] Resolved {collisions} column collisions (nudged right)")

    # ── Render blocks, skip empty ones ────────────────────────────────────────
    blocks = []
    col_start = 0
    while col_start < total_cols:
        col_end = min(col_start + TAB_COLS_PER_BLOCK, total_cols)

        has_content = any(
            any(col in grid[s] for col in range(col_start, col_end))
            for s in range(1, 7)
        )
        if not has_content:
            col_start = col_end
            continue

        # Dynamic column width: 3 if any fret ≥ 10, else 2
        has_high_fret = any(
            int(grid[s][col][0]) >= 10
            for s in range(1, 7)
            for col in range(col_start, col_end)
            if col in grid[s]
        )
        col_w = 3 if has_high_fret else 2

        t_start = col_start * spc
        minutes = int(t_start // 60)
        seconds = int(t_start % 60)

        lines = [f"[{minutes}:{seconds:02d}]"]
        for string_num in range(1, 7):
            cells: list[str] = []
            col = col_start
            while col < col_end:
                if col in grid[string_num]:
                    fret_str, dur_cols = grid[string_num][col]
                    # Fret label padded to col_w, then trailing dashes for remaining duration
                    label     = fret_str.ljust(col_w, "-")
                    trail_len = max(0, dur_cols - 1) * col_w
                    trail     = "-" * trail_len
                    combined  = label + trail
                    # Clip to remaining columns in block
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

    print("[Stage 8] Tab preview (first block):")
    for line in (blocks[0].split("\n") if blocks else []):
        print(" ", line)
    if len(blocks) > 1:
        print(f"  ... ({len(blocks)} blocks total)")

    if save:
        os.makedirs(get_outputs_dir(), exist_ok=True)
        out_path = os.path.join(get_outputs_dir(), "08_tabs.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Guitar Tabs (melody/solo notes only)\n")
            f.write(f"# {len(solo_notes)} notes  |  {len(blocks)} blocks\n\n")
            f.write(tab_str)
        print(f"[Stage 8] Saved -> {out_path}  ({len(blocks)} blocks)")

    return tab_str


def load_tabs() -> str:
    path = os.path.join(get_outputs_dir(), "08_tabs.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()
