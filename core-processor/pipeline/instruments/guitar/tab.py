"""
Stage 8: Tab Generation
Input:  mapped notes [{string, fret, pitch, start, duration}]
        For lead guitar  — melody/solo notes only
        For rhythm/acoustic — all mapped notes (shows chord columns)
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
  - Notes on different strings share the same column (chord block)
  - Empty blocks are skipped
"""

import os
from pipeline.config import get_instrument_dir
from pipeline.settings import TAB_COLS_PER_BLOCK, TAB_DEFAULT_SECONDS_PER_COL, QUANTIZATION_SUBDIVISION

STRING_NAMES = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}


def generate_tabs(
    notes: list[dict],
    chord_groups: list[dict] | None = None,   # kept for signature compat, not used
    tempo_info: dict | None = None,
    capo_fret: int = 0,
    save: bool = True,
) -> str:
    if not notes:
        return "# No notes to display"

    spc     = tempo_info["subdivision_s"] if tempo_info else TAB_DEFAULT_SECONDS_PER_COL
    bpm_str = f"  BPM: {tempo_info['bpm']:.1f}" if tempo_info else ""
    print(f"[Stage 8] Generating tabs for {len(notes)} notes{bpm_str}...")

    # Bar line spacing: number of 16th-note columns per bar.
    # Formula: (ts_num / ts_den) × 4 quarter notes/whole × QUANTIZATION_SUBDIVISION subdivisions/quarter
    # 4/4 → 16, 3/4 → 12, 2/4 → 8, 6/8 → 12 (6 eighths = 3 quarters on quarter-note grid)
    ts_num   = tempo_info.get("time_sig_num", 4) if tempo_info else 4
    ts_den   = tempo_info.get("time_sig_den", 4) if tempo_info else 4
    bar_cols = (ts_num * 4 * QUANTIZATION_SUBDIVISION) // ts_den   # cols per bar

    max_start  = max(n["start"] for n in notes)
    total_cols = int(max_start / spc) + TAB_COLS_PER_BLOCK

    # ── Build note grid ───────────────────────────────────────────────────────
    # grid[string][col] = (fret_str, duration_cols)
    # Notes on different strings share the same column (chord block).
    # Same-string collision: nudge right (physically impossible on real guitar).
    grid: dict[int, dict[int, tuple[str, int]]] = {s: {} for s in range(1, 7)}
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
                # Insert bar line at bar boundary (not at the outer left delimiter)
                if col > col_start and col % bar_cols == 0:
                    cells.append("|")

                if col in grid[string_num]:
                    fret_str, dur_cols = grid[string_num][col]
                    end_col = min(col_end, col + max(1, dur_cols))
                    # First column: fret label (padded/clipped to col_w)
                    cells.append(fret_str.ljust(col_w, "-")[:col_w])
                    # Remaining columns in the duration span — insert bar lines as needed
                    for c in range(col + 1, end_col):
                        if c % bar_cols == 0:
                            cells.append("|")
                        cells.append("-" * col_w)
                    col = end_col
                else:
                    cells.append("-" * col_w)
                    col += 1

            lines.append(f"{STRING_NAMES[string_num]} |{''.join(cells)}|")

        blocks.append("\n".join(lines))
        col_start = col_end

    tab_str = "\n\n".join(blocks)

    if save:
        out_path = os.path.join(get_instrument_dir("guitar"), "08_tabs.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Guitar Tabs\n")
            f.write(f"# {len(notes)} notes  |  {len(blocks)} blocks  |  Time: {ts_num}/{ts_den}\n")
            if capo_fret:
                f.write(f"# Capo: {capo_fret}\n")
            f.write("\n")
            f.write(tab_str)
        extras = []
        if capo_fret:
            extras.append(f"Capo {capo_fret}")
        extras.append(f"Time {ts_num}/{ts_den}")
        print(f"[Stage 8] Saved -> {out_path}  ({len(blocks)} blocks)"
              + (f"  [{', '.join(extras)}]" if extras else ""))

    return tab_str


def load_tabs() -> str:
    path = os.path.join(get_instrument_dir("guitar"), "08_tabs.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()
