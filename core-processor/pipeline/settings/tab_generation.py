"""
Stage 8 — Tab Generation
=========================
ASCII guitar tab layout settings.

Tab is rendered as a grid of columns, one column per 16th-note subdivision.
Columns are grouped into "blocks" (bars); each block spans TAB_COLS_PER_BLOCK
columns, printed as a horizontal band of six string lines.

When BPM is unknown, column width falls back to TAB_DEFAULT_SECONDS_PER_COL
(equivalent to a 16th note at 120 BPM) so the tab still renders at a
reasonable density.
"""

# Columns per tab block.
# 32 columns at 16th-note resolution = 2 bars of 4/4.
# Reduce to 16 for a wider, less dense layout (1 bar per block).
# Increase to 64 to fit more bars per printed line.
TAB_COLS_PER_BLOCK = 32

# Fallback column width in seconds when no tempo info is available.
# 0.125s = 1/8 of a second = 16th note at 120 BPM.
# Adjust if your recordings are typically faster (use 0.0625 for 240 BPM feel)
# or slower (use 0.25 for 60 BPM feel).
TAB_DEFAULT_SECONDS_PER_COL = 0.125
