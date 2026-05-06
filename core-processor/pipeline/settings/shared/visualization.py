"""
Stages 10 & 11 — Visualization
================================
Fretboard diagram (Stage 10) and chord-box sheet (Stage 11) appearance settings.

Stage 10 — Fretboard diagram
-----------------------------
Detected notes are drawn as coloured "pills" on a time-vs-fretboard grid.
Each pill spans the note's duration horizontally; its vertical position indicates
which string the note is mapped to.  Pill shade encodes confidence: dark pills
are high-confidence notes, light grey pills are uncertain detections.

The diagram is split into multiple rows ("sections") so long recordings don't
produce an impossibly wide single image.  Section length adapts to the total
recording duration.

Stage 11 — Chord sheet
-----------------------
Detected chords are shown as standard guitar chord-box diagrams, with a
chord-symbol progression line below.  Multiple chord diagrams are laid out
in a grid.
"""

# ── Fretboard diagram (Stage 10) ─────────────────────────────────────────────

# Pill height expressed in string-spacing units.
# 1.0 would fill the full gap between strings; 0.55 leaves visible spacing.
VIZ_NOTE_HEIGHT_STRING_UNITS = 0.55

# Minimum pill width in seconds.  Very short notes (< 0.18s) are drawn at
# this width so the fret number label is readable.
VIZ_MIN_NOTE_WIDTH_S = 0.18

# Height of each fretboard row in the output image (inches).
# Increase for a taller diagram; decrease to fit more rows per page.
VIZ_ROW_HEIGHT_INCHES = 2.8

# Output image resolution.  140 DPI gives a good balance between file size
# and readability when viewed on screen or printed at full size.
VIZ_DPI = 140

# ── Adaptive section length ───────────────────────────────────────────────────
# The diagram is split into rows.  Ideal section length is chosen so the total
# number of rows is close to VIZ_SECTION_TARGET_ROWS, clamped between MIN and MAX.
#   section_s = clamp(total_duration / VIZ_SECTION_TARGET_ROWS, MIN, MAX)
VIZ_SECTION_TARGET_ROWS = 16    # aim for roughly this many rows
VIZ_SECTION_MIN_S       = 10.0  # never less than 10 seconds per row (avoids tiny rows)
VIZ_SECTION_MAX_S       = 30.0  # never more than 30 seconds per row (avoids wide rows)

# ── Pill colour encoding ──────────────────────────────────────────────────────
# Pill RGB channel value (same for R, G, B → greyscale pill).
# HIGH = dark component → dark pill = high confidence.
# LOW  = light component → light pill = low confidence.
# The actual shade is interpolated between these two values based on confidence.
VIZ_PILL_COLOR_HIGH = 0x22   # near-black for confident notes
VIZ_PILL_COLOR_LOW  = 0xaa   # mid-grey for uncertain notes

# ── Pill text (fret number label) ─────────────────────────────────────────────
# Fret number is printed inside each pill.  Text colour switches between white
# and black depending on the pill shade (so it stays readable on both dark and
# light backgrounds).
VIZ_TEXT_LIGHT_CONF_THRESHOLD = 0.4   # above this confidence → white text on dark pill

# Font size adapts to pill width: wider pills get a larger label.
VIZ_FONT_SIZE_WIDE_NOTE   = 6.5   # font size (pt) for pills wider than threshold
VIZ_FONT_SIZE_NARROW_NOTE = 5.5   # font size (pt) for narrow pills
VIZ_WIDE_NOTE_THRESHOLD_S = 0.3   # seconds — pills wider than this use the larger font


# ── Chord sheet (Stage 11) ────────────────────────────────────────────────────

# Fret rows to display in each chord box diagram.
# 5 covers the most common open and barre chord shapes; increase to 6 or 7
# for extended and jazz voicings that span more frets.
CHORD_SHEET_DISPLAY_FRETS = 5

# Number of chord diagrams per row in the grid layout.
CHORD_SHEET_COLS_PER_ROW = 6

# Number of chord symbols per line in the progression section below the diagrams.
CHORD_SHEET_PROGRESSION_PER_LINE = 8

# Minimum number of strings at the same fret for a barre indicator bar to be drawn.
# 4 strings: draws a bar for partial barres (common in F-shape chords).
# Increase to 5 or 6 for only full barre chords.
CHORD_SHEET_BARRE_MIN_STRINGS = 4

# Output image DPI (matches VIZ_DPI for a consistent look when combining outputs).
CHORD_SHEET_DPI = 140
