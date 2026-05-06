"""
Stage 5 — Key Detection
========================
Krumhansl-Schmuckler (KS) key profiles and scale-selection bias.

The KS algorithm correlates a chroma histogram (pitch-class energy distribution)
built from the detected notes against key templates and picks the highest match.

Template pool
-------------
Western:  24 templates — 12 major + 12 minor, one per root (C through B).
Maqam:    36 templates — 3 maqam modes × 12 roots.

  Maqam Hijaz  — intervals 1 b2 3 4 5 b6 b7.  Defining feature: augmented 2nd
                  between b2 and 3 (e.g. Eb→F# on root D).  Most common maqam
                  in Arabic pop; detected in Jadal's Galbi Metlel Ward.
  Maqam Kurd   — intervals 1 b2 b3 4 5 b6 b7.  Equivalent to Phrygian in Western
                  theory; common in Arabic and Turkish music.
  Maqam Saba   — intervals 1 b2 b3 b4 5 b6 b7.  Distinctive chromatic cluster
                  (b2-b3-b4 in three semitone steps); the tritone from root to b4
                  is unique to Saba.

Adding more maqam modes: define a 12-element root-position weight vector in
music_theory.py (see KS_MAQAM_HIJAZ for the pattern), add it to MAQAM_PROFILES,
add its intervals to SCALES, and add its label to MAQAM_LABELS.  No other
changes needed — _detect_key iterates MAQAM_PROFILES automatically.

Western detection is unaffected by the extra templates: a Western song's pitch
histogram still best-matches its Western template because the maqam profiles have
distinctly different interval structures (augmented 2nds, tritone steps).

Scale-selection bias
--------------------
After the winning key template is chosen, a secondary pass picks the best-fitting
subscale (e.g. pentatonic minor vs natural minor).  This pass is skipped for
maqam detections — the maqam mode is already the scale.

For Western keys, guitar music is heavily skewed toward minor pentatonic / natural
minor for lead playing.  A small bonus for pentatonic minor nudges the algorithm
to prefer minor readings when scores are close, reducing incorrect major-key
labels on blues and rock recordings.
"""

# Score bonus added to the Krumhansl-Schmuckler correlation for the pentatonic
# minor key interpretation during scale selection.
#
# Applied ONLY when the top-2 scale candidates are within KEY_PENTA_BIAS_GATE
# of each other (score gap ≤ gate). This prevents a fixed bias from overriding
# a clear major-key reading; it only tips close calls toward minor.
#
# KEY_PENTATONIC_MINOR_BIAS = 0.01: meaningful for near-ties only.
# KEY_PENTA_BIAS_GATE       = 0.03: the gap at which the bias kicks in.
KEY_PENTATONIC_MINOR_BIAS = 0.01
KEY_PENTA_BIAS_GATE       = 0.03

# Maqam detection guards — two conditions that must BOTH be met for a maqam
# template to override the best Western key.
#
# 1. Minimum note count: maqam detection requires a dense enough histogram to
#    be statistically reliable.  With few notes (e.g. Thunderstruck: 31 notes)
#    the pitch histogram is sparse and a maqam template can correlate higher by
#    chance — especially Hijaz, whose augmented-2nd profile overlaps with many
#    pentatonic patterns.  Western key is forced when note count is below this.
#
# 2. Minimum correlation gap: even with enough notes, maqam only wins if its
#    best score exceeds the best Western score by at least this margin.  This
#    prevents borderline detections when the song sits between two systems.
#
# Observed gaps:
#   Galbi Metlel Ward (405 notes, genuine maqam): D Kurd leads G minor by ~0.05
#   Thunderstruck (31 notes, Western):            B Hijaz leads B major by ~0.09
#   → Note count guard eliminates the Thunderstruck false positive independently
#     of gap, so the gap threshold can be kept low for genuine maqam songs.
KEY_MAQAM_MIN_NOTES = 80    # fewer notes than this → force Western key
KEY_MAQAM_MIN_GAP   = 0.04  # maqam must beat best Western score by this much

# Capo detection ─────────────────────────────────────────────────────────────
# After key detection, test capo positions 1..CAPO_SEARCH_MAX to check whether
# a capo makes the key more guitar-friendly (more open strings in the scale).
#
# Open strings: E(4), A(9), D(2), G(7), B(11) — five pitch classes.
# "Friendly" key = at least CAPO_MIN_FRIENDLY_STRINGS of these five in scale.
# G major, C major, D major, Am, Em all score 5/5; A major, F major score 4/5.
# Keys with scores < 4 (e.g. Ab, Db, Gb) are "guitar-unfriendly" and likely
# played with a capo to reach a friendlier position.
CAPO_SEARCH_MAX           = 5   # maximum capo fret to test (5 covers most songs)
CAPO_MIN_FRIENDLY_STRINGS = 4   # minimum open-string count to call a key "friendly"

# Per-segment key detection ───────────────────────────────────────────────────
# The full-song key histogram conflates notes from different sections when a
# song modulates.  Segmentation splits the note stream at silence gaps and
# detects the key independently per section, so the stage 5b feedback filter
# uses the correct key context for each part of the song.
#
# KEY_SEGMENT_MIN_GAP_S     — silence longer than this triggers a new segment.
# KEY_SEGMENT_MIN_NOTES     — segments with fewer notes fall back to the global key.
# KEY_SEGMENT_MIN_DURATION_S — segments shorter than this are merged with a neighbor.
KEY_SEGMENT_MIN_GAP_S      = 2.0   # seconds of silence → new segment boundary
KEY_SEGMENT_MIN_NOTES      = 16    # fewer notes → inherit from nearest reliable segment
KEY_SEGMENT_MIN_DURATION_S = 15.0  # merge shorter segments into a neighbor
