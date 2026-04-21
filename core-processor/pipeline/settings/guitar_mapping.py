"""
Stage 6 — Guitar Mapping  +  Melody Isolation
==============================================
Assigns each detected note to a (string, fret) position on the fretboard.

The mapper runs a dynamic-programming search over all valid string/fret
positions for each note and picks the assignment that minimises a cost
function combining:
  - distance from the current hand window centre (hand travel penalty)
  - string change penalty (encourages staying on one string for sustained notes)
  - bonus for open strings that are in the detected key

Melody isolation runs between Stage 6 (mapping) and Stage 7 (chord detection).
Notes below the per-type pitch floor are routed to the harmony track and excluded
from the melody track — this prevents low open-string resonances from appearing
in the lead melody line.

Fretboard geometry
------------------
Standard guitar: 6 strings, 21 frets (some guitars have 22 or 24).
Strings in order (low to high): E2 A2 D3 G3 B3 E4 (MIDI 40 45 50 55 59 64).
MIDI pitch of string s at fret f = open_midi[s] + f.
"""

# ── Fretboard bounds ──────────────────────────────────────────────────────────
# Highest fret number on a standard guitar.  Most guitars have 21 or 22; some
# have 24.  The mapper will not produce positions beyond this fret.
MAPPING_MAX_FRET = 21

# ── Hand window ───────────────────────────────────────────────────────────────
# Maximum fret span the fretting hand covers without shifting position.
# 4 is the classic "one finger per fret" span; reduce to 3 for a more
# conservative (easier to play) mapping.
MAPPING_HAND_SPAN = 4

# ── Bend tolerance ────────────────────────────────────────────────────────────
# Pitches above MAX_FRET × string tuning are still valid if they can be reached
# by bending from a lower fret.  Notes within TOLERANCE semitones above the
# highest fret on their string are clamped to that fret (with a bend annotation).
#
# Lead guitar (especially on 24-fret necks with high bends) needs a much wider
# tolerance.  Example: fret 24 on the high-e string = MIDI 88; a whole-step
# bend reaches MIDI 90; a 2-step bend reaches MIDI 92.  With MAX_FRET=21 that's
# 7 semitones of "virtual" range above the declared maximum.
MAPPING_BEND_TOLERANCE      = 2   # acoustic / rhythm — small headroom for slight bends
MAPPING_BEND_TOLERANCE_LEAD = 7   # lead — covers 24-fret neck + 2-step bend

# ── String stability ──────────────────────────────────────────────────────────
# Long sustained notes sound cleaner on a single string (avoids tonal colour
# changes mid-note).  Short ornamental notes can jump freely between strings.
# Notes between SHORT and LONG receive an intermediate penalty.
MAPPING_LONG_NOTE_S  = 0.30   # seconds — prefer staying on the same string
MAPPING_SHORT_NOTE_S = 0.10   # seconds — allow free string changes

# String change penalty (added to position cost; lower total = more preferred).
# These are tuning parameters: higher values lock notes onto fewer strings;
# lower values allow the mapper to explore more ergonomic positions.
MAPPING_STRING_CHANGE_PENALTY_LONG   = 2.5   # sustained note — strong preference for same string
MAPPING_STRING_CHANGE_PENALTY_MEDIUM = 1.2   # medium note — mild preference for same string
MAPPING_STRING_CHANGE_PENALTY_SHORT  = 0.4   # fast note — weak preference only

# ── Open-string bonus ─────────────────────────────────────────────────────────
# Playing a note as an open string is easier than fretting and often sounds
# brighter.  This bonus is subtracted from the position cost (making open-string
# positions more attractive) only when the open string pitch is diatonic in the
# detected key.  Non-diatonic open strings get no bonus to avoid forcing
# out-of-key choices.
MAPPING_OPEN_STRING_IN_KEY_BONUS = 2.5

# ── Hand-centre proximity weight ──────────────────────────────────────────────
# Cost contribution for each fret of distance from the hand window centre:
#   cost += distance_in_frets × MAPPING_HAND_CENTRE_WEIGHT
# Lower values allow the mapper to reach far from the window centre;
# higher values keep the hand anchored near its current position.
MAPPING_HAND_CENTRE_WEIGHT = 0.2   # kept for reference; superseded by Viterbi DP

# ── Viterbi DP transition costs ───────────────────────────────────────────────
# The Viterbi mapper scores full note sequences globally rather than making
# greedy per-note decisions.  These constants control the transition cost
# between consecutive (or simultaneous) note assignments.

# Cost per fret the hand must shift beyond the reachable span (MAPPING_HAND_SPAN).
# Example: moving from fret 5 to fret 12 = shift of 3 frets → cost = 3 × 0.8 = 2.4.
# Open strings (fret 0) are exempt — they don't anchor the hand position.
MAPPING_SHIFT_COST_PER_FRET = 0.8

# Notes whose START TIMES are within this gap (seconds) are treated as
# simultaneous (same chord strum).  Their transitions use the chord rules below
# instead of the sequential string-change penalties.
# 40ms covers a fast strum; at 240 BPM a 16th note = 62.5ms >> 40ms, so fast
# sequential runs are never mistaken for chord strums.
MAPPING_CHORD_GAP_S = 0.04

# Cost per fret added to the emission of any non-open position.
# This penalises high-fret assignments and keeps the mapping in playable range:
# fret 0 → +0, fret 5 → +1.0, fret 12 → +2.4, fret 21 → +4.2.
# Combined with MAPPING_OPEN_STRING_IN_KEY_BONUS, open in-key strings become
# strongly preferred over equivalent high-fret positions.
MAPPING_FRET_HEIGHT_WEIGHT = 0.20

# Penalty for placing two simultaneous (chord) notes on the same string.
# On a real guitar this is physically impossible: one string = one pitch at a time.
# The value is deliberately large so the DP strongly avoids this assignment.
MAPPING_SAME_STRING_CHORD_PENALTY = 10.0

# ── Bend cost ─────────────────────────────────────────────────────────────────
# When a (string, fret) position is reached by bending from fret MAX_FRET,
# the note pitch is higher than the string's natural pitch at that fret.
# Each semitone of bend adds this cost to the emission.
#
# This prevents the mapper from choosing a clamped position that requires a
# large bend on a middle string (e.g., G-string fret 21 → +6st bend to reach
# A#5) when a natural position exists on the high-e string (fret 18, no bend).
#
# Tuning: 1.5 is enough to make a 4+ semitone bend unattractive vs the
# high-e string preference penalty (STRING_PREF[1] = 5).
MAPPING_BEND_SEMITONE_COST = 1.5

# ── Context-aware fret window ─────────────────────────────────────────────────
# Before the Viterbi forward pass, each note is assigned a "context fret center":
# the median reference fret of all surrounding notes within ±WINDOW_S seconds.
# Reference fret = mean fret across all valid (string, fret) candidates for the
# note's pitch — pitch-monotonic, string-agnostic, computed from pitches alone.
#
# The context penalty is then added to emission cost for any fret assignment that
# deviates more than TOLERANCE frets from that center.  This prevents the Viterbi
# from making cheap local transitions to out-of-position frets in the middle of a
# clearly high- or low-position passage.
#
# Open strings (fret=0) are exempt: guitarists use them in any position.
#
# Tuning:
#   More locking:       raise MAPPING_CONTEXT_PENALTY_PER_FRET (try 0.7)
#   Shorter window:     lower MAPPING_CONTEXT_WINDOW_S (try 1.5)
#   Wider free band:    raise MAPPING_CONTEXT_TOLERANCE (try 5)
#   Disable entirely:   set MAPPING_CONTEXT_PENALTY_PER_FRET = 0.0
MAPPING_CONTEXT_WINDOW_S           = 2.0   # half-width of context window in seconds
MAPPING_CONTEXT_TOLERANCE          = 4     # frets around center with no penalty
MAPPING_CONTEXT_PENALTY_PER_FRET   = 0.5   # cost per fret beyond the tolerance band
MAPPING_CONTEXT_OPEN_STRING_EXEMPT = True  # fret=0 skips the context penalty


# ── Melody isolation ──────────────────────────────────────────────────────────
# Applied between Stage 6 (mapping) and Stage 7 (chord detection).
# Notes below the min_pitch floor are excluded from the melody track and
# forwarded to the harmony/chord track instead.
#
# Lead guitar: exclude MIDI < 50 (D3) to avoid low-E string ring and
# accidental bass-note bleed appearing in the melody line.
# Acoustic and rhythm guitar: no floor — these styles often include bass
# melody notes or deliberately use low strings as melodic elements.
MELODY_MIN_PITCH = {
    "lead":     50,   # D3 — filters low-string bleed from melody
    "acoustic":  0,   # no floor — bass notes are legitimate in fingerpicking
    "rhythm":    0,   # no floor — chord bass notes are handled by chord detection
}
