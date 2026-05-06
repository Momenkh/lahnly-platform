"""
Stage 4 — Quantization
=======================
Tempo (BPM) detection and note grid-snap settings.

basic-pitch gives us notes with continuous start/end times.  Quantization
snaps those times to the nearest subdivision of the detected beat grid so
that the resulting tab aligns to a musical grid (16th notes by default).

Two-phase BPM detection
-----------------------
1. Full-song pass  — uses all note onsets to find a global tempo
2. Dense-window pass — focuses on the busiest QUANTIZATION_DENSE_WINDOW_S
   seconds of the song where onsets are most frequent, which tends to give
   a cleaner beat estimate in recordings with long sparse intro/outro sections.
The two estimates are cross-validated; if they disagree, the denser window wins.
"""

# Grid resolution: how many subdivisions per beat to snap to.
# 4 = 16th notes (standard; good for rock, pop, and most guitar styles).
# Raise to 6 for triplet-heavy material; raise to 8 for 32nd-note runs.
QUANTIZATION_SUBDIVISION = 4

# Snap tolerance: a note onset is snapped to the nearest grid point if it lies
# within this fraction of one subdivision interval from the grid point.
# 0.35 means ±35% of a 16th-note duration — generous enough to catch
# expressive timing without pulling clearly off-beat notes to the wrong grid.
QUANTIZATION_SNAP_TOLERANCE_FRAC = 0.35

# Window used for the dense-onset BPM detection pass (seconds).
# A longer window captures more onsets but may straddle tempo changes.
# 60 seconds is a good balance for most song-length recordings.
QUANTIZATION_DENSE_WINDOW_S = 60.0

# BPM search range.  Tempos outside this range are rejected and the fallback
# BPM estimate is used.  Most guitar music falls between 60 and 200 BPM;
# extending below 40 tends to pick up half-time interpretations and above 240
# tends to pick up double-time interpretations of fast passages.
QUANTIZATION_BPM_MIN = 40.0
QUANTIZATION_BPM_MAX = 240.0

# Stability warning threshold: if the standard deviation of inter-beat intervals
# exceeds this many milliseconds the pipeline emits a warning.  High instability
# means the tempo estimate is uncertain and tab alignment may be poor.
QUANTIZATION_INSTABILITY_WARN_MS = 15.0

# Time signature inference ────────────────────────────────────────────────────
# After BPM detection, onset autocorrelation is sampled at multiples of the
# beat period to determine how many beats group into a bar.
#
# TIME_SIG_CANDIDATES  — bar-level numerators to test (denominator assumed = 4).
# TIME_SIG_4_4_BIAS    — score bonus added to the 4/4 candidate to prefer it
#                        when evidence is ambiguous.  Keeps common-time songs
#                        from being flipped to 3/4 by sparse onset patterns.
# TIME_SIG_6_8_RATIO   — if the dotted-quarter lag (1.5×beat) autocorrelation
#                        reaches this fraction of the 3-beat lag score, the
#                        triple-meter winner is reported as 6/8 rather than 3/4.
TIME_SIG_CANDIDATES = [2, 3, 4]   # numerators to test
TIME_SIG_4_4_BIAS   = 0.05        # fractional bonus to the 4/4 candidate
TIME_SIG_6_8_RATIO  = 0.85        # dotted-quarter / 3-beat ratio for 6/8 vs 3/4
