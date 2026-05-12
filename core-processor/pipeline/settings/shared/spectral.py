"""
Spectral analysis settings — shared across all instruments.

Used by pipeline/shared/spectral.py for:
  - Velocity estimation from stem RMS
  - Harmonic coherence check (fundamental vs overtone discrimination)
  - Spectral presence verification (is this pitch in the stem at all?)
  - Attack envelope gate (real notes have an amplitude ramp; noise does not)
  - Pitch-aware isolation filter (only pitch-close neighbors count)
"""

# ── Velocity estimation ───────────────────────────────────────────────────────
# velocity = VELOCITY_ATTACK_WEIGHT × rms_norm + VELOCITY_CONF_WEIGHT × confidence
# rms_norm is the RMS in the first VELOCITY_ATTACK_WINDOW_S of the note, normalized
# against the VELOCITY_NORM_PERCENTILE-th percentile RMS across all notes in the song.
VELOCITY_ATTACK_WEIGHT   = 0.60
VELOCITY_CONF_WEIGHT     = 0.40
VELOCITY_NORM_PERCENTILE = 95    # percentile RMS for normalization
VELOCITY_ATTACK_WINDOW_S = 0.020 # seconds — attack transient window
VELOCITY_MIN             = 1     # minimum MIDI velocity
VELOCITY_MAX             = 127   # maximum MIDI velocity

# ── Harmonic coherence check ──────────────────────────────────────────────────
# For a note at MIDI pitch P, if the energy one octave below (P-12) is
# HARMONIC_DOMINANCE_RATIO times stronger, the note is likely an overtone.
# Notes marked likely_overtone are removed unless confidence >= HARMONIC_CONF_OVERRIDE.
#
# Piano uses a higher ratio because piano fundamentals dominate their harmonics
# much more strongly than guitar — set per instrument in instrument cleaning.
HARMONIC_COHERENCE_ENABLED  = True
HARMONIC_DOMINANCE_RATIO    = 2.5    # sub-octave energy must be 2.5× stronger
HARMONIC_CONF_OVERRIDE      = 0.65   # high-confidence notes survive the overtone check
# CQT parameters for the harmonic coherence computation (reuses shared evaluation CQT)
HARMONIC_CQT_HOP            = 512
HARMONIC_CQT_N_BINS         = 84
HARMONIC_CQT_BINS_PER_OCT   = 12

# ── Spectral presence verification ───────────────────────────────────────────
# For every note, verify the stem's CQT has energy at that pitch at that time.
# Below SPECTRAL_PRESENCE_MIN_ENERGY → the frequency is absent from the stem.
# Skip the check for notes with confidence >= SPECTRAL_PRESENCE_CONF_OVERRIDE.
SPECTRAL_PRESENCE_ENABLED      = True
SPECTRAL_PRESENCE_MIN_ENERGY   = 0.05   # CQT magnitude floor
SPECTRAL_PRESENCE_CONF_OVERRIDE = 0.60  # high-confidence notes bypass the gate
SPECTRAL_PRESENCE_PITCH_TOL    = 2      # ±2 CQT bins tolerance

# ── Attack envelope gate ──────────────────────────────────────────────────────
# Real notes have an amplitude ramp at onset; noise/bleed starts at full amplitude.
# Measure RMS(first 20ms) / RMS(50ms before onset).
# If ratio < ATTACK_GATE_MIN_RATIO the note is likely noise.
ATTACK_GATE_ENABLED        = True
ATTACK_GATE_PRE_WINDOW_S   = 0.030  # pre-onset noise floor window (shorter avoids previous-note tail contamination)
ATTACK_GATE_POST_WINDOW_S  = 0.020  # onset attack window
ATTACK_GATE_MIN_RATIO      = 1.3    # attack must be 1.3× stronger than pre-noise floor (1.8 was killing arpeggios/fast runs)
ATTACK_GATE_CONF_OVERRIDE  = 0.55   # skip for medium-to-high confidence notes (was 0.70 — too strict)

# ── Pitch-aware isolation filter ─────────────────────────────────────────────
# Upgrade over the original isolation filter: only notes within
# CLEANING_ISOLATION_MAX_INTERVAL_ST semitones count as "neighbors."
# A ghost note surrounded only by distant-pitch legitimate notes will fail.
CLEANING_ISOLATION_MAX_INTERVAL_ST = 12  # default; override per instrument in cleaning
CLEANING_ISOLATION_MIN_NEIGHBORS   = 2   # need this many pitch-close neighbors to survive
