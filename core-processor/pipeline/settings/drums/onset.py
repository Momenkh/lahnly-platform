"""
Drums — Onset Detection Settings
===================================
Drums bypass the pitch pipeline entirely.
Stage 2 is replaced by onset detection + per-hit drum class classification.

Hit classes (spectral centroid + bandwidth → class label):
  kick     — fundamental ~60–120 Hz, very low centroid, high energy
  snare    — broadband crack, high spectral flatness, mid centroid
  hihat    — very high centroid (>4 kHz), short duration
  tom      — lower centroid than snare, longer decay
  cymbal   — high centroid + sustained decay (ride / crash)

Output schema per hit: {hit_class, start, confidence}
No duration, no pitch.
"""

# ── Onset detection ───────────────────────────────────────────────────────────
DRUMS_ONSET_DELTA            = 0.07    # librosa onset_detect delta param (sensitivity)
DRUMS_MIN_INTERVAL_S         = 0.040   # minimum gap between two distinct hits (40ms)
DRUMS_ONSET_BACKTRACK        = True    # snap onset to nearest energy trough

# ── Analysis window around each detected onset ────────────────────────────────
DRUMS_HIT_ANALYSIS_WINDOW_S  = 0.040  # seconds of audio to analyse per hit

# ── Spectral centroid thresholds for hit classification ───────────────────────
# These thresholds operate on the centroid in Hz of a short window after each onset.
# Measured on Demucs htdemucs_6s separated stems (centroids are lower than raw mix
# because the separator attenuates out-of-band content):
#   kick:   measured centroid 170–295 Hz  → threshold at 400 for safety
#   tom:    measured centroid 330–1180 Hz → threshold at 1600 (wide floor-tom range)
#   snare:  measured centroid 1400–3700 Hz
#   hihat:  measured centroid 5000–5400 Hz (very tight — only clearest hits)
#   cymbal: measured centroid 5000–6700 Hz; some hi-hat bleed in 3500–5000 Hz range
DRUMS_CENTROID_KICK_MAX      = 400.0   # Hz — raised from 300 to capture wider kick tail
DRUMS_CENTROID_TOM_MAX       = 1600.0  # Hz — raised from 1200 to absorb floor-tom range
DRUMS_CENTROID_HIHAT_MIN     = 3500.0  # Hz — lowered from 5000; captures hi-hat bleed in separated stems

# ── Spectral flatness thresholds ──────────────────────────────────────────────
# High flatness = noise-like = snare crack or hi-hat.
# In Demucs-separated stems flatness is suppressed by ~0.5–0.7× vs raw mix.
DRUMS_FLATNESS_SNARE_MIN     = 0.06   # lowered from 0.12 — real snares in separated stems
DRUMS_FLATNESS_HIHAT_MIN     = 0.10   # lowered from 0.20 — hi-hats in separated stems

# ── Hi-hat open/closed sub-classification ────────────────────────────────────
# Spectral flux of the hit window above this threshold indicates an open hi-hat
# (longer decay keeps energy moving frame-to-frame; closed hi-hats decay fast).
DRUMS_HIHAT_OPEN_FLUX_THRESH = 0.15

# ── Hit class labels ──────────────────────────────────────────────────────────
# "hihat" is kept as a fallback for saved JSON from before open/closed split.
DRUMS_HIT_CLASSES = ["kick", "snare", "hihat", "hihat_open", "hihat_closed", "tom", "cymbal"]

# ── ASCII notation row order (top to bottom in grid) ─────────────────────────
DRUMS_NOTATION_ROW_ORDER = ["cymbal", "hihat", "snare", "tom", "kick"]

# ── Drum grid: subdivisions per bar ──────────────────────────────────────────
DRUMS_GRID_SUBDIVISIONS = 16   # 16th-note grid

# ── Hit confidence formula ────────────────────────────────────────────────────
# Final confidence = class_conf × CLASS_WEIGHT + strength_conf × STRENGTH_WEIGHT
# class_conf    — base confidence from the spectral classifier (_classify_hit)
# strength_conf — normalised onset strength from librosa onset envelope
DRUMS_CONF_CLASS_WEIGHT    = 0.6
DRUMS_CONF_STRENGTH_WEIGHT = 0.4

# Per-class base confidence values from _classify_hit (empirically tuned)
DRUMS_CONF_KICK_BASE        = 0.75   # kick: very reliable at low centroid
DRUMS_CONF_KICK_SCALE       = 0.20   # bonus: tighter centroid → higher kick confidence
DRUMS_CONF_HIHAT_BASE       = 0.75   # hi-hat: reliable when flatness passes threshold
DRUMS_CONF_CYMBAL_BASE      = 0.65   # cymbal (ride/crash): slightly less certain
DRUMS_CONF_SNARE_BASE       = 0.65   # snare: mid-band + high flatness
DRUMS_CONF_SNARE_SCALE      = 0.20   # bonus: higher flatness → higher snare confidence
DRUMS_CONF_SNARE_FLAT_RANGE = 0.15   # flatness range over which snare scale applies
DRUMS_CONF_TOM_BASE         = 0.60   # tom: lower confidence — centroid overlaps snare
DRUMS_CONF_DEFAULT_BASE     = 0.55   # default fallback (mid-band, low flatness)
