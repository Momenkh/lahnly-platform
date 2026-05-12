"""
Vocals — Pitch Extraction Settings
=====================================
Vocals are monophonic → CREPE is the primary extractor.
basic-pitch runs as a secondary pass to catch missed frames and verify onsets.

Voice characteristics that shape these settings:
  - Strictly monophonic (one pitch at a time).
  - Vibrato: ±1 semitone fluctuation is normal, not two separate notes.
  - Melisma: rapid pitch changes on a single syllable need short min_note_ms.
  - Breath gaps: silence between phrases should not trigger false notes.
  - F0 is very clear for CREPE (harmonic, single-source).

Two modes (detected automatically):
  vocals_lead    — foreground lead vocal
  vocals_harmony — background harmony / doubled part
"""

# ── CREPE settings (primary extractor) ───────────────────────────────────────
# CREPE_FMAX must stay below 1975 Hz.  Vocals never exceed ~988 Hz, so 1200 Hz
# gives ample headroom while staying firmly inside the model's valid range.
VOCALS_CREPE_ENABLED    = True
VOCALS_CREPE_MODEL      = "tiny"
VOCALS_CREPE_FMAX       = 1400.0   # Hz — extended from 1200 to capture soprano top (C6 = 1047Hz, C#6 = 1109Hz)
# Per-mode ceiling: harmony can reach slightly higher than lead
VOCALS_CREPE_FMAX_PER_MODE = {
    "vocals_lead":    1200.0,   # typical lead vocal ceiling
    "vocals_harmony": 1400.0,   # harmonies can reach higher
}

# Harmony mode uses basic-pitch as primary (polyphonic → captures harmony lines)
# and CREPE as a confidence booster only. Lead mode keeps CREPE as primary.
VOCALS_HARMONY_BP_PRIMARY = True
VOCALS_CREPE_CONF_THRESHOLDS = {
    "vocals_lead":    0.65,   # main vocal — CREPE scores well on clean voice
    "vocals_harmony": 0.60,   # harmony stems have more bleed → lower threshold
}
VOCALS_CREPE_MIN_NOTE_S   = 0.060   # seconds — shorter than guitar to catch melisma
VOCALS_CREPE_MAX_GAP_S    = 0.120   # wider than guitar — breath pauses inside a phrase
VOCALS_CREPE_PITCH_TOLERANCE = 1    # semitones — vibrato ±1 st counts as one note
VOCALS_CREPE_REPET_ENABLED       = False  # OFF by default; enable with --vocals-repet flag
VOCALS_CREPE_REPET_AUTO_THRESHOLD = 0.55   # auto-enable when stem_confidence < this

# ── basic-pitch settings (secondary / verification pass) ─────────────────────
VOCALS_MULTI_PASS_CONFIGS = {
    # vocals_lead: CREPE is primary extractor; basic-pitch supplements.
    # Single pass — loose pass1 (frame=0.25) was below vocal overtone ceiling
    # and added harmonic noise that CREPE/cleaning then had to undo.
    # vocals_harmony: basic-pitch is primary (VOCALS_HARMONY_BP_PRIMARY).
    # Keeps two passes since harmony lines are quieter and harder to detect,
    # but pass1 tightened to (0.38, 0.28) to reduce overtone leakage.
    "vocals_lead":    [(0.44, 0.32, 60)],                         # single pass
    "vocals_harmony": [(0.38, 0.28, 60),    (0.52, 0.40,  90)],  # two passes (harmony lines quieter)
}

VOCALS_PITCH_THRESHOLDS = {
    "vocals_lead":    (None, None, 60),
    "vocals_harmony": (None, None, 60),
}

VOCALS_PITCH_ONSET_BASE  = 0.48
VOCALS_PITCH_ONSET_SCALE = 0.12
VOCALS_PITCH_FRAME_BASE  = 0.36
VOCALS_PITCH_FRAME_SCALE = 0.10

VOCALS_PITCH_MERGE_PROXIMITY_S = 0.080   # wider: vocal vibrato causes detection gaps

VOCALS_CONF_WEIGHT_STRONG    = 1.00
VOCALS_CONF_WEIGHT_CONFIRMED = 0.88
VOCALS_CONF_WEIGHT_BASE      = 0.65

VOCALS_PYIN_MIN_NOTE_DURATION_S = 0.06

# ── Sustained note confidence boost ──────────────────────────────────────────
VOCALS_SUSTAINED_BOOST_MIN_S  = 0.150
VOCALS_SUSTAINED_BOOST_AMOUNT = 0.08
VOCALS_SUSTAINED_BOOST_CAP    = 0.95
