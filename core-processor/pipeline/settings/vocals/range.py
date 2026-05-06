"""
Vocals — Range Settings
========================
MIDI and Hz bounds for singing voice.

Human voice ranges (approximate):
  Bass voice   : E2–E4  (MIDI 40–64)
  Tenor        : C3–C5  (MIDI 48–72)
  Alto/Mezzo   : F3–F5  (MIDI 53–77)
  Soprano      : C4–C6  (MIDI 60–84)

We use a wide combined range (C3–B5, MIDI 48–83) that covers all common
voices without being so wide that we pick up instrumental harmonics.
"""

VOCALS_MIDI_MIN = 48    # C3 — safe lower bound covering bass/baritone
VOCALS_MIDI_MAX = 83    # B5 — covers soprano with headroom

VOCALS_HZ_MIN = 440.0 * 2 ** ((VOCALS_MIDI_MIN - 69) / 12)   # ~130.8 Hz
VOCALS_HZ_MAX = 440.0 * 2 ** ((VOCALS_MIDI_MAX - 69) / 12)   # ~987.8 Hz

# Voice type sub-ranges for future mode detection
VOCALS_VOICE_RANGES = {
    "low":    (48, 64),   # bass/baritone/alto lower register
    "mid":    (55, 74),   # tenor/alto/mezzo main range
    "high":   (67, 83),   # soprano/tenor upper register
}
