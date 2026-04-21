"""
Stage 9 — Audio Synthesis
==========================
Settings for the synthesized preview WAV file.

The synthesizer uses additive synthesis: each note is a sum of sinusoidal
harmonics with an amplitude envelope (attack + sustain + release).  The result
is a guitar-like pluck sound without requiring a sample library.

Harmonic series
---------------
Guitar tone is dominated by the fundamental and its first few harmonics.
The second harmonic (one octave up, +12 semitones) adds brightness; the third
(octave + fifth, +19 semitones) adds warmth.  Higher harmonics contribute
little perceptible character and are omitted.

Amplitude scaling
-----------------
Per-note amplitude scales with detection confidence so high-confidence notes
are louder and uncertain notes are quieter.  This creates a natural dynamic
that also serves as an audio quality indicator.
"""

# Output sample rate.  Must match the rest of the pipeline's audio processing.
# Do not change unless you also update SEPARATION_SAMPLE_RATE and any
# downstream playback code.
AUDIO_SAMPLE_RATE = 44100   # Hz

# ── Additive synthesis harmonics ─────────────────────────────────────────────
# Amplitude of each harmonic relative to the fundamental (= 1.0).
# Reduce to 0.0 to remove a harmonic entirely for a purer, flute-like tone.
AUDIO_HARMONIC_2ND = 0.30   # first overtone (2× fundamental, one octave up)
AUDIO_HARMONIC_3RD = 0.15   # second overtone (3× fundamental, octave + perfect fifth)

# ── Per-note amplitude ────────────────────────────────────────────────────────
# amp = AUDIO_AMP_BASE + AUDIO_AMP_CONF_SCALE × confidence
# At confidence = 1.0: amp = 0.15 + 0.25 = 0.40
# At confidence = 0.0: amp = 0.15
# The base ensures even very uncertain notes are audible; the scale adds
# dynamic contrast.  Keep AMP_BASE + AMP_CONF_SCALE ≤ 0.6 to avoid clipping
# when multiple notes sound simultaneously.
AUDIO_AMP_BASE       = 0.15
AUDIO_AMP_CONF_SCALE = 0.25

# ── Amplitude envelope ────────────────────────────────────────────────────────
# Simulates the pluck transient (attack) and string decay (release).
# Attack: time in seconds from silence to peak amplitude.
# Release: time in seconds from note end to silence.
AUDIO_ATTACK_S  = 0.008   # 8ms — fast pluck-like transient
AUDIO_RELEASE_S = 0.060   # 60ms — short tail to avoid abrupt cutoffs
