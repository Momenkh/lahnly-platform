"""
Stage 9 — Audio Synthesis
==========================
Settings for the synthesized preview WAV file.

The synthesizer uses Karplus-Strong physical modelling: a delay buffer of
Gaussian noise is fed through a one-pole lowpass feedback loop, producing
a realistic plucked-string decay without requiring a sample library.

Karplus-Strong basics
---------------------
  delay_length = round(sample_rate / frequency)   # one period of the string
  y[n] = damping × 0.5 × (y[n-P] + y[n-P-1])    # lowpass feedback recurrence

The 0.5 factor implements the one-pole lowpass that bleeds energy faster at
high frequencies, replicating how a real string's upper partials damp first.
AUDIO_KS_DAMPING controls the overall decay speed: 1.0 = no decay, 0.99 = fast.

Amplitude scaling
-----------------
Per-note amplitude scales with detection confidence so high-confidence notes
are louder and uncertain notes are quieter.  This creates a natural dynamic
that also serves as an audio quality indicator.
"""

# Output sample rate.  Must match the rest of the pipeline's audio processing.
AUDIO_SAMPLE_RATE = 44100   # Hz

# ── Per-note amplitude ────────────────────────────────────────────────────────
# amp = AUDIO_AMP_BASE + AUDIO_AMP_CONF_SCALE × confidence
# At confidence = 1.0: amp = 0.40; at confidence = 0.0: amp = 0.15
AUDIO_AMP_BASE       = 0.15
AUDIO_AMP_CONF_SCALE = 0.25

# ── Karplus-Strong damping ────────────────────────────────────────────────────
# Controls how quickly the plucked string decays.
# Range: 0.990 (fast decay / short sustain) → 0.999 (slow decay / long sustain)
# Guitar: 0.996 is a good middle ground for a standard plucked note.
# Lower for a more percussive / muted sound; raise for a sustain-heavy sound.
AUDIO_KS_DAMPING = 0.996
