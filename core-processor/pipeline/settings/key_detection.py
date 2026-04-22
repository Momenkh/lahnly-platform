"""
Stage 5 — Key Detection
========================
Krumhansl-Schmuckler (KS) key profiles and scale-selection bias.

The KS algorithm correlates a chroma histogram (pitch-class energy distribution)
built from the detected notes against 24 key templates (12 major + 12 minor).
The key with the highest correlation score is chosen as the tonic and mode.

Scale-selection bias
--------------------
Raw KS scores treat all keys equally.  Guitar music is heavily skewed toward
minor pentatonic / natural minor scales for lead playing.  A small bonus for
pentatonic minor nudges the algorithm to prefer minor readings when the scores
are close, reducing incorrect major-key labels on blues and rock recordings.

Extend this section if you add support for additional scale biases (e.g. Dorian,
Mixolydian) or multi-key / modulating-key detection.
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
