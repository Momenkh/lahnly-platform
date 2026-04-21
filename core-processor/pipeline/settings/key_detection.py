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
# Why pentatonic minor?  It is by far the most common scale in lead guitar
# (blues, rock, metal, country).  When the KS scores for major and pentatonic
# minor are close, this bonus tips the decision toward minor — which is
# almost always the right answer for guitar solos.
#
# 0.04 is conservative: large enough to break ties but too small to override
# a clear major reading (typical KS score differences of 0.1–0.3).
# Raise toward 0.10 if you find the algorithm keeps calling minor-sounding
# solos "major"; lower toward 0.01 if you work mostly with major-key material.
KEY_PENTATONIC_MINOR_BIAS = 0.04
