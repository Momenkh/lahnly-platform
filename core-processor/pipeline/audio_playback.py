"""
Stage 9: Audio Playback
Input:  mapped notes [{string, fret, pitch, start, duration, confidence}]
Output: synthesized audio saved to outputs/09_preview.wav
        optional playback via pygame

Synthesis: Karplus-Strong physical modelling.
A delay buffer of Gaussian noise is recirculated through a one-pole lowpass
(0.5 × average of two consecutive samples × damping factor), which produces
a realistic plucked-string timbre and natural exponential decay — no sample
library required.  The lowpass bleeds high-frequency energy faster than low,
matching how real strings damp their upper partials first.
"""

import numpy as np
import os
import scipy.signal

from pipeline.settings import (
    AUDIO_SAMPLE_RATE,
    AUDIO_AMP_BASE,
    AUDIO_AMP_CONF_SCALE,
    AUDIO_KS_DAMPING,
)


def midi_to_hz(midi_pitch: int) -> float:
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


def _karplus_strong(
    freq: float,
    duration: float,
    amp: float,
    sr: int,
    damping: float = AUDIO_KS_DAMPING,
) -> np.ndarray:
    """
    Karplus-Strong plucked string synthesis via scipy IIR filter.

    The recurrence y[n] = d/2·y[n-P] + d/2·y[n-P-1] is expressed as an IIR
    filter applied to a half-sine pulse excitation (one delay period).
    A sine pulse gives a warm, smooth attack vs the harsh crack of white noise.
    scipy.signal.lfilter handles the IIR feedback in C.
    """
    n_samples = int(duration * sr)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    delay = max(2, round(sr / freq))

    # Excitation: half-sine pulse over one delay period.
    # A smooth sine bump produces a warm pluck attack instead of the harsh
    # crack of white noise — closer to an actual guitar pick or finger pluck.
    excitation = np.zeros(n_samples, dtype=np.float32)
    n_excite   = min(delay, n_samples)
    excitation[:n_excite] = (
        np.sin(np.linspace(0.0, np.pi, n_excite)) * amp
    ).astype(np.float32)

    # IIR denominator polynomial for the KS recurrence
    a = np.zeros(delay + 2, dtype=np.float64)
    a[0]         = 1.0
    a[delay]     = -damping * 0.5
    a[delay + 1] = -damping * 0.5

    wave = scipy.signal.lfilter([1.0], a, excitation).astype(np.float32)

    # Cosine fade-out over the last 20ms — eliminates the click/pop caused by
    # hard-truncating the KS ring at note boundary.
    fade_samples = min(int(0.020 * sr), len(wave) // 4)
    if fade_samples > 1:
        fade = 0.5 - 0.5 * np.cos(np.linspace(np.pi, 2 * np.pi, fade_samples))
        wave[-fade_samples:] *= fade.astype(np.float32)

    return wave


def synthesize_notes(notes: list[dict]) -> np.ndarray:
    """
    Render notes into a mono float32 audio buffer (values in [-1, 1]).
    Each note is synthesized via Karplus-Strong plucked-string modelling.
    Amplitude is confidence-scaled: louder for high-confidence notes.
    """
    if not notes:
        return np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)

    total_duration = max(n["start"] + n["duration"] for n in notes) + 1.0
    total_samples  = int(total_duration * AUDIO_SAMPLE_RATE)
    buffer         = np.zeros(total_samples, dtype=np.float32)

    for note in notes:
        freq = midi_to_hz(note["pitch"])
        dur  = note["duration"]
        conf = float(note.get("confidence", 0.5))
        amp  = AUDIO_AMP_BASE + AUDIO_AMP_CONF_SCALE * conf

        wave = _karplus_strong(freq, dur, amp, AUDIO_SAMPLE_RATE)
        if len(wave) == 0:
            continue

        start_sample = int(note["start"] * AUDIO_SAMPLE_RATE)
        end_sample   = start_sample + len(wave)
        if end_sample > total_samples:
            wave       = wave[:total_samples - start_sample]
            end_sample = total_samples
        buffer[start_sample:end_sample] += wave

    # DC-blocking high-pass (single-pole IIR, ~20 Hz cutoff).
    # The half-sine excitation is always positive, so each KS note carries a DC
    # offset that accumulates across the mix and causes low-frequency rumble.
    b_dc = np.array([1.0, -1.0], dtype=np.float64)
    a_dc = np.array([1.0, -0.9997], dtype=np.float64)
    buffer = scipy.signal.lfilter(b_dc, a_dc, buffer.astype(np.float64)).astype(np.float32)

    peak = np.max(np.abs(buffer))
    if peak > 1e-6:
        buffer /= peak

    return buffer


def save_audio(notes: list[dict]) -> str:
    """
    Synthesize notes and save to outputs/09_preview.wav.
    Returns the path to the saved file.
    """
    import soundfile as sf
    from pipeline.config import get_outputs_dir

    print(f"[Stage 9] Synthesizing {len(notes)} notes (Karplus-Strong)...")
    buffer = synthesize_notes(notes)

    os.makedirs(get_outputs_dir(), exist_ok=True)
    out_path = os.path.join(get_outputs_dir(), "09_preview.wav")
    sf.write(out_path, buffer, AUDIO_SAMPLE_RATE, subtype="PCM_16")
    print(f"[Stage 9] Saved -> {out_path}  ({len(buffer)/AUDIO_SAMPLE_RATE:.1f}s)")
    return out_path


def play_notes(notes: list[dict]) -> None:
    """
    Synthesize and play notes via pygame.
    Blocks until playback finishes.
    """
    import pygame

    print(f"[Stage 9] Synthesizing {len(notes)} notes...")
    buffer = synthesize_notes(notes)

    audio_16bit = (buffer * 32767).astype(np.int16)
    stereo      = np.column_stack([audio_16bit, audio_16bit])

    pygame.mixer.init(frequency=AUDIO_SAMPLE_RATE, size=-16, channels=2, buffer=512)
    sound = pygame.sndarray.make_sound(stereo)

    total_s = len(buffer) / AUDIO_SAMPLE_RATE
    print(f"[Stage 9] Playing {total_s:.1f}s of audio... (press Ctrl+C to stop)")

    sound.play()
    pygame.time.wait(int(total_s * 1000) + 500)
    pygame.mixer.quit()
    print("[Stage 9] Playback complete.")
