"""
Stage 9: Audio Playback
Input:  mapped notes [{string, fret, pitch, start, duration, confidence}]
Output: synthesized audio saved to outputs/09_preview.wav
        optional playback via pygame

Quality improvements:
  - Guitar-like waveform: sine fundamental + 2nd harmonic + 3rd harmonic
    (adds brightness without external libraries)
  - Velocity-aware amplitude: amp = base + scale × confidence
    (confident/loud notes sound stronger; quiet/uncertain notes sound softer)
"""

import numpy as np
import os

from pipeline.settings import (
    AUDIO_SAMPLE_RATE,
    AUDIO_HARMONIC_2ND,
    AUDIO_HARMONIC_3RD,
    AUDIO_AMP_BASE,
    AUDIO_AMP_CONF_SCALE,
    AUDIO_ATTACK_S,
    AUDIO_RELEASE_S,
)


def midi_to_hz(midi_pitch: int) -> float:
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


def synthesize_notes(notes: list[dict]) -> np.ndarray:
    """
    Render notes into a mono float32 audio buffer (values in [-1, 1]).
    Each note uses a guitar-like additive waveform (fundamental + harmonics).
    Amplitude is scaled by note confidence (velocity proxy).
    """
    if not notes:
        return np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)

    total_duration = max(n["start"] + n["duration"] for n in notes) + 0.5
    total_samples  = int(total_duration * AUDIO_SAMPLE_RATE)
    buffer         = np.zeros(total_samples, dtype=np.float32)

    for note in notes:
        freq  = midi_to_hz(note["pitch"])
        dur   = note["duration"]
        conf  = float(note.get("confidence", 0.5))
        amp   = AUDIO_AMP_BASE + AUDIO_AMP_CONF_SCALE * conf

        start_sample = int(note["start"] * AUDIO_SAMPLE_RATE)
        num_samples  = int(dur * AUDIO_SAMPLE_RATE)
        if num_samples <= 0:
            continue

        t = np.linspace(0, dur, num_samples, endpoint=False)

        # Guitar-like additive synthesis: fundamental + 2nd + 3rd harmonic
        wave = (
            np.sin(2 * np.pi * freq * t)
            + AUDIO_HARMONIC_2ND * np.sin(2 * np.pi * freq * 2 * t)
            + AUDIO_HARMONIC_3RD * np.sin(2 * np.pi * freq * 3 * t)
        ).astype(np.float32)

        # Amplitude envelope: attack + exponential decay (pluck feel)
        attack  = min(int(AUDIO_ATTACK_S  * AUDIO_SAMPLE_RATE), num_samples // 4)
        release = min(int(AUDIO_RELEASE_S * AUDIO_SAMPLE_RATE), num_samples // 4)
        envelope = np.ones(num_samples, dtype=np.float32)
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        if release > 0:
            envelope[-release:] = np.linspace(1, 0, release)

        wave *= envelope * amp

        end_sample = start_sample + num_samples
        if end_sample > total_samples:
            wave = wave[:total_samples - start_sample]
            end_sample = total_samples
        buffer[start_sample:end_sample] += wave

    # Normalize to prevent clipping
    peak = np.max(np.abs(buffer))
    if peak > 1.0:
        buffer /= peak

    return buffer


def save_audio(notes: list[dict]) -> str:
    """
    Synthesize notes and save to outputs/09_preview.wav.
    Returns the path to the saved file.
    """
    import soundfile as sf
    from pipeline.config import get_outputs_dir

    print(f"[Stage 9] Synthesizing {len(notes)} notes for export...")
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

    # pygame mixer expects 16-bit stereo
    audio_16bit = (buffer * 32767).astype(np.int16)
    stereo = np.column_stack([audio_16bit, audio_16bit])  # mono -> stereo

    pygame.mixer.init(frequency=AUDIO_SAMPLE_RATE, size=-16, channels=2, buffer=512)
    sound = pygame.sndarray.make_sound(stereo)

    total_s = len(buffer) / AUDIO_SAMPLE_RATE
    print(f"[Stage 9] Playing {total_s:.1f}s of audio... (press Ctrl+C to stop)")

    sound.play()
    pygame.time.wait(int(total_s * 1000) + 500)
    pygame.mixer.quit()
    print("[Stage 9] Playback complete.")
