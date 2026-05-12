"""
Stage 1: Instrument Separation
Input:  full mix audio file
Output: isolated guitar stem  -> outputs/01_guitar_stem.wav
        stem quality metadata -> outputs/01_stem_meta.json

Model cascade (tried in order until a non-silent stem is found):
  1. htdemucs_6s / guitar  — dedicated guitar stem (best)
  2. htdemucs_ft_other / other — fine-tuned fallback
  3. htdemucs / other      — base 4-stem last resort

Quality improvements:
  - Input loudness-normalised before separation (prevents energy misallocation)
  - RMS-based silence check (more reliable than peak for sparse guitar parts)
  - stem_confidence score saved to stem_meta and used by downstream stages
  - overlap=0.5 reduces stitching artifacts at chunk boundaries
  - shifts=1 averages two passes for cleaner output
  - Optional raw mix blending (controlled by SEPARATION_MIX_WITH_RAW in settings):
      final = alpha * stem + (1 - alpha) * original   (alpha = stem_confidence)
    Disabled by default. When enabled and confidence is low, blending back the
    original recovers transients and brightness that Demucs strips — at the cost
    of bleed from other instruments.
"""

import json
import os
import numpy as np

from pipeline.config import get_outputs_dir, get_instrument_dir
from pipeline.settings import (
    SEPARATION_SAMPLE_RATE,
    SEPARATION_RMS_SILENCE_THRESH,
    SEPARATION_SHIFTS,
    SEPARATION_OVERLAP,
    SEPARATION_TARGET_LUFS,
    SEPARATION_MODELS,
    SEPARATION_MODEL_BASE_CONF,
    SEPARATION_MIX_WITH_RAW,
    SEPARATION_MAX_RAW_BLEND,
    SEPARATION_BLEND_GATE_WINDOW,
    SEPARATION_CONF_BASE_WEIGHT,
    SEPARATION_CONF_ENERGY_WEIGHT,
    SEPARATION_PEAK_NORM_TARGET,
)


def separate_guitar(audio_path: str, save: bool = True) -> str:
    """
    Isolate the guitar stem using Demucs.
    Saves stem WAV + stem_meta.json with quality information.
    Returns the path to the saved stem WAV file.
    """
    return _separate_any(
        audio_path,
        model_cascade=SEPARATION_MODELS,
        instrument_label="guitar",
        save=save,
        use_cache=False,
        apply_raw_blend=True,
    )


def separate_stem(
    audio_path: str,
    stem_name: str,
    model_cascade: list[tuple[str, str]],
    instrument_label: str,
    save: bool = True,
) -> str:
    """
    Generic stem extractor for any instrument supported by Demucs.

    Parameters
    ----------
    audio_path      : path to the full-mix audio file
    stem_name       : Demucs stem key, e.g. "bass", "piano", "vocals", "drums"
    model_cascade   : ordered list of (model_name, stem_name) pairs to try
    instrument_label: short label used in filenames, e.g. "bass" → "01_bass_stem.wav"
    save            : write WAV + meta JSON to outputs directory

    Returns the path to the saved stem WAV file.
    Delegates to _separate_any() — identical to separate_guitar() but parameterised.

    Optimisation: when htdemucs_6s is used it produces all 6 stems in one pass.
    This function saves every stem it produces so subsequent instrument runs reuse
    the cached WAVs without re-running Demucs.
    """
    return _separate_any(
        audio_path,
        model_cascade=model_cascade,
        instrument_label=instrument_label,
        save=save,
        use_cache=True,
        apply_raw_blend=False,
    )


def _separate_any(
    audio_path: str,
    model_cascade: list[tuple[str, str]],
    instrument_label: str,
    save: bool = True,
    use_cache: bool = True,
    apply_raw_blend: bool = False,
) -> str:
    """
    Core stem extraction logic shared by separate_guitar() and separate_stem().

    Parameters
    ----------
    model_cascade   : ordered list of (model_name, stem_name) pairs to try
    instrument_label: short label used in filenames, e.g. "guitar", "bass"
    use_cache       : skip Demucs if WAV + meta already exist on disk
    apply_raw_blend : if True, apply SEPARATION_MIX_WITH_RAW confidence-weighted blend
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    out_path  = os.path.join(get_instrument_dir(instrument_label), "01_stem.wav")
    meta_path = os.path.join(get_instrument_dir(instrument_label), "01_stem_meta.json")

    # ── Early return if stem already separated ────────────────────────────────
    if use_cache and os.path.isfile(out_path) and os.path.isfile(meta_path):
        print(f"[Stage 1] Cached stem found — skipping Demucs: {os.path.basename(out_path)}")
        return out_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Stage 1] Loading audio: {audio_path}")
    waveform, sr = _load_audio_tensor(audio_path)

    waveform_np = waveform.numpy()
    waveform_np = _normalize_loudness(waveform_np, sr)
    mix_rms     = float(np.sqrt(np.mean(waveform_np ** 2)))
    waveform    = torch.from_numpy(waveform_np)

    if sr != SEPARATION_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SEPARATION_SAMPLE_RATE)(waveform)

    waveform_batch = waveform.unsqueeze(0).to(device)
    duration_s     = waveform_batch.shape[-1] / SEPARATION_SAMPLE_RATE
    print(f"[Stage 1] Audio loaded: {duration_s:.1f}s  "
          f"(normalised to {SEPARATION_TARGET_LUFS} LUFS)")

    extracted_stem = None
    used_model     = None
    used_stem_key  = None

    for model_name, model_stem in model_cascade:
        print(f"[Stage 1] Trying {model_name} / stem='{model_stem}'  (device: {device})")
        try:
            model = get_model(model_name)
        except Exception as e:
            print(f"[Stage 1] Model {model_name} unavailable ({e}) — skipping")
            continue
        model.to(device)
        model.eval()

        with torch.no_grad():
            sources = apply_model(
                model, waveform_batch,
                device=device,
                shifts=SEPARATION_SHIFTS,
                overlap=SEPARATION_OVERLAP,
                progress=True,
            )

        if model_stem not in model.sources:
            print(f"[Stage 1] Stem '{model_stem}' not in {model_name} — skipping")
            continue

        stem_idx  = list(model.sources).index(model_stem)
        candidate = sources[0, stem_idx]
        stem_rms  = float(candidate.pow(2).mean().sqrt())
        print(f"[Stage 1] '{model_stem}' stem RMS: {stem_rms:.5f}")

        if model_name == "htdemucs_6s" and save:
            # Always cache all 6 stems — even if THIS instrument's stem is silent,
            # the other stems are valid and needed for cross-stem presence checks.
            _cache_all_6s_stems(sources, model, mix_rms, duration_s)

        if stem_rms >= SEPARATION_RMS_SILENCE_THRESH:
            extracted_stem = candidate
            used_model     = model_name
            used_stem_key  = model_stem
            break
        else:
            print(f"[Stage 1] Stem is silent — trying next option...")

    if extracted_stem is None:
        print(f"\n{'=' * 60}")
        print(f"[Stage 1] *** WARNING: Demucs produced a silent stem for '{instrument_label}'.")
        print(f"[Stage 1] *** Falling back to the raw mix. Transcription quality will be degraded.")
        print(f"{'=' * 60}\n")
        extracted_stem = waveform
        used_model     = "raw mix"
        used_stem_key  = "mix"

    print(f"[Stage 1] Using: {used_model} / stem='{used_stem_key}'")

    # Compute stem_confidence
    stem_rms        = float(extracted_stem.pow(2).mean().sqrt())
    base_conf       = SEPARATION_MODEL_BASE_CONF.get((used_model, used_stem_key), 0.5)
    energy_ratio    = min(1.0, stem_rms / max(mix_rms, 1e-8))
    stem_confidence = round(min(1.0, base_conf * 0.7 + energy_ratio * 0.3), 3)
    print(f"[Stage 1] stem_confidence: {stem_confidence:.3f}")

    # ── Optional confidence-weighted blend with original mix ─────────────────
    stem_weight = 1.0
    raw_weight  = 0.0
    if apply_raw_blend and SEPARATION_MIX_WITH_RAW:
        mix_waveform = waveform.to(extracted_stem.device)
        min_len      = min(extracted_stem.shape[-1], mix_waveform.shape[-1])
        extracted_stem = extracted_stem[..., :min_len]
        mix_waveform   = mix_waveform[..., :min_len]

        stem_weight = stem_confidence
        raw_weight  = SEPARATION_MAX_RAW_BLEND * (1.0 - stem_confidence)

        # Energy gate: raw bleed is multiplied by 0 wherever the stem is silent.
        # Prevents raw mix from leaking into intros, outros, or any section
        # where the target instrument isn't playing.
        gate    = _stem_activity_gate(extracted_stem, SEPARATION_BLEND_GATE_WINDOW,
                                      SEPARATION_RMS_SILENCE_THRESH)
        blended = stem_weight * extracted_stem + raw_weight * gate * mix_waveform

        active_pct = gate.mean().item() * 100
        print(f"[Stage 1] Blended: {stem_weight:.2f} x stem "
              f"+ {raw_weight:.2f} x gate x raw  "
              f"(gate open {active_pct:.0f}% of audio)")
    else:
        blended = extracted_stem
        if apply_raw_blend:
            print("[Stage 1] Using stem only (SEPARATION_MIX_WITH_RAW=False)")

    if save:
        import soundfile as sf

        peak = blended.abs().max().item()
        if peak > 0:
            blended = blended / peak * SEPARATION_PEAK_NORM_TARGET

        audio_np = blended.cpu().numpy().T
        sf.write(out_path, audio_np, SEPARATION_SAMPLE_RATE)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"[Stage 1] Saved -> {out_path}  ({size_kb:.0f} KB)")

        meta = {
            "model":             used_model,
            "stem":              used_stem_key,
            "stem_confidence":   stem_confidence,
            "energy_ratio_raw":  round(stem_rms / max(mix_rms, 1e-8), 5),
            "instrument":        instrument_label,
            "mix_with_raw":      apply_raw_blend and SEPARATION_MIX_WITH_RAW,
            "stem_weight":       round(stem_weight, 3),
            "raw_weight":        round(raw_weight, 3),
            "duration_s":        round(duration_s, 2),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Stage 1] Metadata -> {meta_path}")

    return out_path


def _cache_all_6s_stems(sources, model, mix_rms: float, duration_s: float) -> None:
    """
    Save every stem produced by htdemucs_6s to disk so subsequent instrument
    pipelines can skip re-running Demucs entirely.

    Label mapping: Demucs stem name → instrument_label used in filenames.
    Only writes stems whose file does not already exist.
    """
    import soundfile as sf

    _STEM_TO_LABEL = {
        "guitar": "guitar",
        "bass":   "bass",
        "drums":  "drums",
        "vocals": "vocals",
        "piano":  "piano",
        "other":  "other",
    }

    for stem_name in model.sources:
        label = _STEM_TO_LABEL.get(stem_name, stem_name)
        out_wav  = os.path.join(get_instrument_dir(label), "01_stem.wav")
        out_meta = os.path.join(get_instrument_dir(label), "01_stem_meta.json")
        if os.path.isfile(out_wav) and os.path.isfile(out_meta):
            continue   # already saved

        stem_idx  = list(model.sources).index(stem_name)
        stem_data = sources[0, stem_idx]
        stem_rms  = float(stem_data.pow(2).mean().sqrt())

        peak = stem_data.abs().max().item()
        if peak > 0:
            stem_data = stem_data / peak * SEPARATION_PEAK_NORM_TARGET

        audio_np = stem_data.cpu().numpy().T
        sf.write(out_wav, audio_np, SEPARATION_SAMPLE_RATE)

        base_conf       = SEPARATION_MODEL_BASE_CONF.get(("htdemucs_6s", stem_name), 0.55)
        energy_ratio    = min(1.0, stem_rms / max(mix_rms, 1e-8))
        stem_confidence = round(min(1.0, base_conf * SEPARATION_CONF_BASE_WEIGHT + energy_ratio * SEPARATION_CONF_ENERGY_WEIGHT), 3)

        meta = {
            "model":             "htdemucs_6s",
            "stem":              stem_name,
            "stem_confidence":   stem_confidence,
            "energy_ratio_raw":  round(stem_rms / max(mix_rms, 1e-8), 5),
            "instrument":        label,
            "duration_s":        round(duration_s, 2),
        }
        with open(out_meta, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"[Stage 1] All htdemucs_6s stems cached to: {get_outputs_dir()}/<instrument>/")


def load_stem_meta_for(instrument_label: str = "guitar") -> dict:
    """Load stem quality metadata for a given instrument. Returns safe defaults if not found."""
    path = os.path.join(get_instrument_dir(instrument_label), "01_stem_meta.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"stem_confidence": 0.5, "model": "unknown", "stem": "unknown",
                "instrument": instrument_label}


def get_stem_path_for(instrument_label: str = "guitar") -> str:
    return os.path.join(get_instrument_dir(instrument_label), "01_stem.wav")


def gate_notes_by_stem_energy_for(
    notes: list,
    instrument_label: str = "guitar",
    window_s: float = 0.08,
    thresh: float = 0.005,
) -> list:
    """
    Remove notes from silent stem regions for any instrument.
    Same logic as gate_notes_by_stem_energy() but reads the instrument-prefixed stem.
    """
    import soundfile as sf

    stem_path = get_stem_path_for(instrument_label)
    if not os.path.isfile(stem_path):
        return notes

    try:
        audio, sr = sf.read(stem_path, always_2d=True, dtype="float32")
    except Exception:
        return notes

    mono        = audio.mean(axis=1, dtype=np.float32)
    win_samples = max(1, int(window_s * sr))
    n_samples   = len(mono)
    n_windows   = (n_samples + win_samples - 1) // win_samples

    padded = np.zeros(n_windows * win_samples, dtype=np.float32)
    padded[:n_samples] = mono
    chunks     = padded.reshape(n_windows, win_samples)
    window_rms = np.sqrt((chunks ** 2).mean(axis=1))

    def _is_active(start_s: float, end_s: float) -> bool:
        lo = max(0, int(start_s / window_s))
        hi = min(n_windows, int(end_s / window_s) + 1)
        return bool(np.any(window_rms[lo:hi] > thresh))

    return [n for n in notes if _is_active(n["start"], n["start"] + n.get("duration", 0))]


def load_stem_meta() -> dict:
    """Load guitar stem quality metadata. Delegates to load_stem_meta_for('guitar')."""
    return load_stem_meta_for("guitar")


def get_stem_path() -> str:
    return os.path.join(get_instrument_dir("guitar"), "01_stem.wav")


def gate_notes_by_stem_energy(
    notes: list,
    window_s: float = 0.08,
    thresh: float = 0.005,
) -> list:
    """
    Remove notes that fall entirely within silent windows of the saved stem.

    Loads 01_guitar_stem.wav, divides it into windows of `window_s` seconds,
    and computes the RMS of each window.  A note is dropped only when every
    window it overlaps is below `thresh`.  Any overlap with an active window
    keeps the note.  Fails open (returns all notes) if the stem file is missing.
    """
    import numpy as np
    import soundfile as sf

    stem_path = get_stem_path()
    if not os.path.isfile(stem_path):
        return notes

    try:
        audio, sr = sf.read(stem_path, always_2d=True, dtype="float32")
    except Exception:
        return notes

    mono        = audio.mean(axis=1, dtype=np.float32)
    win_samples = max(1, int(window_s * sr))
    n_samples   = len(mono)
    n_windows   = (n_samples + win_samples - 1) // win_samples

    padded = np.zeros(n_windows * win_samples, dtype=np.float32)
    padded[:n_samples] = mono
    chunks     = padded.reshape(n_windows, win_samples)
    window_rms = np.sqrt((chunks ** 2).mean(axis=1))

    def _is_active(start_s: float, end_s: float) -> bool:
        lo = max(0, int(start_s / window_s))
        hi = min(n_windows, int(end_s / window_s) + 1)
        return bool(np.any(window_rms[lo:hi] > thresh))

    return [n for n in notes if _is_active(n["start"], n["start"] + n["duration"])]


# ── Stem activity gate (waveform-level, used during separation) ───────────────

def _stem_activity_gate(stem, window: int, threshold: float):
    """
    Returns a per-sample gate tensor in [0, 1] shaped (1, n_samples).

    Gate is 1 where the stem RMS exceeds `threshold` and 0 where it is silent.
    A short box-filter fade smooths the transitions to avoid clicks.
    Multiply raw bleed by this gate so it only appears where the instrument
    is actually playing.
    """
    import torch
    import torch.nn.functional as F

    stem_mono = stem.mean(dim=0)       # (n_samples,)
    n_samples = stem_mono.shape[0]

    # Pad so length is divisible by window, then compute per-window RMS
    pad_len    = (-n_samples) % window
    padded     = F.pad(stem_mono, (0, pad_len))
    chunks     = padded.reshape(-1, window)        # (n_windows, window)
    window_rms = chunks.pow(2).mean(-1).sqrt()     # (n_windows,)

    # Binary gate per window → expand to per sample
    gate = (window_rms > threshold).float()
    gate = gate.repeat_interleave(window)[:n_samples]   # (n_samples,)

    # Smooth transitions with a short box filter to prevent hard clicks
    fade = max(2, window // 2)
    gate = F.avg_pool1d(
        gate.view(1, 1, -1),
        kernel_size=fade,
        stride=1,
        padding=fade // 2,
        count_include_pad=False,
    ).clamp(0.0, 1.0).view(-1)[:n_samples]

    return gate.unsqueeze(0)   # (1, n_samples) — broadcasts over channels


# ── Audio loading ─────────────────────────────────────────────────────────────

def _normalize_loudness(y_np: np.ndarray, sr: int) -> np.ndarray:
    """
    Normalise to SEPARATION_TARGET_LUFS integrated loudness.
    Uses pyloudnorm if available, falls back to RMS normalisation.
    """
    try:
        import pyloudnorm as pyln
        y_t = y_np.T  # (samples, channels)
        meter    = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y_t)
        if not np.isinf(loudness) and not np.isnan(loudness):
            y_t = pyln.normalize.loudness(y_t, loudness, SEPARATION_TARGET_LUFS)
            return np.clip(y_t.T, -1.0, 1.0)
    except Exception:
        pass
    # RMS fallback: normalise to -20 dBFS
    rms = np.sqrt(np.mean(y_np ** 2))
    if rms > 1e-8:
        target_rms = 10 ** (-20.0 / 20.0)
        y_np = y_np * (target_rms / rms)
    return np.clip(y_np, -1.0, 1.0)


def _load_audio_tensor(audio_path: str):
    import torch
    import torchaudio

    ext = os.path.splitext(audio_path)[1].lower()

    if ext in (".wav", ".flac"):
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            # torchaudio may require torchcodec on newer versions — fall back to soundfile
            import soundfile as sf
            import torch
            data, sr = sf.read(audio_path, always_2d=True)
            y = data.T.astype(np.float32)   # (channels, samples)
            if y.shape[0] == 1:
                y = np.repeat(y, 2, axis=0)
            return torch.from_numpy(y), sr
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        return waveform, sr

    import av
    container = av.open(audio_path)
    stream    = next(s for s in container.streams if s.type == "audio")
    sr        = stream.codec_context.sample_rate
    resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout, rate=sr)

    chunks = []
    for frame in container.decode(stream):
        for out_frame in resampler.resample(frame):
            arr = out_frame.to_ndarray()
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            chunks.append(arr.astype(np.float32))
    for out_frame in resampler.resample(None):
        arr = out_frame.to_ndarray()
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        chunks.append(arr.astype(np.float32))
    container.close()

    y = np.concatenate(chunks, axis=-1)
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)

    import torch
    return torch.from_numpy(y), sr
