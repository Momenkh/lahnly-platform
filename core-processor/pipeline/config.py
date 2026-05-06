"""
Shared runtime config — set once in main.py before any stage runs.
Each pipeline module imports get_outputs_dir() instead of hardcoding a path.
"""

import os

_outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def set_outputs_dir(audio_path: str) -> str:
    """
    Derive a per-song output directory from the audio filename and set it globally.
    e.g. A.mp3 -> outputs/A/
    Returns the directory path.
    """
    global _outputs_dir
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    _outputs_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "outputs", stem
    )
    os.makedirs(_outputs_dir, exist_ok=True)
    return _outputs_dir


def get_outputs_dir() -> str:
    os.makedirs(_outputs_dir, exist_ok=True)
    return _outputs_dir


def get_instrument_dir(instrument: str) -> str:
    """Return (and create) the per-instrument subdirectory: outputs/{song}/{instrument}/"""
    path = os.path.join(get_outputs_dir(), instrument)
    os.makedirs(path, exist_ok=True)
    return path


def get_shared_dir() -> str:
    """Return (and create) the shared/ subdirectory for cross-instrument outputs."""
    path = os.path.join(get_outputs_dir(), "shared")
    os.makedirs(path, exist_ok=True)
    return path
