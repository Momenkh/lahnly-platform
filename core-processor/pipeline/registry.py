"""
Instrument Registry
====================
Maps instrument names to pipeline runners.

Adding a new instrument:
  1. Create pipeline/instruments/<instrument>/pipeline.py with run_<instrument>_pipeline(args)
  2. Import and register it in REGISTRY below.
  3. The --instrument flag in main.py dispatches generically via REGISTRY.
"""

from dataclasses import dataclass
from typing import Callable

from pipeline.instruments.guitar.pipeline import run_guitar_pipeline
from pipeline.instruments.bass.pipeline   import run_bass_pipeline
from pipeline.instruments.piano.pipeline  import run_piano_pipeline
from pipeline.instruments.vocals.pipeline import run_vocals_pipeline
from pipeline.instruments.drums.pipeline  import run_drums_pipeline


@dataclass
class InstrumentProfile:
    name: str
    run_pipeline: Callable   # run_pipeline(args) -> None


REGISTRY: dict[str, InstrumentProfile] = {
    "guitar": InstrumentProfile(
        name="guitar",
        run_pipeline=run_guitar_pipeline,
    ),
    "bass": InstrumentProfile(
        name="bass",
        run_pipeline=run_bass_pipeline,
    ),
    "piano": InstrumentProfile(
        name="piano",
        run_pipeline=run_piano_pipeline,
    ),
    "vocals": InstrumentProfile(
        name="vocals",
        run_pipeline=run_vocals_pipeline,
    ),
    "drums": InstrumentProfile(
        name="drums",
        run_pipeline=run_drums_pipeline,
    ),
}
