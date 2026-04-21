"""
pipeline.settings — unified re-export package
===============================================
Importing from this package is identical to importing from the old
pipeline/settings.py module.  All existing `from pipeline.settings import X`
statements in the codebase continue to work without modification.

Settings are split across files by pipeline stage:
  guitar_range.py    — shared MIDI/Hz bounds (used by stages 2, 3, 5b)
  separation.py      — Stage 1: Demucs source separation
  pitch_extraction.py — Stage 2: basic-pitch thresholds and multi-pass config
  note_cleaning.py   — Stage 3: confidence gate, duration filter, merge, polyphony
  quantization.py    — Stage 4: BPM detection and grid-snap
  key_detection.py   — Stage 5: Krumhansl-Schmuckler and scale bias
  guitar_mapping.py  — Stage 6: fretboard mapping + melody isolation
  chord_detection.py — Stage 7: strum grouping and chord validity
  tab_generation.py  — Stage 8: ASCII tab layout
  audio_synthesis.py — Stage 9: synthesized preview WAV
  visualization.py   — Stages 10 & 11: fretboard diagram and chord sheet
"""

from .guitar_range      import *   # noqa: F401,F403
from .separation        import *   # noqa: F401,F403
from .pitch_extraction  import *   # noqa: F401,F403
from .note_cleaning     import *   # noqa: F401,F403
from .quantization      import *   # noqa: F401,F403
from .key_detection     import *   # noqa: F401,F403
from .guitar_mapping    import *   # noqa: F401,F403
from .chord_detection   import *   # noqa: F401,F403
from .tab_generation    import *   # noqa: F401,F403
from .audio_synthesis   import *   # noqa: F401,F403
from .visualization     import *   # noqa: F401,F403
