"""
pipeline.settings — unified settings package
=============================================
All pipeline code does:  from pipeline.settings import SOME_CONSTANT
This file re-exports every constant from every settings sub-module so that
single import statement always works, regardless of which instrument or stage
the constant belongs to.

Layout
------
settings/
  shared/          Constants used by all instruments
    separation     Demucs model cascade, RMS thresholds
    quantization   BPM detection, grid-snap tolerances
    key_detection  Krumhansl-Schmuckler weights, scale bias
    audio_synthesis Sample rate, waveform envelope
    visualization  DPI, section lengths, row heights
  guitar/          Guitar-specific constants
    range          MIDI / Hz bounds for guitar
    pitch          basic-pitch thresholds, multi-pass configs
    cleaning       Confidence gates, duration filters, polyphony limits
    mapping        Fretboard Viterbi DP, string/fret preferences
    chords         Strum grouping, chord naming
    tab            ASCII tab column widths, section lengths
  bass/            Bass-specific constants (same sub-structure as guitar)
  piano/           Piano-specific constants
  vocals/          Vocals-specific constants
  drums/           Drums onset-detection and hit-classification constants
"""

# ── Shared (instrument-agnostic) ──────────────────────────────────────────────
from .shared.separation      import *   # noqa: F401,F403
from .shared.quantization    import *   # noqa: F401,F403
from .shared.key_detection   import *   # noqa: F401,F403
from .shared.audio_synthesis import *   # noqa: F401,F403
from .shared.visualization   import *   # noqa: F401,F403
from .shared.evaluation      import *   # noqa: F401,F403
from .shared.spectral        import *   # noqa: F401,F403
from .shared.presence        import *   # noqa: F401,F403

# ── Guitar ────────────────────────────────────────────────────────────────────
from .guitar.range    import *   # noqa: F401,F403
from .guitar.pitch    import *   # noqa: F401,F403
from .guitar.cleaning import *   # noqa: F401,F403
from .guitar.mapping  import *   # noqa: F401,F403
from .guitar.chords   import *   # noqa: F401,F403
from .guitar.tab      import *   # noqa: F401,F403

# ── Bass ──────────────────────────────────────────────────────────────────────
from .bass.range    import *   # noqa: F401,F403
from .bass.pitch    import *   # noqa: F401,F403
from .bass.cleaning import *   # noqa: F401,F403

# ── Piano ─────────────────────────────────────────────────────────────────────
from .piano.range          import *   # noqa: F401,F403
from .piano.pitch          import *   # noqa: F401,F403
from .piano.cleaning       import *   # noqa: F401,F403
from .piano.chord_recovery import *   # noqa: F401,F403

# ── Vocals ────────────────────────────────────────────────────────────────────
from .vocals.range    import *   # noqa: F401,F403
from .vocals.pitch    import *   # noqa: F401,F403
from .vocals.cleaning import *   # noqa: F401,F403

# ── Drums ─────────────────────────────────────────────────────────────────────
from .drums.onset import *   # noqa: F401,F403
