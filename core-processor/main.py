"""
AI Music Transcription Platform
End-to-end pipeline: audio file -> guitar tabs + playback + visualization

Usage:
    python main.py <audio_file> [options]

Pipeline stages:
    1  Instrument separation  (Demucs — extracts guitar stem)
    2  Pitch extraction       (basic-pitch polyphonic ML model)
    3  Note cleaning          (filters, merges, polyphony limit)
    4  Quantization           (tempo detection, snap to 16th-note grid)
    5  Key / scale analysis   (Krumhansl-Schmuckler)
    5b Key-context octave correction
    6  Guitar mapping         (string + fret assignment)
    7  Chord detection        (groups simultaneous notes, names chords)
    8  Tab generation         (ASCII guitar tab — solo notes only)
    9  Audio preview          (synthesized WAV + optional playback)
   10  Fretboard diagram      (PNG visualization)
   11  Chord sheet            (PNG chord box diagrams + progression)

Options:
    --instrument I  guitar (default) | bass | piano | vocals | drums | all
                  Which instrument to transcribe.  Use 'all' to run every
                  instrument in one pass (guitar separation caches all stems).
    --type T      acoustic | clean | distorted  (default: clean)
                  Tonal character of the guitar.
                  acoustic   = nylon/steel string — middle-ground thresholds
                  clean      = clean electric — default electric settings
                  distorted  = distorted electric — HPSS pre-filter + tighter thresholds
    --role R      lead | rhythm  (default: rhythm)
                  Playing style.
                  lead       = single-note solo — lower polyphony, melody isolation
                  rhythm     = chords / rhythm parts — full polyphony, all notes used
    --start MM:SS     Only process notes from this time onward
    --end   MM:SS     Only process notes up to this time
    --bpm-override N  Force BPM to N instead of auto-detecting
    --force-tempo     Re-detect BPM even if a saved tempo exists
    --score           Compute chroma similarity score (stem vs preview) and print it
    --no-play         Skip audio playback (still saves 09_preview.wav)
    --no-viz          Skip fretboard and chord sheet images
    --no-separate     Skip separation (use raw mix for pitch detection)
    --no-quantize     Skip tempo detection and note quantization
    --from-stage N    Resume from stage N (2-11); requires prior run outputs
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.config import set_outputs_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Guitar tab transcription pipeline")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (WAV, FLAC, M4A, MP3)")
    parser.add_argument(
        "--instrument",
        choices=["guitar", "bass", "piano", "vocals", "drums", "all"],
        default="guitar",
        help="Instrument to transcribe: guitar (default) | bass | piano | vocals | drums | all",
    )
    parser.add_argument("--score",         action="store_true", help="Compute chroma similarity score (stem vs preview) after processing")
    parser.add_argument("--no-play",      action="store_true", help="Skip audio playback")
    parser.add_argument("--no-viz",       action="store_true", help="Skip visualization")
    parser.add_argument("--no-separate",  action="store_true", help="Skip separation (use raw mix)")
    parser.add_argument("--no-quantize",  action="store_true", help="Skip tempo detection and quantization")
    parser.add_argument("--force-tempo",  action="store_true", help="Re-detect BPM even if saved tempo exists")
    parser.add_argument(
        "--type",
        choices=["acoustic", "clean", "distorted"],
        default=None,
        dest="guitar_type",
        help="Guitar type: acoustic, clean, distorted (default: auto-detected from stem)",
    )
    parser.add_argument(
        "--role",
        choices=["lead", "rhythm"],
        default=None,
        dest="guitar_role",
        help="Guitar role: lead or rhythm (default: auto-detected from stem)",
    )
    parser.add_argument(
        "--bpm-override",
        type=float,
        default=None,
        metavar="BPM",
        help="Force a specific BPM instead of auto-detecting",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="MM:SS",
        help="Only process notes starting from this timestamp (e.g. 5:25 or 325)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="MM:SS",
        help="Only process notes up to this timestamp (e.g. 6:10 or 370)",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=2,
        metavar="N",
        help="Resume from stage N (2-11). Requires prior run to have saved outputs.",
    )
    parser.add_argument(
        "--vocals-repet",
        action="store_true",
        help="Enable REPET-SIM background removal before CREPE pitch extraction for vocals "
             "(use for songs with a heavy sustained instrumental bed under the vocals)",
    )
    return parser.parse_args()



def main():
    args = parse_args()

    if args.audio_file:
        out_dir = set_outputs_dir(args.audio_file)
        print(f"Outputs -> {out_dir}")

    if args.from_stage <= 1 and not args.audio_file:
        print("Error: audio_file is required unless --from-stage > 1")
        sys.exit(1)

    if args.audio_file and not os.path.isfile(args.audio_file):
        print(f"Error: file not found: {args.audio_file}")
        sys.exit(1)

    # ── Dispatch to instrument-specific pipeline ─────────────────────────────
    from pipeline.registry import REGISTRY

    if args.instrument == "all":
        # guitar runs first — htdemucs_6s caches all 6 stems in one Demucs pass,
        # so bass, piano, vocals, drums find their stems already on disk and skip Stage 1.
        order = ["guitar", "bass", "piano", "vocals", "drums"]
        for name in order:
            print(f"\n{'=' * 60}")
            print(f"  {name.upper()}")
            print(f"{'=' * 60}")
            REGISTRY[name].run_pipeline(args)
        return

    if args.instrument not in REGISTRY:
        print(f"Error: instrument '{args.instrument}' not yet implemented.")
        sys.exit(1)
    REGISTRY[args.instrument].run_pipeline(args)


if __name__ == "__main__":
    main()
