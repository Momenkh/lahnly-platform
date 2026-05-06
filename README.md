# Lahnly Platform

A music learning platform that lets users upload a song (or paste a YouTube link) and get back everything they need to actually play it on their instrument — tabs, sheet music, chord charts, and more.

---

## What It Does

Given any audio file, the platform:

1. **Source separation** — splits the audio into isolated layers: vocals, guitar, bass, drums, piano, and other instruments
2. **Per-instrument transcription** — runs a dedicated extraction pipeline for each instrument with instrument-specific detection models and parameters
3. **Note cleaning & quantization** — filters noise, merges fragments, snaps to tempo grid, infers time signature (2/4, 3/4, 4/4, 6/8)
4. **Music theory analysis** — detects key, scale, tempo, chord progressions; runs per-section key detection for songs that modulate; infers capo position for guitar
5. **Instrument mapping** — maps notes to real playable positions (guitar/bass frets, piano keys, drum hits)
6. **Output generation** — produces tabs, piano rolls, chord sheets, fretboard diagrams, melodic contour overlays, drum grids, and synthesized audio previews

---

## Platform Architecture

The platform is planned as a set of independent services:

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (TBD)                   │
│         Song upload / YouTube link / playback       │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│                 API Gateway / Backend (TBD)         │
│          Auth, job queue, user management           │
└──────┬──────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────┐
       │                                      │
┌──────▼──────────────────┐      ┌────────────▼──────────────────┐
│    core-processor  ✓    │      │   other services (planned)    │
│  Audio → tabs/chords/   │      │   YouTube ingestion, storage, │
│  fretboard diagrams     │      │   user library, export, etc.  │
└─────────────────────────┘      └───────────────────────────────┘
```

---

## Services

### `core-processor` — Audio Processing Pipeline

> **Status: active development**

The only service currently implemented. Takes any audio file and runs independent transcription pipelines for all five instruments simultaneously or for a specific instrument on request.

**Supported instruments:**

| Instrument | Output | Notes |
|------------|--------|-------|
| Guitar | ASCII tabs, fretboard diagram, chord sheet | Viterbi DP string/fret assignment; capo inference; 6 tonal modes |
| Bass | 4-string ASCII tabs, fretboard diagram | Monophonic mapping; slap vs fingered style detection |
| Piano | Piano roll (PNG), chord sheet | Left/right hand split; sustain-aware note merging |
| Vocals | Melodic contour overlay on spectrogram | CREPE + basic-pitch fusion; monophonic F0 tracking |
| Drums | ASCII drum grid notation, hit visualization | Onset detection + classification into kick/snare/hihat/tom/cymbal |

See [`core-processor/readme.md`](core-processor/readme.md) for full usage, CLI options, and pipeline details.

---

## Roadmap

- [x] Core audio processing pipeline (separation → transcription → output for all 5 instruments)
- [x] Guitar: Viterbi DP fret assignment, per-section key detection, capo inference, chord-tone protection
- [x] Guitar: Time signature inference (2/4, 3/4, 4/4, 6/8) with bar line rendering in tabs
- [x] Bass: 4-string mapping, style auto-detection (slap vs fingered)
- [x] Piano: Piano roll, left/right hand split, chord recovery
- [x] Vocals: CREPE + basic-pitch fusion, melodic contour visualization
- [x] Drums: Onset detection, hit classification, ASCII drum grid notation
- [ ] YouTube link ingestion
- [ ] Backend API & job queue
- [ ] Frontend (upload, playback, results viewer)
- [ ] User accounts and saved transcriptions
- [ ] Cross-instrument translation (e.g. guitar solo → piano notation)
- [ ] Mobile-friendly output formats

---

## Tech Stack (current)

| Layer | Technology |
|-------|-----------|
| Audio separation | [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`) |
| Pitch detection (guitar/bass/piano) | basic-pitch (Spotify polyphonic ML model) |
| Pitch detection (vocals) | CREPE (monophonic harmonic F0) + basic-pitch verification |
| Drum detection | librosa onset detection + spectral classification |
| Music theory | Custom Krumhansl-Schmuckler + maqam support |
| Fret assignment | Viterbi dynamic programming (globally optimal) |
| Visualization | matplotlib |
| Runtime | Python 3.14, PyTorch (CUDA optional) |

---

## Getting Started

Only the `core-processor` service is available right now.

```bash
cd core-processor
pip install -r requirements.txt

# Transcribe all instruments in a song
python main.py your_song.mp3 --instrument all

# Transcribe a specific instrument
python main.py your_song.mp3 --instrument guitar
python main.py your_song.mp3 --instrument bass
python main.py your_song.mp3 --instrument piano
python main.py your_song.mp3 --instrument vocals
python main.py your_song.mp3 --instrument drums

# Guitar-specific options
python main.py your_song.mp3 --instrument guitar --type lead
python main.py your_song.mp3 --instrument guitar --no-play
python main.py your_song.mp3 --instrument guitar --from-stage 3
```

GPU (CUDA) is used automatically if available. CPU fallback works but source separation is significantly slower.
