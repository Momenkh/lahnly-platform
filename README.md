# Lahnly Platform

A music learning platform that lets users upload a song (or paste a YouTube link) and get back everything they need to actually play it on their instrument — tabs, sheet music, chord charts, and more.

---

## What It Does

1. **Source separation** — splits the audio into isolated layers: vocals, guitar, bass, drums, and other instruments
2. **Pitch & note extraction** — detects every note in the layer of interest using a polyphonic ML model
3. **Note cleaning & quantization** — filters noise, merges fragments, snaps to tempo grid; infers time signature (2/4, 3/4, 4/4, 6/8)
4. **Music theory analysis** — detects key, scale, tempo, chord progressions; runs per-section key detection for songs that modulate; infers capo position for guitar
5. **Instrument mapping** — maps notes to real playable positions on the target instrument (frets, strings, etc.), capo-relative where applicable
6. **Output generation** — produces tabs with bar lines, chord sheets, fretboard diagrams, and a synthesized audio preview

Future phases will add cross-instrument translation — so a user can take a guitar solo and get it transposed and mapped for piano, or vice versa (drums excluded).

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

The only service currently implemented. Takes any audio file and runs it through an 11-stage pipeline:

| Stage | What it does |
|-------|-------------|
| 1 | Instrument separation (Demucs neural model, GPU-accelerated) |
| 2 | Pitch extraction (polyphonic ML model) |
| 3 | Note cleaning — duration filter, confidence filter, merge, polyphony cap |
| 4 | Tempo detection & quantization to 16th-note grid; time signature inference (2/4, 3/4, 4/4, 6/8) |
| 5 | Key / scale detection — per-section (detects modulations); capo inference for guitar |
| 5b | Key-confidence feedback filter with chord-tone protection |
| 6 | Instrument mapping (guitar string + fret assignment, capo-relative) |
| 7 | Chord detection (template matching, key-aware) |
| 8 | Tab generation (ASCII guitar tabs with bar lines) |
| 9 | Audio preview (synthesized WAV) |
| 10 | Fretboard diagram (PNG visualization) |
| 11 | Chord sheet (fingering diagrams + chord progression) |

See [`core-processor/readme.md`](core-processor/readme.md) for full usage, CLI options, and pipeline details.

---

## Roadmap

- [x] Core audio processing pipeline (separation → tabs → chords → diagrams)
- [x] Per-section key detection (handles songs that modulate)
- [x] Capo inference and capo-relative tab output
- [x] Chord-tone protection in key-confidence filter
- [x] Time signature inference (2/4, 3/4, 4/4, 6/8) with bar line rendering
- [ ] YouTube link ingestion
- [ ] Backend API & job queue
- [ ] Frontend (upload, playback, results viewer)
- [ ] User accounts and saved transcriptions
- [ ] Cross-instrument translation (e.g. guitar solo → piano notation)
- [ ] Additional instrument support beyond guitar
- [ ] Mobile-friendly output formats

---

## Tech Stack (current)

| Layer | Technology |
|-------|-----------|
| Audio separation | [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`) |
| Pitch detection | librosa pyin (basic-pitch planned once Python 3.14 wheels ship) |
| Music theory | Custom Krumhansl-Schmuckler implementation + maqam support |
| Fret assignment | Viterbi dynamic programming (globally optimal) |
| Visualization | matplotlib |
| Runtime | Python 3.14, PyTorch (CUDA optional) |

---

## Getting Started

Only the `core-processor` service is available right now.

```bash
cd core-processor
pip install -r requirements.txt

# Transcribe a song
python main.py your_song.mp3

# Options
python main.py your_song.mp3 --guitar-type lead   # electric solo
python main.py your_song.mp3 --no-play            # skip playback
python main.py your_song.mp3 --from-stage 3       # resume from stage 3
```

GPU (CUDA) is used automatically if available. CPU fallback works but source separation is significantly slower.
