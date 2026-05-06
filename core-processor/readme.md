# core-processor

The audio processing engine behind the Lahnly platform. Takes any audio file and runs it through an end-to-end pipeline that separates instrument layers, extracts notes, analyzes music theory, and produces playable output — tabs, chord sheets, diagrams, and a synthesized preview.

> **Current scope:** The pipeline is built around guitar output (tabs, fretboard diagrams). The underlying stages — separation, pitch extraction, note cleaning, key detection, chord detection — are instrument-agnostic by design, so extending the output layer to other instruments is the planned next step.

---

## What It Does

Given any audio file (MP3, WAV, FLAC, M4A), the pipeline:

1. Separates the mix into instrument layers using a neural source separation model
2. **Auto-detects the guitar type and role from the stem** (or accepts explicit flags)
3. Extracts every note using a polyphonic ML pitch detector with multi-pass thresholding
4. Cleans, filters, and quantizes the notes to a musical grid; infers time signature (2/4, 3/4, 4/4, 6/8) from onset patterns
5. Detects the song's key, scale, and tempo — per section for songs that modulate — and infers capo position for guitar
6. Protects chord tones from key-confidence deletion, then filters remaining off-key low-confidence noise
7. Maps every note to a playable fretboard position using global Viterbi DP, capo-relative where a capo was detected
8. Identifies chords from simultaneous notes
9. Generates tabs with bar lines, chord sheets, a fretboard diagram, and a synthesized audio preview

---

## Usage

```bash
# Full pipeline — type and role are auto-detected from the stem
python main.py song.mp3

# Skip audio playback (still saves 09_preview.wav)
python main.py song.mp3 --no-play

# Skip image output
python main.py song.mp3 --no-viz

# Resume from stage 3 — reuse saved stem and raw notes
python main.py song.mp3 --from-stage 3

# Skip quantization (useful for free-tempo / rubato recordings)
python main.py song.mp3 --no-quantize

# Specify type and/or role explicitly (overrides auto-detection)
python main.py song.mp3 --type clean --role lead
python main.py song.mp3 --type distorted --role rhythm
python main.py song.mp3 --type acoustic --role lead
python main.py song.mp3 --role lead     # auto-detect type, force role

# Score the transcription quality after processing
python main.py song.mp3 --score

# Batch benchmark all songs in the current directory
python batch_score.py
python batch_score.py --type lead       # only lead-role songs
python batch_score.py --from-stage 2    # skip re-separation
```

---

## Guitar Type + Role

The pipeline separates two independent dimensions of guitar playing that previously were conflated under a single `--guitar-type` flag.

### `--type` — Tonal character of the instrument

| Type | Description | Key Processing Difference |
|------|-------------|--------------------------|
| `acoustic` | Nylon or steel string acoustic | Middle-ground thresholds; bend detection on |
| `clean` | Clean electric guitar | Default electric settings |
| `distorted` | Distorted/overdriven electric | HPSS pre-filter removes percussive harmonics before pitch detection; tighter thresholds |

### `--role` — Playing style in the song

| Role | Description | Polyphony | Melody Isolation |
|------|-------------|-----------|-----------------|
| `lead` | Single-note solo, runs, bends | Max 2 simultaneous notes | Yes — top voice only used for tabs and preview |
| `rhythm` | Chord strumming, accompaniment | Max 5–6 simultaneous notes | No — all notes used |

These combine into six valid modes: `acoustic_lead`, `acoustic_rhythm`, `clean_lead`, `clean_rhythm`, `distorted_lead`, `distorted_rhythm`. Each has independently tuned confidence thresholds, polyphony limits, and minimum note durations.

### Auto-detection (Stage 1.5)

When `--type` and/or `--role` are omitted, the pipeline detects them automatically from the separated stem using fast spectral features (~2–3 seconds):

- **Type** — Spectral flatness (distortion leaves a noise-like spectrum even after separation), spectral contrast, and spectral centroid
- **Role** — Chroma-based polyphony estimate (active simultaneous pitch classes per frame) and onset density

Each detection comes with a confidence score. Low-confidence results print a warning suggesting you use the explicit flag:

```
[Auto] Detecting guitar type + role from: 01_guitar_stem.wav
[Auto] flatness=0.0890  contrast=22.1  centroid=2180Hz  polyphony=1.8  onsets/s=4.2
[Auto] Type  -> distorted   (81% confidence)
[Auto] Role  -> lead        (74% confidence)
```

---

## Scoring

The `--score` flag computes a **chroma similarity score** between the separated guitar stem and the synthesized preview after the full pipeline runs. It measures how well the detected pitches match the original recording.

```
[Score] Chroma similarity (stem vs preview): 0.766
        (0.55 = poor  |  0.70 = usable  |  0.85 = very good)
```

The score uses librosa CQT chroma features compared frame-by-frame with cosine similarity. It is insensitive to octave errors (chroma folds octaves) and timing offsets, so it rewards pitch accuracy over rhythmic precision.

### Batch Benchmarking

`batch_score.py` auto-discovers all audio files in the current directory, infers type+role from the filename, runs the full pipeline, and appends timestamped results to `benchmark_results.json`:

```bash
python batch_score.py                  # all songs
python batch_score.py --type lead      # lead-role songs only
python batch_score.py --type distorted # distorted-type songs only
python batch_score.py --from-stage 2   # skip separation, reuse saved stems
python batch_score.py --no-save        # print results only
```

Filename inference rules: `"lead"`/`"solo"` in name → role=lead; `"acoustic"` → type=acoustic; `"distorted"`/`"dist"` → type=distorted; no role keyword → two runs (rhythm + lead).

Results are stored in `benchmark_results.json` with `guitar_type`, `guitar_role`, `mode`, `score`, and `timestamp` fields.

---

## Pipeline Stages

All outputs are saved to `outputs/<song_name>/`.

| Stage | Name | Output Files |
|-------|------|-------------|
| 1 | Instrument Separation | `01_stem.wav`, `01_stem_meta.json` |
| 1.5 | Auto Type+Role Detection | *(in memory — sets `--type` and `--role` if not specified)* |
| 2 | Pitch Extraction | `02_raw_notes.json`, `02_pass*_preview.wav` |
| 3 | Note Cleaning | `03_cleaned_notes.json`, `03_clean_meta.json` |
| 4 | Tempo, Quantization & Time Signature | `04_quantized_notes.json`, `04_tempo.json` |
| 5 | Key / Scale Detection (per-section) + Capo Inference | `05_key_analysis.json` |
| 5b | Chord-Tone Protection + Key-Confidence Filter | *(in memory)* |
| 6 | Guitar Mapping (capo-relative) | `06_mapped_notes.json` |
| 7 | Chord Detection | `07_chords.json` |
| 8 | Tab Generation (with bar lines) | `08_tabs.txt` |
| 9 | Audio Preview | `09_preview.wav` |
| 10 | Fretboard Diagram | `10_fretboard.png` |
| 11 | Chord Sheet | `11_chord_sheet.png` |

Use `--from-stage N` to skip stages 1 through N-1 and reuse their saved outputs.

---

## Pipeline Flow

```
Audio File (MP3/WAV/FLAC/M4A)
        |
        v
  Stage 1: Separation
    Demucs htdemucs_6s (GPU-accelerated, 6-stem model)
    Falls back: htdemucs_ft -> htdemucs -> raw mix
    Outputs: stem_confidence (0-1) propagated to all later stages
        |
        v
  Stage 1.5: Auto Type+Role Detection  (skipped if --type and --role both set)
    Spectral flatness    -> distortion detection
    Spectral contrast    -> acoustic vs clean
    Chroma polyphony     -> lead vs rhythm
    Outputs: guitar_type, guitar_role with confidence scores
        |
        v
  Stage 2: Pitch Extraction
    basic-pitch polyphonic ML model (CUDA-capable)
    Multi-pass: model inference once, note extraction 2x at different thresholds
    Distorted type: HPSS pre-filter (harmonic/percussive separation) before inference
    Per-note confidence = mean frame-level detection probability
    Confidence merge: strict-pass confirmed -> weight 1.00; base-only -> 0.65
        |
        v
  Stage 3: Note Cleaning
    1. Stem energy gate  — drop ghost notes from silent stem regions
    2. Duration filter   — BPM-aware minimum note duration
    3. Confidence filter — adaptive threshold from stem quality + mode
    4. Bass filter       — stricter gate below C2
    5. Bend merge        — collapse semitone-adjacent consecutive notes (lead/acoustic)
    6. Fragment merge    — merge same-pitch notes with small gaps
    7. Local pitch filter— remove outlier pitches vs local median
    8. Polyphony limit   — evict lowest-confidence / highest-pitch extras
        |
        v
  Stage 4: Tempo, Quantization & Time Signature
    librosa beat tracking on densest 60s window of note activity
    Half/double-time BPM correction (tests bpm, bpm/2, bpm*2)
    Snaps note start times to nearest 16th-note grid (35% tolerance)
    Snaps note durations to nearest subdivision (very short articulations exempt)
    Time signature inference via FFT onset autocorrelation:
      samples AC at 2x, 3x, 4x beat lags; 4/4 bias for ambiguous patterns
      6/8 vs 3/4 discrimination via dotted-quarter lag ratio
    Outputs: bpm, subdivision_s, time_sig_num, time_sig_den
        |
        v
  Stage 5: Key / Scale Detection (per-section) + Capo Inference
    Splits note stream at silence gaps >= 2s (min 16 notes, min 15s per section)
    Krumhansl-Schmuckler algorithm independently per section
    Dual-weighted pitch histogram: 50% duration + 50% onset count
    Searches all 24 major/minor keys + Hijaz / Kurd / Saba maqamat
    Sparse sections inherit key from the nearest reliable neighbor
    Capo inference: scans capo 1-5, picks position maximising open-string fit
      (how many of E/A/D/G/B fall in the key's diatonic scale)
    Returns: global key, per-section key list, capo_fret (0 = none), confidence
        |
        v
  Stage 5b: Chord-Tone Protection + Key-Confidence Filter
    Pre-chord pass: groups simultaneous notes into chord candidates
    Collects all pitch classes found in qualifying chord groups -> protected set
    Deletes notes only if: off-key AND low-confidence AND NOT in protected set
    Per-section key used for in-key check (falls back to global key)
        |
        v
  Stage 6: Guitar Mapping (capo-relative)
    Effective tuning = standard tuning shifted up by capo_fret semitones
    Viterbi DP — globally optimal (string, fret) assignment over full sequence
    Minimises: hand shift cost + string change penalty + fret height penalty
    Open-string-in-key bonus; simultaneous-note same-string penalty
    Context-aware fret window: passage median anchors position globally
    Fret numbers output are capo-relative (fret 0 = nut or capo position)
    Melody isolation (lead role): top-voice notes routed to tabs + preview;
                                  remaining harmony notes go to chord detection
        |
        +---------------------------+
        |                           |
        v                           v
  Stage 7: Chord Detection     (melody or all notes)
    Groups notes within strum      |
    window (~40ms)                 |
    Names chords via template      |
    matching with penalty scoring  |
    Key-aware tie-breaking         |
        |                           |
        +----------+                |
        |          |                |
        v          v                v
  Stage 8      Stage 11        Stage 9 & 10
  Tab          Chord Sheet     Audio Preview (Karplus-Strong synthesis)
  Generation   (diagrams +     Fretboard Diagram
  (bar lines   progression)
  from time
  signature)
```

---

## Music Theory Layer

### Key Detection

The Krumhansl-Schmuckler (KS) algorithm correlates the song's pitch class histogram against 24 major/minor key profiles. The histogram is dual-weighted: 50% note duration + 50% onset count, which prevents long open-string sustains from dominating the result.

**Per-section detection** splits the note stream at silence gaps ≥ 2 seconds and runs independent key detection per section. This handles songs that modulate between keys or shift between verse and chorus tonalities. Sections with too few notes (< 16) inherit the key from the nearest reliable neighbor.

**Maqam support** extends detection to three Arabic modes (Hijaz, Kurd, Saba) with guard gates: a maqam result requires ≥ 80 notes and a 0.04 score gap over the nearest Western key. This prevents false Arabic-mode classifications on sparse Western songs.

### Capo Inference

After key detection, the pipeline tests capo positions 1–5 by measuring how many of the five guitar open-string pitch classes (E, A, D, G, B) fall within the key's diatonic scale at each transposition. If a capo position scores higher than the no-capo baseline and meets the minimum threshold (4 of 5 strings), it is applied. All downstream fret numbers are then capo-relative, and the tab header includes `# Capo: N`.

Maqam songs never receive a capo annotation.

### Chord-Tone Protection

Before the key-confidence feedback filter runs, a lightweight pre-chord pass identifies all pitch classes that appear in valid chord groups (≥ 3 simultaneous notes within the strum window). These pitch classes are added to a protected set. The filter then skips deletion for any note whose pitch class is in that set, even if the note is off-key and low-confidence. This prevents the filter from removing chord tones that happen to be chromatic (e.g., a major III chord in a minor key, borrowed bVII, or a secondary dominant).

### Time Signature Inference

After BPM detection, note onsets are binned onto a fine grid (beat/16 resolution) and the autocorrelation is computed via FFT. The autocorrelation is sampled at 2×, 3×, and 4× the beat period; the candidate with the highest value wins, with a small bias toward 4/4 to avoid flipping common-time songs on sparse onset patterns.

6/8 vs 3/4 disambiguation: if the 3-beat candidate wins, the pipeline also samples the autocorrelation at 1.5× the beat period (dotted-quarter pulse). If that value is ≥ 85% of the 3-beat value, the song is classified as 6/8 rather than 3/4.

The detected time signature is stored in `04_tempo.json` and used by the tab renderer to place bar lines at the correct column positions.

---

## Audio Preview

The synthesized preview (`09_preview.wav`) uses **Karplus-Strong physical string modelling** — a delay-line IIR filter that produces a realistic plucked-string timbre without any sample library:

- Half-sine pulse excitation (warm attack, no harsh crack)
- One-pole lowpass recirculation with configurable damping
- Cosine fade-out on each note's tail (eliminates click/pop at note boundaries)
- DC-blocking high-pass filter on the final mix (removes low-frequency rumble from accumulated note offsets)

---

## Architecture Notes

- **Separation:** Demucs `htdemucs_6s` (6-stem model). Falls back to `htdemucs_ft` → `htdemucs` → raw mix. Input is loudness-normalised to -16 LUFS before separation.

- **Pitch detection:** basic-pitch polyphonic ML model (Spotify, ICASSP 2022). Multi-pass strategy: the neural network runs once per song; note extraction post-processing runs twice at different thresholds and results are merged by confidence weight. Distorted guitar additionally runs HPSS (librosa harmonic/percussive separation) before the model to suppress saturated harmonics.

- **Confidence:** Per-note mean frame-level detection probability from the basic-pitch model — not velocity. Propagated through all downstream stages and used for polyphony eviction, visualization shading, and key-context filtering.

- **Guitar mapping:** Viterbi dynamic programming over the full note sequence. Globally optimal fret assignments minimize total hand travel + string change cost. Open-string-in-key positions receive a bonus; same-string chord notes receive a large penalty (physically impossible on guitar). When a capo is detected, the effective string tuning is shifted and all output fret numbers are capo-relative.

- **Key detection:** Krumhansl-Schmuckler algorithm with dual-weighted pitch class histogram (50% duration + 50% onset count). Runs per-section to handle modulating songs. Returns top-3 candidates with confidence, plus capo_fret and per-section segment list.

- **BPM & Time Signature:** librosa beat tracking on the densest 60-second window of the recording, with automatic half/double-time correction. FFT-based onset autocorrelation then infers time signature (2/4, 3/4, 4/4, 6/8).

- **Chord naming:** Template matching against 15 chord types with penalty scoring (`matches - 0.3 × extra pitch classes`) and key-aware tie-breaking.

- **Tab bar lines:** Column spacing is `bar_cols = (ts_num × 4 × subdivision) // ts_den`, giving 16 columns/bar for 4/4, 12 for 3/4 and 6/8, 8 for 2/4. Bar lines are also inserted inside note duration spans that cross a bar boundary.

---

## Settings

All tunable parameters live in `pipeline/settings/` — one file per pipeline stage. Nothing is hardcoded in the processing code. Each file has extensive comments explaining what each constant controls and how to tune it.

| File | Controls |
|------|----------|
| `guitar_range.py` | MIDI/Hz bounds for all pipeline stages |
| `separation.py` | Demucs model selection, silence thresholds, LUFS target |
| `pitch_extraction.py` | Per-mode thresholds, multi-pass configs, HPSS margin |
| `note_cleaning.py` | Confidence gates, duration filters, merge gaps, polyphony limits |
| `quantization.py` | BPM detection window, grid snap tolerance, time signature candidates and bias |
| `key_detection.py` | KS algorithm weights, scale bias, capo search range, segment split parameters |
| `guitar_mapping.py` | Viterbi transition costs, fret penalties, melody isolation pitches |
| `chord_detection.py` | Strum window, chord template scoring |
| `tab_generation.py` | Column width, block length |
| `audio_synthesis.py` | Karplus-Strong damping, amplitude scaling |
| `visualization.py` | Diagram dimensions, color scheme |

---

## Future Plans

- **Multi-instrument output** — piano roll, sheet music, bass tabs alongside or instead of guitar tabs
- **Layer selection** — let the user choose which separated layer to transcribe (e.g. bass instead of guitar)
- **YouTube ingestion** — accept a URL instead of a local file
- **Cross-instrument translation** — remap a note sequence from one instrument to another
- **Expanded maqam support** — Nahawand, Ajam, Bayati, and other common Arabic modes
- **Alternate tuning detection** — Drop D, Open G/D, DADGAD
- **Roman numeral chord labels** — I, IV, V, vi alongside raw chord names

---

## Requirements

```
torch torchaudio demucs
basic-pitch
librosa
numpy soundfile scipy
matplotlib pygame-ce
av
```

GPU (CUDA) is used automatically if available. CPU fallback works but separation is significantly slower (~5× on a typical song).
