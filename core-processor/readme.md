# core-processor

The audio processing engine behind the Lahnly platform. Takes any audio file and runs independent end-to-end transcription pipelines for up to five instruments simultaneously — guitar, bass, piano, vocals, and drums. Each pipeline handles separation, pitch/onset extraction, note cleaning, music theory analysis, instrument mapping, and output generation.

---

## Supported Instruments

| Instrument | Primary Extractor | Output |
|------------|------------------|--------|
| **Guitar** | basic-pitch (polyphonic ML) | ASCII tabs, fretboard diagram, chord sheet, audio preview |
| **Bass** | basic-pitch (monophonic tuned) | 4-string ASCII tabs, fretboard diagram, audio preview |
| **Piano** | basic-pitch (polyphonic ML) | Piano roll (PNG), chord sheet, audio preview |
| **Vocals** | CREPE + basic-pitch fusion | Melodic contour on spectrogram (PNG) |
| **Drums** | Onset detection + spectral classifier | ASCII drum grid, hit visualization |

---

## Usage

```bash
# Transcribe all instruments
python main.py song.mp3 --instrument all

# Transcribe a specific instrument
python main.py song.mp3 --instrument guitar
python main.py song.mp3 --instrument bass
python main.py song.mp3 --instrument piano
python main.py song.mp3 --instrument vocals
python main.py song.mp3 --instrument drums

# Skip audio playback (still saves preview WAV)
python main.py song.mp3 --no-play

# Skip image output
python main.py song.mp3 --no-viz

# Resume from stage 3 — reuse saved stem and raw notes
python main.py song.mp3 --from-stage 3

# Skip quantization (useful for free-tempo / rubato recordings)
python main.py song.mp3 --no-quantize

# Guitar type and role (overrides auto-detection)
python main.py song.mp3 --instrument guitar --type clean --role lead
python main.py song.mp3 --instrument guitar --type distorted --role rhythm
python main.py song.mp3 --instrument guitar --type acoustic

# BPM override (all instruments)
python main.py song.mp3 --bpm-override 120

# Trim audio before processing (seconds)
python main.py song.mp3 --start 30 --end 90

# Score the transcription quality after processing
python main.py song.mp3 --score

# Batch benchmark all songs in the current directory
python batch_score.py
python batch_score.py --type lead       # only lead-role songs
python batch_score.py --from-stage 2    # skip re-separation

# Verbose per-stage diagnostics
python diagnose.py song.mp3

# Run all five instrument pipelines in one shot and compare scores
python test_all_instruments.py song.mp3
```

---

## Guitar Type + Role

The guitar pipeline separates two independent dimensions of playing style.

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

These combine into six tuned modes: `acoustic_lead`, `acoustic_rhythm`, `clean_lead`, `clean_rhythm`, `distorted_lead`, `distorted_rhythm`. Each has independently calibrated confidence thresholds, polyphony limits, and minimum note durations.

### Auto-detection (Stage 1.5)

When `--type` and/or `--role` are omitted, the pipeline detects them automatically from the separated stem using fast spectral features (~2–3 seconds):

- **Type** — Spectral flatness (distortion leaves a noise-like spectrum even after separation), spectral contrast, spectral centroid
- **Role** — Chroma-based polyphony estimate (active simultaneous pitch classes per frame), onset density

```
[Auto] Detecting guitar type + role from: 01_guitar_stem.wav
[Auto] flatness=0.0890  contrast=22.1  centroid=2180Hz  polyphony=1.8  onsets/s=4.2
[Auto] Type  -> distorted   (81% confidence)
[Auto] Role  -> lead        (74% confidence)
```

The same auto-detection logic applies to bass (slap vs fingered), piano (sparse vs dense playing), vocals (clean vs breathy), and drums (tight vs loose feel).

---

## Scoring

The `--score` flag computes a **chroma similarity score** between the separated stem and the synthesized preview after the full pipeline runs.

```
[Score] Chroma similarity (stem vs preview): 0.766
        (0.55 = poor  |  0.70 = usable  |  0.85 = very good)
```

The score uses librosa CQT chroma features compared frame-by-frame with cosine similarity. It rewards pitch accuracy over rhythmic precision (chroma folds octaves; timing offsets are aligned before scoring). Drums use an F1-based onset matching score instead of chroma similarity.

### Batch Benchmarking

`batch_score.py` auto-discovers all audio files in the current directory, infers type+role from filename, runs the full pipeline, and appends timestamped results to `benchmark_results.json`:

```bash
python batch_score.py                  # all songs
python batch_score.py --type lead      # lead-role songs only
python batch_score.py --from-stage 2   # skip separation, reuse saved stems
python batch_score.py --no-save        # print results only
```

### Diagnostics

`diagnose.py` runs the pipeline with verbose per-stage logging — note counts, filtering decisions, confidence distributions, BPM detection steps — to help debug poor transcription quality on a specific song.

### Multi-Instrument Integration Test

`test_all_instruments.py` processes a single audio file through all five instrument pipelines in one run, reports per-stage timing and quality scores side-by-side, and highlights any pipeline failures or regressions.

---

## Pipeline Stages

All outputs are saved to `outputs/<song_name>/`.

### Shared outputs (one per song, all instruments read these)

| Stage | Name | Output |
|-------|------|--------|
| 1 | Instrument Separation | `shared/01_<instrument>_stem.wav` per instrument |
| 4 | Tempo, Quantization & Time Signature | `shared/04_quantized_notes.json`, `shared/04_tempo.json` |
| 5 | Key / Scale Detection | `shared/05_key_analysis.json` |

### Per-instrument outputs

| Stage | Guitar | Bass | Piano | Vocals | Drums |
|-------|--------|------|-------|--------|-------|
| 1.5 | Type+role auto-detect | Style auto-detect | Mode auto-detect | Mode auto-detect | Character auto-detect |
| 2 | `02_raw_notes.json` | `02_raw_notes.json` | `02_raw_notes.json` | `02_raw_notes.json` | `02_drum_hits.json` |
| 3 | `03_cleaned_notes.json` | `03_cleaned_notes.json` | `03_cleaned_notes.json` | `03_cleaned_notes.json` | *(onset-based, no cleaning)* |
| 5b | Key-conf filter (in memory) | *(skipped)* | *(skipped)* | *(skipped)* | *(skipped)* |
| 6 | `06_mapped_notes.json` | `06_mapped_notes.json` | `06_piano_keys.json` | *(no mapping)* | *(classification only)* |
| 7 | `07_chords.json` | *(skipped)* | `07_chords.json` | *(skipped)* | *(skipped)* |
| 8 | `08_tabs.txt` | `08_tabs.txt` | *(piano roll)* | *(contour PNG)* | `08_drum_grid.txt` |
| 9 | `09_preview.wav` | `09_preview.wav` | `09_preview.wav` | *(skipped)* | *(skipped)* |
| 10 | `10_fretboard.png` | `10_fretboard.png` | `10_piano_roll.png` | `10_contour.png` | `10_hits.png` |
| 11 | `11_chord_sheet.png` | *(skipped)* | `11_chord_sheet.png` | *(skipped)* | *(skipped)* |

Use `--from-stage N` to skip stages 1 through N-1 and reuse their saved outputs.

---

## Pipeline Flow (Guitar — most complete instrument)

```
Audio File (MP3/WAV/FLAC/M4A)
        |
        v
  Stage 1: Separation
    Demucs htdemucs_6s (GPU-accelerated, 6-stem model)
    Falls back: htdemucs_ft -> htdemucs -> raw mix
    Loudness normalised to -16 LUFS before separation
    Stem confidence (0-1) propagated to all later stages
    Presence check: suppress ghost stems from residual bleed
        |
        v
  Stage 1.5: Auto Type+Role Detection  (skipped if --type and --role both set)
    Spectral flatness    -> distortion detection
    Spectral contrast    -> acoustic vs clean
    Chroma polyphony     -> lead vs rhythm
    Onset density        -> confirms role
    Outputs: guitar_type, guitar_role with confidence scores
        |
        v
  Stage 2: Pitch Extraction
    basic-pitch polyphonic ML model (CUDA-capable)
    Multi-pass: model inference once; note extraction twice at different thresholds
    Distorted type: HPSS pre-filter (harmonic/percussive separation) before model
    Per-note confidence = mean frame-level detection probability
    Confidence merge: strict-pass confirmed -> 1.00; base-only -> 0.65
        |
        v
  Stage 3: Note Cleaning
    1. Stem energy gate   — drop ghost notes from silent stem regions
    2. Duration filter    — BPM-aware minimum note length
    3. Confidence filter  — adaptive threshold from stem quality + mode
    4. Bass filter        — stricter gate below C2
    5. Bend merge         — collapse semitone-adjacent consecutive notes (lead/acoustic)
    6. Fragment merge     — merge same-pitch notes with small gaps
    7. Local pitch filter — remove outlier pitches vs local median
    8. Polyphony limit    — evict lowest-confidence / highest-pitch extras
        |
        v
  Stage 4: Tempo, Quantization & Time Signature
    librosa beat tracking on densest 60s window of note activity
    Half/double-time BPM correction (tests bpm, bpm/2, bpm*2)
    Snaps note starts to nearest 16th-note grid (35% BPM-relative tolerance)
    Snaps note durations to nearest subdivision (very short articulations exempt)
    Time signature inference via FFT onset autocorrelation:
      samples AC at 2x, 3x, 4x beat lags; small bias toward 4/4
      6/8 vs 3/4 discrimination via dotted-quarter lag ratio
    Outputs: bpm, subdivision_s, time_sig_num, time_sig_den
        |
        v
  Stage 5: Key / Scale Detection (per-section) + Capo Inference
    Splits note stream at silence gaps >= 2s (min 16 notes, min 15s per section)
    Krumhansl-Schmuckler per section; sparse sections inherit nearest key
    Dual-weighted pitch histogram: 50% duration + 50% onset count
    Searches all 24 major/minor keys + Hijaz / Kurd / Saba maqamat
    Maqam guard: requires >= 80 notes and 0.04 score gap over nearest Western key
    Capo inference: tests capo 1-5, picks position maximising open-string fit
      (how many of E/A/D/G/B fall in the key's diatonic scale)
    Outputs: global key, per-section list, capo_fret (0 = no capo), confidence
        |
        v
  Stage 5b: Chord-Tone Protection + Key-Confidence Filter
    Pre-chord pass: groups simultaneous notes into chord candidates
    Collects pitch classes in qualifying chord groups -> protected set
    Deletes note only if: off-key AND low-confidence AND NOT in protected set
    Per-section key used for in-key check (falls back to global key)
        |
        v
  Stage 6: Guitar Mapping (capo-relative)
    Effective tuning = standard tuning shifted up by capo_fret semitones
    Viterbi DP: globally optimal (string, fret) over full sequence
    Minimises: hand shift cost + string change penalty + fret height penalty
    Open-string-in-key bonus; simultaneous-note same-string penalty
    Context-aware fret window: passage median anchors position globally
    Fret numbers are capo-relative (0 = nut or capo position)
    Melody isolation (lead): top-voice -> tabs + preview; harmony -> chord detection
        |
        +---------------------------+
        |                           |
        v                           v
  Stage 7: Chord Detection     (melody or all notes)
    Groups notes within strum      |
    window (~40ms)                 |
    15 chord templates             |
    Penalty: matches - 0.3x extra  |
    Key-aware tie-breaking         |
        |                           |
        +----------+                |
        |          |                |
        v          v                v
  Stage 8      Stage 11        Stage 9 & 10
  Tab          Chord Sheet     Audio Preview (Karplus-Strong synthesis)
  Generation   (fingering       Fretboard Diagram (PNG)
  (bar lines   diagrams +
  from time    progression)
  signature)
```

---

## Per-Instrument Pipeline Notes

### Bass
- 4-string tuning: E1, A1, D2, G2
- Monophonic extraction (no Viterbi needed; simple string/fret lookup)
- Style auto-detection: slap vs fingered smooth vs fingered picked (spectral flatness + onset analysis)
- Simpler note cleaning (polyphony limit = 1 by default; no bend merge)
- 4-string ASCII tab output

### Piano
- No fretboard mapping — every MIDI pitch maps directly to one piano key
- Left/right hand split at C4 (configurable)
- Sustain-aware fragment merging (notes may extend across chord changes)
- Polyphony limit up to ~10 simultaneous notes
- Piano roll visualization: horizontal bars (time × pitch), left=blue, right=orange, confidence → opacity
- Chord recovery via simplified template matching (major, minor, dom7, maj7, min7 on piano voicings)

### Vocals
- CREPE primary extractor (monophonic harmonic F0 tracking)
- basic-pitch secondary verification pass
- Vibrato tolerance: consecutive similar-pitch detections merged into one sustained note
- Sustained note confidence boost (longer notes more reliable for CREPE)
- No mapping stage (every vocal pitch maps directly to its MIDI note)
- Output: spectrogram (CQT background) with F0 contour overlay and note segment bars

### Drums
- Fundamentally different from pitched instruments — no pitch extraction, no key detection, no mapping
- Stage 2D: onset detection + spectral classification into:
  - Kick (low centroid, peaky envelope)
  - Snare (high flatness, midrange)
  - Hihat closed/open (high centroid, flux discriminates open vs closed)
  - Tom (midrange centroid)
  - Cymbal (high-frequency energy)
- Stage 4D: tempo only (no quantization to grid; drums are event-based)
- Stage 8D: ASCII drum grid notation
  - Rows: CY (cymbal), HH (hihat), SN (snare), TM (tom), K (kick)
  - Columns: 16th-note subdivisions per bar
  - `x` = normal hit, `·` = ghost (soft), `O` = open hihat, `-` = rest, `|` = bar line

---

## Music Theory Layer (Guitar + Piano)

### Key Detection

The Krumhansl-Schmuckler (KS) algorithm correlates the song's pitch class histogram against 24 major/minor key profiles. The histogram is dual-weighted: 50% note duration + 50% onset count, which prevents long open-string sustains from dominating.

**Per-section detection** splits the note stream at silence gaps ≥ 2 seconds and runs independent key detection per section. Sections with too few notes (< 16) inherit the key from the nearest reliable neighbor. This handles songs that modulate or shift tonality between sections.

**Maqam support** extends detection to three Arabic modes (Hijaz, Kurd, Saba) with guard gates: a maqam result requires ≥ 80 notes and a 0.04 score gap over the nearest Western key.

### Capo Inference (Guitar only)

After key detection, the pipeline tests capo positions 1–5 by measuring how many of the five guitar open-string pitch classes (E, A, D, G, B) fall within the key's diatonic scale at each transposition. If a capo position scores higher than the no-capo baseline and meets the minimum threshold (4 of 5 strings), it is applied. All downstream fret numbers are capo-relative and the tab header includes `# Capo: N`. Maqam songs never receive a capo annotation.

### Chord-Tone Protection (Guitar only)

Before the key-confidence feedback filter runs, a pre-chord pass identifies all pitch classes appearing in valid chord groups (≥ 3 simultaneous notes within the strum window). These form a protected set. The filter skips deletion for any note whose pitch class is in that set, even if the note is off-key and low-confidence. This prevents removing chromatic chord tones (borrowed chords, secondary dominants, modal mixture).

### Time Signature Inference

After BPM detection, note onsets are binned at beat/16 resolution and autocorrelated via FFT. The autocorrelation is sampled at 2×, 3×, and 4× the beat period. 6/8 vs 3/4 disambiguation: if 3 beats wins and the dotted-quarter autocorrelation (1.5× beat) is ≥ 85% of the 3-beat value, the result is 6/8 instead of 3/4. The detected time signature drives bar line placement in tab output.

---

## Audio Preview (Guitar + Bass + Piano)

The synthesized preview uses **Karplus-Strong physical string modelling**:

- Half-sine pulse excitation (warm attack, no harsh crack)
- One-pole lowpass recirculation with configurable damping
- Cosine fade-out on each note's tail (eliminates click/pop at note boundaries)
- DC-blocking high-pass filter on the final mix

---

## Architecture Notes

- **Separation:** Demucs `htdemucs_6s` (6-stem model). Falls back to `htdemucs_ft` → `htdemucs` → raw mix. Input loudness-normalised to -16 LUFS. Stem presence detection guards against ghost notes from residual bleed.

- **Instrument registry:** `pipeline/registry.py` maps instrument names to pipeline runner functions. The `--instrument` flag dispatches through this registry, enabling `--instrument all` to run all five in sequence without code duplication.

- **Stage executor:** `pipeline/shared/stage_executor.py` (`StageExecutor`) encapsulates the run-or-load pattern used by every stage — checks whether to run the stage or load its cached output based on `--from-stage N`.

- **Output layout:** `pipeline/config.py` manages per-song output paths. `get_shared_dir()` returns the cross-instrument shared directory (tempo, key, quantized notes). `get_instrument_dir(instrument)` returns the per-instrument subdirectory. Migration scripts (`migrate_outputs.py`, `migrate_paths.py`) handle legacy flat-layout outputs.

- **Guitar mapping:** Viterbi DP over the full note sequence. Globally optimal fret assignments minimize total hand travel + string change cost. Capo shifts the effective tuning. Output fret numbers are capo-relative.

- **Vocal extraction:** CREPE is the primary F0 detector for vocals. basic-pitch runs as a secondary verification pass. Results are fused by confidence weight.

- **Drum classification:** Pure signal-processing (no ML model). Onset detection via librosa; classification uses spectral centroid, flatness, and flux computed in a short window around each onset.

- **Key detection:** Krumhansl-Schmuckler with dual-weighted histogram. Per-section for modulating songs. Maqam guard gates prevent false Arabic-mode classification on sparse Western songs.

- **BPM & time signature:** librosa beat tracking on the densest 60-second window; automatic half/double-time correction; FFT-based onset autocorrelation for time signature (2/4, 3/4, 4/4, 6/8).

- **Chord naming:** Template matching against 15 chord types with penalty scoring (`matches - 0.3 × extra pitch classes`) and key-aware tie-breaking.

- **Tab bar lines:** `bar_cols = (ts_num × 4 × subdivision) // ts_den` — 16 columns/bar for 4/4, 12 for 3/4 and 6/8, 8 for 2/4. Bar lines inserted inside note duration spans that cross a bar boundary.

---

## Settings

All tunable parameters live in `pipeline/settings/` — one file per pipeline stage per instrument. Nothing is hardcoded in the processing code.

### Shared settings (all instruments)

| File | Controls |
|------|----------|
| `shared/separation.py` | Demucs model cascade, silence thresholds, LUFS target, presence detection |
| `shared/quantization.py` | BPM detection window, grid snap tolerance, time signature candidates and bias |
| `shared/key_detection.py` | KS algorithm weights, maqam guard gates, segment split params, capo search range |
| `shared/audio_synthesis.py` | Karplus-Strong damping, amplitude scaling |
| `shared/visualization.py` | DPI, section lengths, row heights, color schemes |
| `shared/evaluation.py` | Chroma shift tolerance, onset matching tolerance, active frame threshold |
| `shared/presence.py` | Hard energy floors, per-instrument thresholds, cross-stem rivalry logic |

### Guitar settings

| File | Controls |
|------|----------|
| `guitar/range.py` | MIDI/Hz bounds |
| `guitar/pitch.py` | Per-mode extraction thresholds (6 modes) |
| `guitar/cleaning.py` | Confidence gates, duration filters, bend/merge gaps, polyphony limits |
| `guitar/mapping.py` | Viterbi transition costs, fret penalties, open-string bonuses |
| `guitar/chords.py` | Strum window, template penalty weights |
| `guitar/tab.py` | Column width, block length |

### Bass, Piano, Vocals, Drums settings

Each instrument has its own `range.py`, `pitch.py`, and `cleaning.py` under `settings/<instrument>/`, tuned independently for that instrument's characteristics.

---

## Tests

Unit tests live in `tests/`. Run all with:

```bash
python -m unittest discover tests
```

| File | Covers |
|------|--------|
| `test_music_theory.py` | KS key detection, scale generation, maqam detection, capo inference, per-section segmentation |
| `test_note_cleaning.py` | Duration filter, confidence gate, polyphony eviction, bend/fragment merging, chord-tone protection, segmented key filter |
| `test_guitar_mapping.py` | Viterbi DP optimality, cost minimization, open-string bonuses, same-string chord penalties |
| `test_tab_generation.py` | Column alignment, bar line placement, multi-digit fret formatting, time signature bar_cols formula |
| `test_bass_mapping.py` | 4-string fret lookup, open string positions, out-of-range handling |
| `test_piano_mapping.py` | MIDI to key index, left/right hand split, range bounds |
| `test_drums_onset.py` | Hit classification accuracy, confidence scoring, onset timing |

---

## Future Plans

- **Expanded maqam support** — Nahawand, Ajam, Bayati, and other common Arabic modes
- **Alternate guitar tuning detection** — Drop D, Open G/D, DADGAD
- **Roman numeral chord labels** — I, IV, V, vi alongside raw chord names
- **Secondary dominant detection** — V/vi, V/ii annotations in chord sheet
- **YouTube ingestion** — accept a URL instead of a local file
- **Cross-instrument translation** — remap a note sequence from one instrument to another

---

## Requirements

```
torch torchaudio demucs
basic-pitch
crepe
librosa
numpy soundfile scipy
matplotlib pygame-ce
av
```

GPU (CUDA) is used automatically if available. CPU fallback works but separation is significantly slower (~5× on a typical song).
