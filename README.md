# CASA/NMF Benchmark

A research-oriented benchmark of non-negative matrix factorisation configurations for sound-source separation and computational auditory scene analysis.

The experiment evaluates 288 configurations across polyphonic choir, rainforest, and reverberant-speech textures using SDR, SAR, and MCI-related measures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Results are written to `results/`. The original audio recordings are not included. Prepare compatible input audio separately and check the dataset licences before redistribution.

## Repository layout

- `main.py` - experiment runner
- `audio_data_preparation.py` - input preparation
- `generate_spectrograms.py` - visualisation helpers
- `quicktest.py` - small validation run
- `results/` - selected tabular and summary results

## Status

This is an academic benchmark and exploratory research codebase. Reproducibility may require adjusting input paths and configuration for the local dataset.

## Attribution

Created by Emil Zawistowski for coursework in Sound and Music Computing at Aalborg University.
