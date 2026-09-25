# Brain2Text

Research code for translating neural activity into phoneme sequences and language.
The project combines signal augmentation, phoneme sequence denoising, CTC decoding,
and an LLM-based second stage for converting decoded phonemes into language.

## Project layout

```text
src/brain2text/        Reusable package utilities and project paths
scripts/                Dataset prep, corpus cleaning, augmentation, and analysis utilities
data/raw/               Local neural recordings and source corpora (not committed)
data/processed/         Generated JSONL datasets and language-model inputs
artifacts/checkpoints/  Local model checkpoints (not committed)
notebooks/              Exploratory training and model-selection notebooks
```

Project data-prep scripts live under `scripts/` so the repository root stays
focused on the package and generated artifacts. Keep generated output under
`data/processed/` or `artifacts/`.

## Pipeline

1. Place source recordings or annotated corpus files in `data/raw/`.
2. Clean corpus text with `scripts/process_corpus.py`.
3. Convert cleaned sentences to phonemes with `scripts/generate_phonemes.py`.
4. Train or evaluate the phoneme denoiser using `pho_refinement_tools.py` and
   `pho_refinement.py`.
5. Decode neural outputs into phoneme sequences.
6. Pass the phoneme sequence to the selected LLM language-generation stage.
7. Write confusion-matrix reports to `reports/`.

The committed source includes the phoneme vocabulary and processing logic. Large
neural recordings, trained weights, and language-model binaries are intentionally
excluded from version control; see `data_link.txt` for the source-data pointer.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
```
