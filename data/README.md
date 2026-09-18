# Data

Keep local or restricted datasets here, separated by lifecycle:

- `raw/`: source recordings, annotations, and corpus exports.
- `processed/`: generated JSONL datasets, lexicons, and decoder inputs.

Large neural recordings and language-model binaries are ignored by Git. Store the
source link, release/version, and preprocessing command in `data_link.txt` or in an
experiment note. Do not commit participant-identifying metadata.
