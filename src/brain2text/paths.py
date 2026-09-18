"""Project-relative paths shared by training and preprocessing scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_project_dirs() -> None:
    """Create generated-data directories when a pipeline is run locally."""

    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)