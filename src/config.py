from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
REPORTS_DIR = ROOT_DIR / "Reports"

DATASETS = {
    "raw": DATA_DIR / "Raw" / "dataset.csv",
    "clean": DATA_DIR / "Processed" / "df_clean.csv",
    "clustering": DATA_DIR / "Processed" / "df_prepare_clustering.csv",
}
