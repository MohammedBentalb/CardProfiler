from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
REPORTS_DIR = ROOT_DIR / "Reports"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "artifacts"

PIPELINE_PATH = MODELS_DIR / "cardprofiler_pipeline.joblib"

DATASETS = {
    "raw": DATA_DIR / "Raw" / "dataset.csv",
    "clean": DATA_DIR / "Processed" / "df_clean.csv",
    "clustering": DATA_DIR / "Processed" / "df_prepare_clustering.csv",
    "clustered": DATA_DIR / "Processed" / "df_clustered.csv",
    "labeled": DATA_DIR / "Labeled" / "df_clustered.csv",
}


MLFLOW_URI = f"sqlite:///{(ROOT_DIR / "mlflow.db")}"
MLFLOW_MODEL_CHAMPION_ALIAS = "champion"
REGISTERED_MODEL_COLLECTION_NAME = "classification_champs"