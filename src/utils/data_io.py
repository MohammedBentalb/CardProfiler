from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from src.config import DATASETS, PIPELINE_PATH

READ_ONLY_DATASETS = {"raw"}


def load_data(path):
    return pd.read_csv(path)

def save_data(df: pd.DataFrame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def get_dataset_path(name):
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}', expected one of {list(DATASETS)}")
    return DATASETS[name]

def load_dataset(name):
    return load_data(get_dataset_path(name))

def save_dataset(df: pd.DataFrame, name):
    if name in READ_ONLY_DATASETS:
        raise ValueError(f"Dataset '{name}' is read-only")
    save_data(df, get_dataset_path(name))

def save_pipeline(pipeline, path=PIPELINE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path

def load_pipeline(path=PIPELINE_PATH):
    return joblib.load(path)

def save_plot(fig, path):
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close()
    return path