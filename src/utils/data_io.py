from pathlib import Path
import pandas as pd

from src.config import DATASETS

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
