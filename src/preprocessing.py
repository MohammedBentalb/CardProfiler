import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.data_io import get_dataset_path, load_dataset, save_dataset


def remove_duplicates(df: pd.DataFrame):
    return df.drop_duplicates().reset_index(drop=True)

def drop_columns(df: pd.DataFrame, columns=("CUST_ID",)):
    return df.drop(columns=list(columns))

def drop_null(df: pd.DataFrame, subset=("CREDIT_LIMIT",)):
    return df.dropna(subset=list(subset)).reset_index(drop=True)

def clean_data(df: pd.DataFrame):
    df = remove_duplicates(df)
    df = drop_columns(df)
    df = drop_null(df)
    return df

def log_transform(df: pd.DataFrame):
    return np.log1p(df.select_dtypes("number"))

def skew_comparison(before: pd.DataFrame, after: pd.DataFrame):
    return pd.DataFrame({
        "skew_before": before.skew(),
        "skew_after": after.skew(),
    }).round(2)

def make_hists_after_log(df_log: pd.DataFrame):
    for col in df_log.columns:
        plt.figure(figsize=(6, 3))
        sns.histplot(df_log[col], kde=True)
        plt.title(f"log({col}) - skew = {df_log[col].skew():0.2f}")
        plt.show()

def scale_data(df: pd.DataFrame):
    scaled = StandardScaler().fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)

def prepare_clustering(df_clean: pd.DataFrame):
    return scale_data(log_transform(df_clean.copy()))

def pca_variance(df_scaled: pd.DataFrame):
    pca = PCA().fit(df_scaled)
    explained = pca.explained_variance_ratio_
    return pd.DataFrame({
        "component": [f"PC{i}" for i in range(1, len(explained) + 1)],
        "variance_explained": explained,
        "variance_cumulative": np.cumsum(explained),
    }).round(4)

def find_n_components(pca_table: pd.DataFrame, threshold=0.8):
    return int(np.argmax(pca_table["variance_cumulative"].to_numpy() >= threshold) + 1)

def plot_pca_variance(pca_table: pd.DataFrame, threshold=0.8):
    n_components = find_n_components(pca_table, threshold)
    x = np.arange(1, len(pca_table) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(x, pca_table["variance_cumulative"], marker="o")
    plt.axhline(y=threshold, linestyle=":", color="red", label=f"{threshold:.0%} threshold")
    plt.axvline(x=n_components, linestyle="--", color="gray", label=f"{n_components} components")
    plt.xticks(x)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.legend()
    plt.show()

def run_preprocessing():
    df = load_dataset("raw")
    df_clean = clean_data(df)
    save_dataset(df_clean, "clean")
    df_prepare_clustering = prepare_clustering(df_clean)
    save_dataset(df_prepare_clustering, "clustering")
    return df_clean, df_prepare_clustering


if __name__ == "__main__":
    df_clean, df_prepare_clustering = run_preprocessing()
    n_components = find_n_components(pca_variance(df_prepare_clustering))
    print(f"df_clean: {df_clean.shape} -> saved to {get_dataset_path('clean')}")
    print(f"df_prepare_clustering: {df_prepare_clustering.shape} -> saved to {get_dataset_path('clustering')}")
    print(f"PCA components for 80% variance: {n_components}")