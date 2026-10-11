import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.utils.data_io import get_dataset_path, load_dataset, save_dataset


def run_kmeans(df: pd.DataFrame, components_range=range(2, 6), k_range=range(2, 8), random_state=42):
    results = []
    for n in components_range:
        x_pca = PCA(n_components=n, random_state=random_state).fit_transform(df)
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(x_pca)
            results.append({
                "n_components": n,
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(x_pca, labels),
            })
    results = pd.DataFrame(results)

    best = results.loc[results["silhouette"].idxmax()]
    best_n = int(best["n_components"])
    best_k = int(best["k"])

    pca = PCA(n_components=best_n, random_state=random_state)
    X = pd.DataFrame(pca.fit_transform(df), columns=[f"PC{i}" for i in range(1, best_n + 1)])
    model = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)

    return {
        "model": model,
        "pca": pca,
        "labels": labels,
        "X": X,
        "params": {"n_components": best_n, "k": best_k},
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X, labels),
        "results": results,
    }

def plot_kmeans_search(results: pd.DataFrame):
    sns.lineplot(data=results, x="k", y="inertia", hue="n_components", marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Different PCA Components")
    plt.show()

    sns.lineplot(data=results, x="k", y="silhouette", hue="n_components", marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette")
    plt.title("Silhouette Method for Different PCA Components")
    plt.show()

    pivot = results.pivot(index="n_components", columns="k", values="silhouette")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Silhouette by PCA components x K")
    plt.xlabel("K")
    plt.ylabel("PCA components")
    plt.show()

def plot_clusters(kmeans: dict):
    data = kmeans["X"].assign(clusters=kmeans["labels"])
    centroids = pd.DataFrame(kmeans["model"].cluster_centers_)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=data, x="PC1", y="PC2", hue="clusters", palette="Set2")
    plt.scatter(centroids[0], centroids[1], c="black", marker="X", s=250, label="centroids")
    plt.legend()
    plt.title("Data distribution")
    plt.xlabel("First dimension PC1")
    plt.ylabel("Second dimension PC2")
    plt.show()

def run_dbscan(df: pd.DataFrame, components_range=range(2, 6), eps_values=(0.2, 0.3, 0.5, 0.7, 1.0), min_samples_values=(5, 10, 20), max_noise=5, max_cluster_size=90, random_state=42):
    results = []
    for n in components_range:
        x_pca = PCA(n_components=n, random_state=random_state).fit_transform(df)
        for eps in eps_values:
            for ms in min_samples_values:
                labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(x_pca)
                noise = labels == -1
                clusters = len(set(labels)) - (1 if noise.any() else 0)
                sil, max_size, min_size = np.nan, np.nan, np.nan

                if clusters >= 2:
                    size = pd.Series(labels[~noise]).value_counts(normalize=True)
                    max_size, min_size = size.max() * 100, size.min() * 100
                    sil = silhouette_score(x_pca[~noise], labels[~noise])

                results.append({
                    "clusters": clusters,
                    "eps": eps,
                    "n_components": n,
                    "min_samples": ms,
                    "noise": noise.mean() * 100,
                    "silhouette": sil,
                    "max": max_size,
                    "min": min_size,
                })
    results = pd.DataFrame(results).sort_values("silhouette", ascending=False).reset_index(drop=True)

    candidates = results[(results["noise"] < max_noise) & (results["max"] < max_cluster_size)]    
    if candidates.empty:
        raise ValueError(f"ma kaynch {max_noise} {max_cluster_size}")
    
    best = candidates.sort_values(by=["silhouette", "n_components", "noise"], ascending=[False, True, True]).iloc[0]
    best_n = int(best["n_components"])
    best_eps = float(best["eps"])
    best_ms = int(best["min_samples"])

    #a7ssen params b les condition  2 pca 3 clusters
    pca = PCA(n_components=best_n, random_state=random_state)
    X = pd.DataFrame(pca.fit_transform(df), columns=[f"PC{i}" for i in range(1, best_n + 1)])
    model = DBSCAN(eps=best_eps, min_samples=best_ms)
    labels = model.fit_predict(X)
    noise = labels == -1

    return {
        "model": model,
        "pca": pca,
        "labels": labels,
        "X": X,
        "params": {"n_components": best_n, "eps": best_eps, "min_samples": best_ms},
        "noise": noise.mean() * 100,
        "silhouette": silhouette_score(X[~noise], labels[~noise]),
        "results": results,
    }

def plot_dbscan_search(results: pd.DataFrame):
    sns.lineplot(data=results, x="clusters", y="silhouette", hue="n_components", marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette")
    plt.title("DBSCAN silhouette for different PCA components")
    plt.show()
    
    pivot = results.pivot_table(index="n_components", columns="eps", values="silhouette", aggfunc="max")
    sns.heatmap(pivot, annot=True)
    

def plot_dbscan_clusters(dbscan: dict):
    data = dbscan["X"].assign(clusters=dbscan["labels"])
    # noise = data[data["clusters"] == -1]
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=data[data["clusters"] != -1], x="PC1", y="PC2", hue="clusters", palette="Set2")
    # plt.scatter(noise["PC1"], noise["PC2"], c="black", marker="x", s=15, label="noise (-1)")
    plt.legend()
    plt.title("DBSCAN clusters")
    plt.xlabel("First dimension PC1")
    plt.ylabel("Second dimension PC2")
    plt.show()

def evaluate(X: pd.DataFrame, labels, name):
    X, labels = np.asarray(X), np.asarray(labels)
    mask = labels != -1
    sizes = pd.Series(labels[mask]).value_counts(normalize=True) * 100
    return {
        "method": name,
        "clusters": len(sizes),
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
        "noise_%": (~mask).mean() * 100,
        "min_cluster_%": sizes.min(),
        "max_cluster_%": sizes.max(),
    }

def select_best(kmeans: dict, dbscan: dict, margin=0.05, max_noise=5, min_size=10, max_size=90):
    km_p, db_p = kmeans["params"], dbscan["params"]
    comparison = pd.DataFrame([
        evaluate(kmeans["X"], kmeans["labels"], f"K-Means (PCA={km_p['n_components']}, K={km_p['k']})"),
        evaluate(dbscan["X"], dbscan["labels"], f"DBSCAN (PCA={db_p['n_components']}, eps={db_p['eps']}, ms={db_p['min_samples']})"),
    ])

    km, db = comparison.iloc[0], comparison.iloc[1]
    dbscan_wins = (
        db["silhouette"] > km["silhouette"] + margin
        and db["noise_%"] < max_noise
        and db["min_cluster_%"] >= min_size
        and db["max_cluster_%"] <= max_size
    )
    method = "dbscan" if dbscan_wins else "kmeans"
    return method, comparison

def attach_labels(df_clean: pd.DataFrame, kmeans: dict, dbscan: dict, method):
    df = df_clean.copy()
    df["cluster_kmeans"] = kmeans["labels"]
    df["cluster_dbscan"] = dbscan["labels"] 
    df["est_atypique_dbscan"] = dbscan["labels"] == -1
    df["cluster_final"] = df[f"cluster_{method}"]

    if df["cluster_final"].isna().any():
        raise ValueError("cluster_final has missing values")
    return df

def run_clustering():
    df_clean = load_dataset("clean")
    df_prepare_clustering = load_dataset("clustering")

    kmeans = run_kmeans(df_prepare_clustering)
    dbscan = run_dbscan(df_prepare_clustering)

    method, comparison = select_best(kmeans, dbscan)
    df_clustered = attach_labels(df_clean, kmeans, dbscan, method)
    save_dataset(df_clustered, "clustered")
    return df_clustered, method, comparison


if __name__ == "__main__":
    df_clustered, method, comparison = run_clustering()
    print(comparison.round(3).to_string(index=False))
    print(f"\nSelected method: {method}")
    print(f"df_clustered: {df_clustered.shape} -> saved to {get_dataset_path('clustered')}")
