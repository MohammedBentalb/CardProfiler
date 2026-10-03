import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import REPORTS_DIR


def profile_report(df: pd.DataFrame, file_name="profile_report.html", title="Profiling Report"):
    from data_profiling import ProfileReport
    report = ProfileReport(df, title=title)
    path = REPORTS_DIR / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_file(path)
    return path


def overview(df: pd.DataFrame):
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.notna().sum(),
    })


def describe_data(df: pd.DataFrame):
    return df.describe().T.round(2)


def duplicates_summary(df: pd.DataFrame, id_col="CUST_ID"):
    summary = {"duplicated_rows": int(df.duplicated().sum())}
    if id_col in df.columns:
        summary["duplicated_ids"] = int(df[id_col].duplicated().sum())
    return summary


def missing_table(df: pd.DataFrame):
    return pd.DataFrame({
        "number": df.isna().sum(),
        "percent": ((df.isna().sum() / len(df)) * 100).round(2),
    }).sort_values("number", ascending=False)


def get_skewness(df: pd.DataFrame):
    return df.select_dtypes("number").skew().sort_values(ascending=False).round(2)


def make_hists(df: pd.DataFrame):
    for col in df.select_dtypes("number").columns:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} - skew = {df[col].skew():0.2f}")
        plt.show()


def get_corr(df: pd.DataFrame):
    return df.select_dtypes("number").corr()


def plot_corr_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(8, 7))
    sns.heatmap(get_corr(df), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.show()


def get_outliers(df: pd.DataFrame, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_value = q1 - 1.5 * iqr
    max_value = q3 + 1.5 * iqr

    mask = (df[col] < lower_value) | (df[col] > max_value)
    return df[mask], mask


def outliers_summary(df: pd.DataFrame):
    rows = []
    for col in df.select_dtypes("number").columns:
        _, mask = get_outliers(df, col)
        rows.append({
            "column": col,
            "outliers": int(mask.sum()),
            "percent": round(mask.mean() * 100, 2),
        })
    return pd.DataFrame(rows).set_index("column").sort_values("outliers", ascending=False)


def make_boxplots(df: pd.DataFrame):
    for col in df.select_dtypes("number").columns:
        plt.figure(figsize=(6, 2))
        sns.boxplot(x=df[col])
        plt.title(f"Box plot for {col}")
        plt.show()
