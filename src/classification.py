import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import PLOTS_DIR
from src.mlflow import initialize_mlflow, load_champion, log_run, name_champion
from src.utils.data_io import load_dataset, save_pipeline, save_plot


features = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]
RANDOM_STATE = 42

models = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, solver="saga"),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RandomForestClassifier': RandomForestClassifier(random_state=RANDOM_STATE),
    'SVC': SVC(random_state=RANDOM_STATE)
}

models_params = {
    'LogisticRegression':  {
        "classifier__C": [0.1, 1, 10],
        "classifier__penalty": ["l1", "l2"]
        },
    'DecisionTreeClassifier': {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [2, 4, 6, 8, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
    },
    'RandomForestClassifier': {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10],
        "classifier__max_features": ["sqrt", None]
    },
    'SVC': {
        "classifier__kernel": ["rbf"],
        "classifier__C": [1, 10],
        "classifier__gamma": ["scale", 0.1]
    }
}

def build_pipline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model),
    ])

def split_data(df: pd.DataFrame, test_size = 0.2, random_state=42):
    X, Y = df[features], df["target"]
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

def run_models(x_train, y_train, x_test, y_test):
    result = []

    with mlflow.start_run(run_name="classification_run"):
        for name, model in models.items():
            with mlflow.start_run(run_name=f"Grid_{name}_model", nested=True):
                grid = GridSearchCV(
                    build_pipline(model),
                    cv = 3,
                    param_grid=models_params[name],
                    scoring={
                        "f1": "f1_macro",
                        "accuracy": "accuracy",
                        "precision": "precision_macro",
                        "recall": "recall_macro",
                    },
                    n_jobs=-1,
                    refit='f1'
                )

                start = time.perf_counter()
                grid.fit(x_train, y_train)
                end = time.perf_counter()

                i = grid.best_index_
                model = grid.best_estimator_
                metrics = {
                        'accuracy': grid.cv_results_["mean_test_accuracy"][i],
                        'precision' :grid.cv_results_["mean_test_precision"][i],
                        'recall': grid.cv_results_["mean_test_recall"][i],
                        'f1': grid.cv_results_["mean_test_f1"][i]
                    }
                
                plot = plot_confusion_matrices(f"confusion_matrix_{name}", model.predict(x_test), y_test)
                save_plot(plot, PLOTS_DIR / f"{name}_confusion_matrix")
                model_uri = log_run(name, grid.best_estimator_, grid.best_params_ , metrics, {f"{name}_confusion_matrix": plot})
                
                result.append({
                    "name": name,
                    "duration" : end - start,
                    "model": model,
                    "best_params": grid.best_params_,
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1": metrics['f1'],
                    "model_uri": model_uri 
                })
                

    return pd.DataFrame(result)

def plot_f1(result: pd.DataFrame):
    sns.barplot(data=result, x='name', y='f1')
    plt.xticks(rotation=45)
    plt.show()

def plot_confusion_matrices(plot_name: str, y_predict : list, y_test: list):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(data=confusion_matrix(y_test, y_predict), fmt="d", annot=True, yticklabels=y_test.unique(), xticklabels=y_test.unique(), ax=ax)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title(plot_name)
    return fig

def evaluate_best(result: pd.DataFrame, x_test, y_test):
    best = result.sort_values(by='f1', ascending=False).iloc[0]
    best_model = best['model']
    y_predict = best_model.predict(x_test)

    report = pd.DataFrame(classification_report(y_test, y_predict, output_dict=True))
    accuracy = accuracy_score(y_test, y_predict)
    return best_model, best['model_uri'], report, accuracy

def plot_report_metrics(report: pd.DataFrame):
    metrics = report.loc[
        ['precision', 'recall', 'f1-score'],
        ['Baseline Client', 'Card Active Client', 'Cash Active Client']
    ]

    sns.heatmap(
        metrics,
        annot=True,
        fmt=".3f",
        cmap="Blues"
    )
    return metrics

def run_classification():
    initialize_mlflow()
    data = load_dataset('labeled')
    x_train, x_test, y_train, y_test = split_data(data)
    result = run_models(x_train, y_train, x_test, y_test)
    best_model, model_uri, report, accuracy = evaluate_best(result, x_test, y_test)

    version = name_champion(model_uri)
    champion = load_champion()
    champion_accuracy = accuracy_score(y_test, champion.predict(x_test))
    print(f"champion registered as v{version.version}, reloaded accuracy: {champion_accuracy:.3f}")

    save_pipeline(best_model)
    return result, best_model, report, accuracy


if __name__ == "__main__":
    result, best_model, report, accuracy = run_classification()
    print(result.drop(columns="model").sort_values(by="f1", ascending=False).to_string(index=False))
    print(report.round(3).to_string())
    print(f"\naccuracy: {accuracy:.3f}")
