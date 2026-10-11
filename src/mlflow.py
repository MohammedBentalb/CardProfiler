
import mlflow
import mlflow.sklearn

from src.config import MLFLOW_MODEL_CHAMPION_ALIAS, MLFLOW_URI, REGISTERED_MODEL_COLLECTION_NAME


def initialize_mlflow(uri=MLFLOW_URI, expirement_name="classification-experiment"):
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(expirement_name)

def log_run(name: str, model, params: dict, metrics: dict, plots):
    mlflow.log_params(params)
    mlflow.log_params({k.replace("classifier__", "") : v  for k, v in params.items()})
    mlflow.log_metrics(metrics)
    for name, plot in plots.items():
       mlflow.log_figure(plot, f"{name}.png")
    md =  mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")
    return md.model_uri

def name_champion(model_uri: str, name: str = REGISTERED_MODEL_COLLECTION_NAME ):
   version = mlflow.register_model(model_uri=model_uri, name=name)
   mlflow.MlflowClient().set_registered_model_alias(name, MLFLOW_MODEL_CHAMPION_ALIAS, version.version)
   return version

def load_champion(name: str =REGISTERED_MODEL_COLLECTION_NAME, uri: str = MLFLOW_URI):
    mlflow.set_tracking_uri(uri)
    return mlflow.sklearn.load_model(f"models:/{name}@{MLFLOW_MODEL_CHAMPION_ALIAS}")