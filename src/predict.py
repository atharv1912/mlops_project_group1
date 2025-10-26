from mlflow.tracking import MlflowClient

client = MlflowClient()
run_id = mlflow.last_active_run().info.run_id
model_uri = f"runs:/{run_id}/model"

client.create_registered_model("HeartDiseaseModel")
model_version = client.create_model_version("HeartDiseaseModel", model_uri, run_id)

print(f"Model version {model_version.version} registered successfully.")
