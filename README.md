# MLOps Project Group 1

End-to-end MLOps example with training, experiment tracking, and a simple model server.

## Structure

- `create_deployable_model.py` – package/prepare trained model for deployment
- `start_model_server.py` – start the prediction server
- `Dockerfile` – container image for serving
- `requirements.txt` – Python dependencies
- `src/train.py` – training script (logs to MLflow)
- `src/predict.py` – prediction utilities
- `src/Reviews.csv` – dataset (ignored by Git by default due to size)
- `src/mlruns/` – MLflow runs (ignored)
- `mlruns/` – additional MLflow runs (ignored)
- `temp_artifacts/` – temporary artifacts (ignored)

## Quickstart

1. Create and activate a virtual environment
2. Install dependencies
3. Train a model
4. Start the server

```bash
# 1) venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install
pip install -r requirements.txt

# 3) Train
python src/train.py

# 4) Serve
python start_model_server.py
```

## Notes

- Large artifacts (mlruns/, models/, temp_artifacts/, and `src/Reviews.csv`) are gitignored to keep the repo lean.
- Use Dockerfile for containerized serving if preferred.
