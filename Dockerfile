FROM python:3.9

# Minimal, production-friendly defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose MLflow serving port (configurable via PORT env)
EXPOSE 5001
ENV PORT=5001

# Default model URI (override at runtime with -e MLFLOW_MODEL_URI=...)
# Tip: you can also mount a model volume and point MLFLOW_MODEL_URI there.
ENV MLFLOW_MODEL_URI=/app/src/mlruns/339841543153088714/models/m-0bba6e509f1943909e822e05090a3a8f

# Run MLflow model server using the environment inside the image
# --env-manager local avoids creating a new conda/venv at runtime
CMD ["sh", "-lc", "mlflow models serve -m \"$MLFLOW_MODEL_URI\" --env-manager local -p ${PORT} --host 0.0.0.0"]
