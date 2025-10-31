# ===============================
# ResumeGuard API – Fixed Dockerfile (with scikit-learn)
# ===============================

# Use lightweight Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first so Docker can cache efficiently
COPY requirements.txt /app/requirements.txt

# Force clean pip install of all dependencies (including scikit-learn)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the repo
COPY . /app

# Expose default FastAPI port
EXPOSE 8080

# Start the FastAPI app
CMD ["uvicorn", "auto_score_service:app", "--host", "0.0.0.0", "--port", "8080"]
