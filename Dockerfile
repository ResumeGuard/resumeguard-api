# ===============================
# ResumeGuard API – Fixed Dockerfile
# ===============================

# Use lightweight Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all repo files into container
COPY . .

# Install dependencies (added python-multipart)
RUN pip install --no-cache-dir fastapi uvicorn docx2txt pdfminer.six python-multipart

# Expose default FastAPI port
EXPOSE 8080

# Start the FastAPI app
CMD ["uvicorn", "auto_score_service:app", "--host", "0.0.0.0", "--port", "8080"]


