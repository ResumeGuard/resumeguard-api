# ===============================
# ResumeGuard API – Fixed Dockerfile
# ===============================

FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files (ensures both .py files are included)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    pydantic==2.6.3

# Set environment variable for port (Railway expects this)
ENV PORT=8080

# Expose the same port
EXPOSE 8080

# Run FastAPI app with uvicorn
CMD ["uvicorn", "authenticity_service_det:app", "--host", "0.0.0.0", "--port", "8080"]
