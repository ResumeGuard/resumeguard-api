FROM python:3.11-slim
WORKDIR /app
COPY authenticity_service_det.py .
RUN pip install --no-cache-dir fastapi uvicorn pydantic
ENV PORT=8080
CMD ["uvicorn", "authenticity_service_det:app", "--host", "0.0.0.0", "--port", "8080"]
