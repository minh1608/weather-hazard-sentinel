
# ===== Dockerfile (for Cloud Run deployment) =====
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files
COPY agents.py main.py ./ 

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    google-generativeai \
    google-cloud-storage \
    requests \
    pandas

# Environment variables (Cloud Run can override)
ENV PORT=8080
EXPOSE 8080

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
