# FINSIGHT AI - Docker Image (Optimized for Render 512MB RAM)

FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder to global location
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY api.py .
COPY prediction.py .

# Pre-download the model during build (PyTorch)
RUN python -c "from transformers import BertForSequenceClassification, BertTokenizer; \
    BertForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
    BertTokenizer.from_pretrained('ProsusAI/finbert')"

# Expose port (Render defaults to 8000 but probes for any)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application (using single worker to save memory)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
