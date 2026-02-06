# FINSIGHT AI - Docker Image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .
COPY prediction.py .

# Pre-download the FinBERT model
RUN python -c "from transformers import TFBertForSequenceClassification, BertTokenizer; \
    TFBertForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
    BertTokenizer.from_pretrained('ProsusAI/finbert')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
