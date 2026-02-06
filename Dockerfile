# FINSIGHT AI - Hugging Face Spaces Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=user api.py .
COPY --chown=user prediction.py .

# Pre-download FinBERT model
RUN python -c "from transformers import TFBertForSequenceClassification, BertTokenizer; \
    TFBertForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
    BertTokenizer.from_pretrained('ProsusAI/finbert')"

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
