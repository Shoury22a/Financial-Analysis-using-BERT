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
# Install CPU-only PyTorch to save 2GB+ space and speed up build
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=user api.py .
COPY --chown=user prediction.py .

# Pre-download FinBERT model (PyTorch version)
RUN python -c "from transformers import BertForSequenceClassification, BertTokenizer; \
    BertForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
    BertTokenizer.from_pretrained('ProsusAI/finbert')"

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run Streamlit (full UI dashboard)
CMD ["streamlit", "run", "prediction.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
