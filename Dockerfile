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

# Copy training data and scripts
COPY --chown=user financial_news_augmented.csv .
COPY --chown=user create_dataset.py .
COPY --chown=user fine_tune.py .
COPY --chown=user prediction.py .
COPY --chown=user api.py .
COPY --chown=user verify_model.py .

# --- BUILD-TIME TRAINING (The "Smart" Docker Build) ---
# Instead of downloading the generic model, we RETRAIN it right here in the builder.
# This ensures the deployed app has the 150+ sample "Smart Brain" without Git LFS.
RUN python fine_tune.py

# Expose port 7860 (Standard)
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "prediction.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
