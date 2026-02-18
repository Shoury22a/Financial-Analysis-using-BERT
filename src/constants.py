"""
Centralized Constants for FINSIGHT AI
All non-configurable constants in one place
"""
import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== MODEL CONSTANTS ====================
DEFAULT_MODEL_NAME = "ProsusAI/finbert"
MAX_SEQUENCE_LENGTH = 128
NUM_LABELS = 3

# Label mapping is now loaded dynamically from trained model's config.json
# This ensures consistency between training and inference
# NO HARDCODED LABEL MAPPINGS HERE!

# ==================== DATA PROCESSING ====================
RANDOM_SEED = 42
MIN_TEXT_LENGTH = 5  # Minimum words in text
MAX_TEXT_LENGTH = 512  # Maximum tokens

# ==================== TRAINING ====================
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 10
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01

# ==================== API CONSTANTS ====================
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
API_VERSION = "1.0.0"
API_TITLE = "FINSIGHT AI API"
API_DESCRIPTION = "AI-Powered Financial Sentiment Analysis & Stock Intelligence"

# ==================== VALIDATION ====================
# Valid sentiment labels (order matters for model training)
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# Financial datasets available for loading
AVAILABLE_DATASETS = [
    "financial_phrasebank",
    "zeroshot/twitter-financial-news-sentiment",
]

# ==================== LOGGING ====================
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
JSON_LOG_FORMAT = {
    "timestamp": "asctime",
    "name": "name",
    "level": "levelname",
    "message": "message"
}

# ==================== UI CONSTANTS ====================
# Chart configuration
CANDLESTICK_UP_COLOR = "#00d4aa"
CANDLESTICK_DOWN_COLOR = "#ff6b6b"
SMA20_COLOR = "#ffd93d"
SMA50_COLOR = "#00a8e8"

# ==================== VALIDATION THRESHOLDS ====================
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MIN_CONFIDENCE_THRESHOLD = 0.5

# ==================== FILE EXTENSIONS ====================
ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp']
ALLOWED_DATA_EXTENSIONS = ['.csv', '.json', '.parquet']
