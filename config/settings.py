import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
from pathlib import Path
from loguru import logger
from typing import Dict

load_dotenv()
# paths

BASE_DIR= Path(__file__).resolve().parent.parent
DATA_DIR=BASE_DIR / 'data'
RAW_DATA_DIR= DATA_DIR / 'raw'
PROCESSED_DATA_DIR= DATA_DIR / 'processed'
MODELS_DIR= BASE_DIR / 'models'
TF_IDF_DIR=MODELS_DIR/'TF-IDF'
DATABASE_DIR=BASE_DIR /'database'
SRC_DIR=BASE_DIR /'src'
AI_ADVISOR_DIR= SRC_DIR/'ai_advisor'
CLASSIFIER_DIR=SRC_DIR/'classifier'
SCRIPTS_DIR=BASE_DIR/'scripts'
LOGS_DIR=BASE_DIR/'logs'
for dir in [DATA_DIR,DATA_DIR,RAW_DATA_DIR,PROCESSED_DATA_DIR,MODELS_DIR,DATABASE_DIR,SRC_DIR,AI_ADVISOR_DIR,CLASSIFIER_DIR,TF_IDF_DIR,SCRIPTS_DIR,LOGS_DIR]:
    dir.mkdir(parents=True,exist_ok=True)

# logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

LOG_FILE = LOGS_DIR / 'app.log'
ERROR_LOG_FILE = LOGS_DIR / 'error.log'

LOG_ROTATION = os.getenv('LOG_ROTATION', '10 MB')
LOG_RETENTION = os.getenv('LOG_RETENTION', '30 days')

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# database
DB_NAME=os.getenv("DB_NAME","customer_loan")
if not DB_NAME:
    logger.warning("DB_NAME is not set. Using default value 'customer_loan'.")
    DB_NAME = "customer_loan"
DB_USER=os.getenv("DB_USER")
DB_PASSWORD=os.getenv("DB_PASSWORD")
DB_HOST=os.getenv("DB_HOST","localhost")
DB_PORT=os.getenv("DB_PORT","5432")
DB_TYPE=os.getenv("DB_TYPE","postgres")

DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '20'))
DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', '30'))
DB_ECHO = os.getenv('DB_ECHO', 'False').lower() == 'true'
APP_ENV="testing"
DB_URL=os.getenv("DATABASE_URL",f"{DB_TYPE}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# classifier
AVAILABLE_CLASSIFIERS=[ 'logistic_regression','naive_bayes','svm','random_forest','gradient_boosting','xgboost','lightgbm','catboost']

MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0.0')
DEFAULT_CLASSIFIER = os.getenv('DEFAULT_CLASSIFIERS', 'random_forest')

TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
CV_FOLDS = int(os.getenv('CV_FOLDS', '5'))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.5'))

# written by scripts/train_model.py, read by the app at inference time
BEST_MODEL_PATH = MODELS_DIR / 'best_model.joblib'

MODEL_SELECTION_METRIC = os.getenv('MODEL_SELECTION_METRIC', 'roc_auc')

SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.85'))
HASH_ALGORITHM = os.getenv('HASH_ALGORITHM', 'sha256')

# RAG
RAG_CONFIG = {
    "enabled": True,
    "embedding_model": "text-embedding-3-large",
    "vector_store": "chromadb",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "top_k": 5,
}

# HuggingFace advisor
HF_TOKEN=os.getenv("HF_API_TOKEN")
HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"


OUTPUT_RULES = {
    "allow_probabilities": True,
    "confidence_threshold": 0.65,
    "disallowed_phrases": [
        "guaranteed",
        "100% sure",
        "no risk"
    ],
    "explanation_style": "business_friendly",  # technical | exec | business
}

DECISION_POLICY = {
    "high_risk_threshold": 0.8,
    "medium_risk_threshold": 0.5,
    "actions": {
        "high": "immediate_retention_offer",
        "medium": "engagement_campaign",
        "low": "monitor_only"
    }
}

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
}

def get_model_config() -> Dict:
    """Get model configuration as dictionary."""
    return {
        'classifier': DEFAULT_CLASSIFIER,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'cv_folds': CV_FOLDS,
        'model_version': MODEL_VERSION,
        'min_confidence': MIN_CONFIDENCE_THRESHOLD
    }


__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'BEST_MODEL_PATH',
    'MODEL_SELECTION_METRIC',
    'get_model_config',
]
from loguru import logger

logger.remove()

logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    format=LOG_FORMAT
)

logger.success("Logging configured successfully. Logs will be saved to: {}", LOG_FILE)