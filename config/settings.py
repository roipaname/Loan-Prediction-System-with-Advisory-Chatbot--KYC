import os
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger
from typing import Dict

load_dotenv()
#==================================
# Project root directory and Paths
#==================================

BASE_DIR= Path(__file__).resolve().parent.parent
DATA_DIR=BASE_DIR / 'data'
RAW_DATA_DIR= DATA_DIR / 'raw'
PROCESSED_DATA_DIR= DATA_DIR / 'processed'
MODELS_DIR= BASE_DIR / 'models'
DATABASE_DIR=BASE_DIR /'database'
SRC_DIR=BASE_DIR /'src'
AI_ADVISOR_DIR= SRC_DIR/'ai_advisor'
CLASSIFIER_DIR=SRC_DIR/'classifier'
TF_IDF_DIR=SRC_DIR/'tf_idf'
SCRIPTS_DIR=BASE_DIR/'scripts'
LOGS_DIR=BASE_DIR/'logs'
for dir in [DATA_DIR,DATA_DIR,RAW_DATA_DIR,PROCESSED_DATA_DIR,MODELS_DIR,DATABASE_DIR,SRC_DIR,AI_ADVISOR_DIR,CLASSIFIER_DIR,TF_IDF_DIR,SCRIPTS_DIR,LOGS_DIR]:
    dir.mkdir(parents=True,exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log level
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# Log file paths
LOG_FILE = LOGS_DIR / 'app.log'
ERROR_LOG_FILE = LOGS_DIR / 'error.log'

# Log rotation
LOG_ROTATION = os.getenv('LOG_ROTATION', '10 MB')
LOG_RETENTION = os.getenv('LOG_RETENTION', '30 days')

# Log format
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)







from loguru import logger

logger.remove()
logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    format=LOG_FORMAT
)
