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




