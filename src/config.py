"""
Configuration constants for the topic modeling pipeline.
"""
import os
from pathlib import Path
import torch

# Paths
REPO_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_ROOT / "data"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SRC_DIR = REPO_ROOT / "src"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# Discord
DISCORD_CHANNEL_ID = "1481309299761348714"

# Model
LABSE_MODEL = "sentence-transformers/LaBSE"

# Device selection: CUDA if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset paths
UC_DATASET_PATH = DATA_DIR / "uc_dataset_20krows1.csv"
AUGMENTED_DATASET_PATH = DATA_DIR / "feedback_augmented_v1.json"
UC_FILTERED_DATASET_PATH = DATA_DIR / "uc_dataset_filtered.json"  # sentiment-gated real data

# Embedding cache
EMBEDDINGS_CACHE_DIR = DATA_DIR
