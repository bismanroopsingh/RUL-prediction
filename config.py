from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_PATH = RAW_DATA / "train_FD001.txt"
TEST_PATH = RAW_DATA / "test_FD001.txt"
RUL_PATH = RAW_DATA / "RUL_FD001.txt"

# ==========================================================
# Randomness
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# Dataset
# ==========================================================

RUL_CAP = 125

TEST_SIZE = 0.20

# ==========================================================
# Feature Engineering
# ==========================================================

ROLLING_WINDOW = 5

LAG_STEPS = 3

EMA_SPAN = 5

# ==========================================================
# Feature Selection
# ==========================================================

TOP_FEATURES = 60

CORRELATION_THRESHOLD = 0.95

VARIANCE_THRESHOLD = 0.0

# ==========================================================
# Optuna
# ==========================================================

N_TRIALS = 50

N_SPLITS = 5

# ==========================================================
# XGBoost Default
# ==========================================================

XGB_DEFAULT = {

    "objective": "reg:squarederror",

    "random_state": RANDOM_STATE,

    "n_jobs": -1,

}

# ==========================================================
# Output Files
# ==========================================================

MODEL_PATH = MODEL_DIR / "xgboost.pkl"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

FEATURE_PATH = MODEL_DIR / "selected_features.pkl"

BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"