"""Configuration for the ML service."""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*a, **kw): pass

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(BASE_DIR / "checkpoints")))

# Data sources (CSV paths; can be under DATA_DIR)
REDBENCH_CSV = os.getenv("REDBENCH_CSV") or str(DATA_DIR / "redbench" / "redbench.csv")
NAVIROCKER_CSV = os.getenv("NAVIROCKER_CSV") or str(DATA_DIR / "navirocker" / "adversarial_prompts.csv")

# Training
DEFAULT_EPOCHS_BERT = int(os.getenv("DEFAULT_EPOCHS_BERT", "10"))
DEFAULT_EPOCHS_ENSEMBLE = int(os.getenv("DEFAULT_EPOCHS_ENSEMBLE", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256"))
BERT_MODEL = os.getenv("BERT_MODEL", "bert-base-uncased")

# GPU (DGX / CUDA): use GPU when available; set USE_GPU=0 to force CPU
def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
USE_GPU = os.getenv("USE_GPU", "1").strip().lower() not in ("0", "false", "no") and _cuda_available()

# Optional Perspective API
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY", "")

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Labels and categories (align with frontend types)
VULNERABILITY_CATEGORIES = ["jailbreaking", "bias", "toxicity", "privacy", "misinformation"]
LABEL_MAP = {"safe": 0, "vulnerable": 1}
INV_LABEL_MAP = {0: "safe", 1: "vulnerable"}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
