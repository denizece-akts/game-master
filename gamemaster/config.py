import os
import json
from pathlib import Path
import random

import numpy as np
import torch


def load_config(path=None):
    if path is None:
        path = os.environ.get("GAMEMASTER_CONFIG", "config.json")
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()
HF_TOKEN = CONFIG.get("hf_token") or os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = CONFIG.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
WANDB_API_KEY = CONFIG.get("wandb_api_key") or os.environ.get("WANDB_API_KEY", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Weights & Biases configuration
WANDB_PROJECT = CONFIG["wandb_project"]
WANDB_ENTITY = CONFIG.get("wandb_entity")  # Can be None
WANDB_RUN_NAME = CONFIG.get("wandb_run_name")  # Can be None for auto-generated name

# Configurable golden set path
GOLDEN_SET_FILENAME = CONFIG["golden_set_filename"]
GOLDEN_SET_PATH = OUTPUT_DIR / GOLDEN_SET_FILENAME


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])

