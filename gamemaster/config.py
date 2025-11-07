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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])
