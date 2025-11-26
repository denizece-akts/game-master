import os
import json
from pathlib import Path
import random

import numpy as np
import torch


def load_config(path=None):
    if path is None:
        path = Path(__file__).parent.parent.parent / "config.json"
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat_config[k] = v
        else:
            flat_config[section] = values
    
    config = flat_config

    project_root = Path(__file__).parent.parent.parent.resolve()
    
    for key in ["desc_csv_path_full", "reviews_csv_path_full", "output_dir", "embedding_local_dir", "llm_local_dir", "qa_set_filename"]:
        if key in config:
            p = Path(config[key])
            if not p.is_absolute():
                config[key] = str(project_root / p)
                
    return config


CONFIG = load_config()
HF_TOKEN = CONFIG.get("hf_token") or os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = CONFIG.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
WANDB_API_KEY = CONFIG.get("wandb_api_key") or os.environ.get("WANDB_API_KEY", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PROJECT = CONFIG["wandb_project"]
WANDB_ENTITY = CONFIG.get("wandb_entity")
WANDB_RUN_NAME = CONFIG.get("wandb_run_name")

QA_SET_FILENAME = CONFIG["qa_set_filename"]
QA_SET_PATH = OUTPUT_DIR / QA_SET_FILENAME
if not QA_SET_PATH.exists():
     resource_path = Path(__file__).parent.parent / "resources" / QA_SET_FILENAME
     if resource_path.exists():
         QA_SET_PATH = resource_path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])

