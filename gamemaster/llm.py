import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from .config import CONFIG, HF_TOKEN, DEVICE


def _ensure_local_model():
    remote_id = CONFIG["llm_model"]
    local_dir = Path(CONFIG.get("llm_local_dir", "./llm_model"))

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Using existing local model dir: {local_dir}")
        return str(local_dir)

    print(f"Local model dir {local_dir} missing/empty, downloading from {remote_id}...")
    local_dir.mkdir(parents=True, exist_ok=True)

    token = HF_TOKEN or os.environ.get("HF_TOKEN", None)
    if not token:
        raise RuntimeError(
            "No HF token found. Set 'hf_token' in config.json or export HF_TOKEN in the environment."
        )

    snapshot_download(
        repo_id=remote_id,
        token=token,
        local_dir=str(local_dir),
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tokenizer.*",
            "*.model",
            "*.txt",
            "*.py",
        ],
        ignore_patterns=["original/*"],
    )

    print("✅ Download complete.")
    return str(local_dir)


def load_llm():
    load_kwargs = {"device_map": "auto"}
    if CONFIG["use_4bit"]:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["quantization_config"] = bnb_config
    elif CONFIG["use_8bit"]:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model_path = _ensure_local_model()

    print("Loading LLM from:", model_path)
    token = HF_TOKEN or os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=token,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=False,
        token=token,
        **load_kwargs,
    ).eval()

    try:
        model = model.to(DEVICE)
        model.generation_config.use_cache = True
    except Exception as e:
        print("Note: could not enable use_cache tweak:", e)

    return tokenizer, model
