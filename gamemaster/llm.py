import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import CONFIG, HF_TOKEN, DEVICE


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

    print("Loading LLM (from HF cache):", CONFIG["llm_model"])
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["llm_model"], use_fast=True, token=HF_TOKEN
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["llm_model"],
        trust_remote_code=False,
        token=HF_TOKEN,
        **load_kwargs,
    ).eval()
    try:
        model = model.to(DEVICE)
        model.generation_config.use_cache = True
    except Exception as e:
        print("Note: could not enable use_cache tweak:", e)
    return tokenizer, model
