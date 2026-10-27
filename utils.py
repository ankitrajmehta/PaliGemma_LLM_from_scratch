from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    missing, unexpected = model.load_state_dict(tensors, strict=False)

    # The same matrix serves as both the input embedding table and the output vocabulary projection in Gemma, so the checkpoint stores it only once as embed_tokens.weight
    # lm_head is tied to the embed_tokens (through tie_weights() function),
    missing = [k for k in missing if not k.startswith("language_model.lm_head")]
    if missing or unexpected:
        raise RuntimeError(
            f"""Checkpoint does not match the model architecture.\n
            Missing from checkpoint (would stay randomly initialized): {missing}\n
            Unexpected in checkpoint (silently dropped): {unexpected}"""
        )

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)