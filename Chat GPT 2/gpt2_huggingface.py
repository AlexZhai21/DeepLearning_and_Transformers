import torch
from transformers import GPT2LMHeadModel

from models import GPT, GPTConfig


def from_pretrained(model_type):
    """Load pretrained GPT-2 model weights from Hugging Face."""
    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

    print("loading weights from pretrained gpt: %s" % model_type)

    config_args = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_type]

    config_args["vocab_size"] = 50257
    config_args["block_size"] = 1024

    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model
