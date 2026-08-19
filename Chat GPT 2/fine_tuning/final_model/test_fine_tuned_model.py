import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from gpt2_huggingface import from_pretrained
from tokenizer import get_gpt2_tokenizer


checkpoint_path = Path(__file__).resolve().parent / "gpt2_sft_final.pt"
max_new_tokens = 200
temperature = 0.7
top_k = 40

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(checkpoint_path, map_location=device)
model = from_pretrained("gpt2")
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)

enc = get_gpt2_tokenizer()


def generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k):
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    while x.size(1) < len(tokens) + max_new_tokens:
        with torch.no_grad():
            idx_cond = x[:, -model.config.block_size :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

        generated_text = enc.decode(x[0, len(tokens) :].tolist())
        if "<|endoftext|>" in generated_text:
            break

    generated_text = enc.decode(x[0, len(tokens) :].tolist())
    generated_text = generated_text.split("<|endoftext|>", 1)[0]
    return generated_text.strip()


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print(f"Loaded fine-tuned checkpoint from {checkpoint_path}")
print(f"Chatting with your fine-tuned GPT-2 model on {device}. Type 'quit' or 'exit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"quit", "exit"}:
        break
    if not user_input:
        continue

    prompt = f"User: {user_input}\nAssistant:"
    reply = generate(prompt)
    print(f"GPT: {reply}\n")
