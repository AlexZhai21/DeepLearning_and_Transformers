import torch
from torch.nn import functional as F

from gpt2_huggingface import from_pretrained
from tokenizer import get_gpt2_tokenizer


max_new_tokens = 200
temperature = 0.7
top_k = 40

model = from_pretrained("gpt2")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

enc = get_gpt2_tokenizer()


def generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k):
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    stop_strings = []

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
        if any(stop in generated_text for stop in stop_strings):
            break

    generated_text = enc.decode(x[0, len(tokens) :].tolist())
    for stop in stop_strings:
        generated_text = generated_text.split(stop, 1)[0]
    return generated_text.strip()


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print(f"Chatting with your GPT-2 model on {device}. Type 'quit' or 'exit' to stop.")
print("Heads up: base GPT-2 completes text. It is not instruction/chat fine-tuned.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"quit", "exit"}:
        break
    if not user_input:
        continue

    prompt = user_input
    reply = generate(prompt)
    print(f"GPT: {reply}\n")
