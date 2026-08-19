import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from gpt2_huggingface import from_pretrained
from fine_tune_class import ChatSFTDataset
from data import load_oasst1_examples
from torch.utils.data import DataLoader


total_batch_size = 16384
B = 8
T = 512

examples = load_oasst1_examples()
dataset = ChatSFTDataset(examples, block_size=T)

device = "cuda"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)

print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(dataset, batch_size=B, shuffle=True, drop_last=True)
train_iter = iter(train_loader)

torch.set_float32_matmul_precision("high")

model = from_pretrained("gpt2")
model.to(device)
max_lr = 5e-5
min_lr = max_lr * 0.1
warmup_steps = 50
max_steps = 1000

optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=max_lr,
    device=device,
)

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


for step in range(max_steps):
    loss_accum = 0.0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(grad_accum_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()

    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0) * 1000

    tokens_per_sec = (B * T * grad_accum_steps) / (t1 - t0)

    print(
        f"step {step} loss: {loss_accum.item():.6f}, "
        f"dt: {dt:.2f}ms, "
        f"tok/sec: {tokens_per_sec:.2f}, "
        f"norm: {norm:.4f}, "
        f"lr: {lr:.4e}"
    )
save_dir = Path(__file__).resolve().parent / "final_model"
save_dir.mkdir(exist_ok=True)

torch.save(
    {
        "model": model.state_dict(),
        "config": model.config,
        "optimizer": optimizer.state_dict(),
        "step": max_steps,
    },
    save_dir / "gpt2_sft_final.pt",
)
