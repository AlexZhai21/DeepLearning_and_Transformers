import math
import time

import torch

from datautils import DataLoaderLite
from models import GPT, GPTConfig


device = "cuda"
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="train")
torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig(vocab_size=50304)) #50304 is a much nicer number than our initial vocab size, since it contains many powers of 2.
model.to(device)
# model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

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
    optimizer.zero_grad()

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
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
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {step} loss: {loss_accum.item():.6f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}, norm: {norm:.4f}, lr: {lr:.4e}")
