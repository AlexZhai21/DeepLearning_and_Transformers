import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass 
import math
import transformers 
import tiktoken
import os
n_embd = 768
img_size = 28
patch_size =4
in_channels = 1
batch_size = 8
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, n_embd):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Linear(in_channels * patch_size * patch_size, n_embd)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(B, -1, C * P * P)

        x = self.proj(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, num_classes, device = "cuda"): #the input x right here is the patches matrix so b x n x patch_dim
        super().__init__()
        self.device = device
        
        self.patch_embd = PatchEmbed(img_size, patch_size, in_channels, n_embd)
        self.classification_head = nn.Linear(n_embd, num_classes, bias = False)
        self.pos_emb = nn.Embedding(self.patch_embd.num_patches + 1, n_embd)
        self.architecture = nn.ModuleList([Block(num_heads) for _ in range(num_layers)])
        self.cls = nn.Parameter(torch.randn(1, 1, n_embd))


    def forward(self, x):
        x = self.patch_embd(x) # b x n x n_embd
        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat((cls, x), dim = 1)
     
        pos = torch.arange(0, self.patch_embd.num_patches + 1, dtype = torch.long, device = self.device)
        x = x + self.pos_emb(pos) #b x n + 1 x n_embd
        
        
        for block in self.architecture:
            x = block(x)
        y = x[:, 0, :] #this is the cls token
        logits = self.classification_head(y) #logits over a vector which is num_classes dim
        probabilities = F.softmax(logits, dim = -1) #probiabilities
        return logits
class SelfAttention(nn.Module):
    def __init__(self, head_size): 
        super().__init__()
        self.q = nn.Linear(n_embd, head_size, bias = False)
        self.k = nn.Linear(n_embd, head_size, bias = False)
        self.v = nn.Linear(n_embd, head_size, bias = False)
        self.head_size = head_size
        self.last_weights = None
    def forward(self, x): #x is b x n + 1 x n_embd, THE INPUT X SHOULD be the image patches passed through the linear layer
        q_x = self.q(x) #b x n + 1 x head_size
        k_x = self.k(x) #b x n + 1 x head_size
        v_x = self.v(x) #b x n + 1 x head_size
        attention = q_x @ k_x.transpose(-2, -1) * (self.head_size ** -0.5) # (n + 1) x (n + 1)  #make sure to do scaled self attention
        weights = F.softmax(attention, dim = -1) # (n + 1) x (n + 1)
        self.last_weights = weights.detach()
        y = weights @ v_x # (n + 1) x head_size
        return y
    
class MultiSA(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        assert n_embd % num_heads == 0, f"please choose a number of heads that divides {n_embd}"
        self.attention_heads = nn.ModuleList([SelfAttention(n_embd // num_heads) for _ in range(num_heads)])
        self.w0 = nn.Linear(n_embd, n_embd, bias = False)
    def forward(self, x):
        out = torch.cat([head(x) for head in self.attention_heads], dim = -1) # b x (n + 1) x n_embd, n_embd is just head_size * num_heads
        return self.w0(out) #mixes the information from the heads 

class Block(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.msa = MultiSA(self.num_heads)
        self.mlp = MLP()
        self.ln1 = nn.LayerNorm(normalized_shape= n_embd)
        self.ln2 = nn.LayerNorm(normalized_shape = n_embd)
    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(n_embd, 4 * n_embd, bias = False)
        self.l2 = nn.Linear(4 * n_embd, n_embd, bias = False)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.l1(x) # (n + 1) x 4 * n_embd
        x = self.gelu(x) # (n + 1) x 4 * n_embd
        x = self.l2(x) # (n + 1) x n_embd
        return x


def patchify(images, patch_size):
    # images: (B, C, H, W) -> patches: (B, num_patches, C * patch_size * patch_size)
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(B, -1, C * patch_size * patch_size)


def get_mnist_loaders(batch_size=64, data_dir="visiontransformer_data"):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        if logits is None:
            raise RuntimeError("VisionTransformer.forward() returned None. Finish the model forward pass so it returns class logits.")
        if logits.ndim != 2:
            raise RuntimeError(f"Expected logits shape (B, num_classes), got {tuple(logits.shape)}")

        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def save_cls_attention_heatmap(model, data_loader, device, output_path="visiontransformer_data/attention_to_cls_heatmap.png"):
    from pathlib import Path
    from PIL import Image

    model.eval()
    images, labels = next(iter(data_loader))
    original_image = images[0, 0].detach().cpu()
    images = images.to(device)
    _ = model(images[:1])

    last_block = model.architecture[-1]
    head_maps = [head.last_weights[0].detach().cpu() for head in last_block.msa.attention_heads]
    attention = torch.stack(head_maps, dim=0).mean(dim=0)

    # attention[row, col] means token at row attends to token at col.
    # Column 0 is the CLS token, rows 1: are image patch tokens.
    patch_attention_to_cls = attention[1:, 0]
    grid_size = int(model.patch_embd.num_patches ** 0.5)
    heat = patch_attention_to_cls.view(grid_size, grid_size)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    heat_np = (heat.numpy() * 255).astype("uint8")
    heat_img = Image.fromarray(heat_np, mode="L").resize((280, 280), resample=Image.Resampling.NEAREST)
    heat_img = heat_img.convert("RGB")

    heat_pixels = heat_img.load()
    for y in range(heat_img.height):
        for x in range(heat_img.width):
            v = heat_pixels[x, y][0]
            heat_pixels[x, y] = (v, 0, 255 - v)

    original_image = (original_image * 0.3081) + 0.1307
    original_image = original_image.clamp(0, 1)
    original_np = (original_image.numpy() * 255).astype("uint8")
    original_img = Image.fromarray(original_np, mode="L").resize((280, 280), resample=Image.Resampling.NEAREST)
    original_img = original_img.convert("RGB")

    img = Image.blend(original_img, heat_img, alpha=0.45)

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    img.save(output_path)
    print(f"saved final-layer CLS attention overlay to {output_path.resolve()}")


def train_mnist():
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    num_layers = 4
    num_heads = 8
    num_classes = 10
    learning_rate = 3e-4
    epochs = 1

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    model = VisionTransformer(
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    print(f"device: {device}")
    print(f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")
    print(f"patches per image: {model.patch_embd.num_patches}, patch size: {model.patch_embd.patch_size}")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            if logits is None:
                raise RuntimeError("VisionTransformer.forward() returned None. Finish the model forward pass so it returns class logits.")
            if logits.ndim != 2:
                raise RuntimeError(f"Expected logits shape (B, num_classes), got {tuple(logits.shape)}")

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

            if step % 100 == 0:
                running_loss = total_loss / total_examples
                running_acc = total_correct / total_examples
                print(
                    f"epoch {epoch + 1}/{epochs} step {step:04d} "
                    f"loss {running_loss:.4f} acc {running_acc:.4f}"
                )

        train_loss = total_loss / total_examples
        train_acc = total_correct / total_examples
        test_loss, test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0

        print(
            f"epoch {epoch + 1}/{epochs} done in {dt:.2f}s | "
            f"train loss {train_loss:.4f} train acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} test acc {test_acc:.4f}"
        )

    save_cls_attention_heatmap(model, test_loader, device)


if __name__ == "__main__":
    train_mnist()





