import time
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from visiontransformer import VisionTransformer


def get_mnist_loaders(batch_size=64, data_dir="visiontransformer_data"):
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
        if logits.ndim != 2:
            raise RuntimeError(f"Expected logits shape (B, num_classes), got {tuple(logits.shape)}")

        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def save_cls_attention_heatmap(
    model,
    data_loader,
    device,
    output_path="visiontransformer_data/attention_to_cls_heatmap.png",
):
    model.eval()
    images, labels = next(iter(data_loader))
    original_image = images[0, 0].detach().cpu()
    images = images.to(device)
    _ = model(images[:1])

    last_block = model.architecture[-1]
    head_maps = [head.last_weights[0].detach().cpu() for head in last_block.msa.attention_heads]
    attention = torch.stack(head_maps, dim=0).mean(dim=0)

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
