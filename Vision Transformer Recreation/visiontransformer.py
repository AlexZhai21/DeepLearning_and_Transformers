import torch
import torch.nn as nn
from torch.nn import functional as F

n_embd = 768
img_size = 28
patch_size = 4
in_channels = 1


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

        return self.proj(x)


class VisionTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, num_classes):
        super().__init__()
        self.patch_embd = PatchEmbed(img_size, patch_size, in_channels, n_embd)
        self.classification_head = nn.Linear(n_embd, num_classes, bias=False)
        self.pos_emb = nn.Embedding(self.patch_embd.num_patches + 1, n_embd)
        self.architecture = nn.ModuleList([Block(num_heads) for _ in range(num_layers)])
        self.cls = nn.Parameter(torch.randn(1, 1, n_embd))

    def forward(self, x):
        x = self.patch_embd(x)
        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)

        pos = torch.arange(
            0,
            self.patch_embd.num_patches + 1,
            dtype=torch.long,
            device=x.device,
        )
        x = x + self.pos_emb(pos)

        for block in self.architecture:
            x = block(x)

        y = x[:, 0, :]
        return self.classification_head(y)


class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size
        self.last_weights = None

    def forward(self, x):
        q_x = self.q(x)
        k_x = self.k(x)
        v_x = self.v(x)
        attention = q_x @ k_x.transpose(-2, -1) * (self.head_size ** -0.5)
        weights = F.softmax(attention, dim=-1)
        self.last_weights = weights.detach()
        return weights @ v_x


class MultiSA(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        assert n_embd % num_heads == 0, f"please choose a number of heads that divides {n_embd}"
        self.attention_heads = nn.ModuleList(
            [SelfAttention(n_embd // num_heads) for _ in range(num_heads)]
        )
        self.w0 = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        return self.w0(out)


class Block(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.msa = MultiSA(num_heads)
        self.mlp = MLP()
        self.ln1 = nn.LayerNorm(normalized_shape=n_embd)
        self.ln2 = nn.LayerNorm(normalized_shape=n_embd)

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.l2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        return self.l2(x)
