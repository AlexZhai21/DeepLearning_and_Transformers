import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from urllib.request import urlretrieve

# hyperparameters
batch_size = 32 
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
# ------------

torch.manual_seed(1337)

input_path = Path(__file__).resolve().parent / 'data' / 'tinyshakespeare' / 'input.txt'
input_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

if not input_path.exists():
    print('Downloading input.txt...')
    urlretrieve(input_url, input_path)

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) #B, T, head_size * num_heads
        return self.proj(out)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias= False) #B, T, head_size
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        weights = q @ k.transpose(-2, -1) * C**-0.5 #B, T, T
        weights = weights.masked_fill(self.tril[:T, :T] ==0, float('-inf')) #self.tril[:T, :T], slices it down to the actual sequence length

        weights = F.softmax(weights, dim = -1)
        out = weights @ v
        return out
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
    def forward(self, x):
        return self.net(x)
class AttnBlock(nn.Module):
    def __init__(self, num_heads, n_embed):
        super().__init__()
        self.heads = MultiHead(num_heads,  n_embed// num_heads)
        self.ffn = FeedForward(n_embed)
    def forward(self, x):
        x = x + self.heads(x)
        x = x + self.ffn(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            AttnBlock(4, n_embed),
            AttnBlock(4, n_embed),
            AttnBlock(4, n_embed),
            AttnBlock(4, n_embed)
        )
        self.ln_head= nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers.
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device =device)) #T,C
        x= tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.ln_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
    # idx is a (B, T) array of indices in the current context.
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens.
            idx_cond = idx[:, -block_size:]

            # Get the predictions.
            logits, loss = self(idx_cond)

            # Focus only on the last time step.
            logits = logits[:, -1, :]  # (B, C)

            # Apply softmax to get probabilities.
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution.
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence.
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx



model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


