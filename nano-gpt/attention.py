import os
import time
import tiktoken
from torch.nn import functional as F
import torch.nn as nn
import torch

# hyperparameters
batch_size = 16
block_size = 64
learning_rate = 1e-3
max_iters = 100000
eval_iters = 200
eval_interval = 1000
n_embed = 32
n_head = 4
n_layer = 4
dropout = 0.2
should_train = False


# torch.manual_seed(1337)

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# TODO: redo with tiktoken
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# stoi = {s: i for i, s in enumerate(chars)}
# itos = {i: s for s, i in stoi.items()}
# def encode(s): return [stoi[s] for s in text]
# def decode(s): return ''.join([itos[i] for i in s])


enc = tiktoken.get_encoding("cl100k_base")
# enc = tiktoken.get_encoding("r50k_base")
vocab_size = enc.max_token_value + 1


data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


xb, yb = get_batch('train')
# print(xb.shape, yb.shape)


for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t + 1]
        target = yb[b, t]
        # print(f'{b} {t} {context.tolist()} -> {target}')


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        wei = q @ k.transpose(-1, -2) * C**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,head_size)

        out = wei @ v # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_ embd: embedding dimension, n head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # B = batch size, T = time, C = channel
        tok_emb = self.embedding(idx) # (B,T,C)
        pos_emb = self.positional_embedding(torch.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=10):
        # for _ in range(max_new_tokens):
        print(enc.decode(idx[0].tolist()), end="")
        while True:
            time.sleep(0.1)
            idx_cond = idx[:, -block_size:] # (B,T)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (B,C)
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            print(enc.decode(idx_next[0].tolist()), end="")
            idx = torch.cat([idx, idx_next], dim=1)
        return enc.decode(idx[0].tolist())


m = BigramLanguageModel()
if os.path.exists('model.pt'):
    m.load_state_dict(torch.load('model.pt'))

print(f'parameters {sum(p.numel() for p in m.parameters() if p.requires_grad)}')

# train
if should_train:
    m.train()
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
    start = time.time()
    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            dur = time.strftime("%M:%S", time.gmtime((time.time() - start)))
            print(f'{iter} ({dur}) train_loss={losses["train"]:.2f} val_loss={losses["val"]:.2f}')
            start = time.time()
            torch.save(m.state_dict(), "./model.pt")

        xb, yb = get_batch('train')

        logits, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
m.eval()
input = 'Oh how my love'
m.generate(torch.tensor([enc.encode(input)]))
# max_new_tokens=300, do_sample=True)
# print(enc.decode(gen[0].tolist()))

