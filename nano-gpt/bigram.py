from torch.nn import functional as F
import torch.nn as nn
import torch

# hyperparameters
block_size = 8
batch_size = 32
learning_rate = 0.001
max_iters = 10000
eval_iters = 200
eval_interval = 1000
n_embed = 32


torch.manual_seed(1337)

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# TODO: redo with tiktoken
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
def encode(s): return [stoi[s] for s in text]
def decode(s): return ''.join([itos[i] for i in s])


data = torch.tensor(encode(text), dtype=torch.long)
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


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # B = batch size, T = time, C = channel
        tok_emb = self.embedding(idx) # (B,T,C)
        pos_emb = self.positional_embedding(torch.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, max_new_tokens):
        idx = torch.zeros((1, 1), dtype=torch.long)
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] # (B,C)
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat([idx, idx_next], dim=1)
        return decode(idx[0].tolist())


m = BigramLanguageModel()
logits, loss = m(xb, yb)
# print(loss, logits.shape)

# train
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'iter={iter} train_loss={losses["train"]:.2f} val_loss={losses["val"]:.2f}')

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(m.generate(max_new_tokens=100))
