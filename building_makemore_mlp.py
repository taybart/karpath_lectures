import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

# hyperparameters
block_size = 3 # context length; how many characters do we take to predict the next one?
embedding_size = 10 # how many dimensions do we use to represent each character
hidden_size = 200
batch_size = 32


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

C = torch.randn((vocab_size, embedding_size))
W1 = torch.randn((embedding_size * block_size, hidden_size))
b1 = torch.randn(hidden_size)
W2 = torch.randn((hidden_size, vocab_size))
b2 = torch.randn(vocab_size)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

lossi = []
stepi = []

for i in range(200000):

    # construct batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    # forward pass
    emb = C[Xtr[ix]] # (32, 3, 2)
    h = torch.tanh(emb.view(-1, embedding_size * block_size) @ W1 + b1) # (32, 100)
    logits = h @ W2 + b2 # (32, 27)
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    learning_rate = 0.09 if i < 150000 else 0.01
    for p in parameters:
        p.data += -learning_rate * p.grad

    # stats
    stepi.append(i)
    lossi.append(loss.log10().item())

plt.plot(stepi, lossi)


# full model loss (training)
emb = C[Xtr] # (32, 3, 2)
h = torch.tanh(emb.view(-1, embedding_size * block_size) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytr)
print(loss.item())

# full model loss (dev), if training and dev losses are similar the model is too simple
emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, embedding_size * block_size) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(-1, embedding_size * block_size) @ W1 + b1) # (32, 100)
        logits = h @ W2 + b2 # (32, 27)
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(ix)
    print(''.join(itos[i] for i in out))


# notes
# torch.cat(emb[:, 0, :],emb[:, 1, :],emb[:, 2, :],1)
# torch.cat(torch.unbind(emb, 1)) # cat is slow
# emb.view(32,6) <- fastest
# emb.view(emb.shape[0], 6) @ W1 + b1
# emb.view(-1, 6) @ W1 + b1
