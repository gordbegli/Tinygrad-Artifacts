import os
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.optim import Adam

# Load and preprocess data
with open('data/names.txt', 'r') as f:
    words = f.read().splitlines()
Tensor.manual_seed(42)

# Create character mappings
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Prepare input and target data
xs = []
ys = []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = Tensor(xs)
ys = Tensor(ys)
num = len(xs)
W = Tensor.randn((27, 27), requires_grad=True)

# Initialize the optimizer
opt = Adam([W], lr=0.01)

# Enable training mode
Tensor.training = True

# Precompute one-hot encodings as they do not change
xenc = xs.one_hot(27).float()

# Training loop
for k in range(50):
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[Tensor.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    if k == 0:
        print(loss.numpy())
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(k)

    out = []
    if k == 49:
        print(loss.numpy())
        while True:
            xenc = Tensor.one_hot(Tensor([0]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdim=True)
            ix = Tensor.multinomial(p, num_samples=1, replacement=True).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

