import os
import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import Adam

with open('data/names.txt', 'r') as f:
    words = f.read().splitlines()
Tensor.manual_seed(42)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

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
opt = Adam([W], lr=0.1)

# Add this after initializing the optimizer
g = Tensor.manual_seed(2147483647)

# Need to enable training mode
Tensor.training = True  # <--- Added line

for k in range(200):
    xenc = xs.one_hot(27).float()
    print("One-hot encoded input shape:", xenc.shape)
    
    logits = xenc @ W 
    print("Logits shape:", logits.shape)
    
    counts = logits.exp() #Formatting
    print("Counts shape:", counts.shape)
    
    probs = counts / counts.sum(1, keepdim=True)
    print("Probabilities shape:", probs.shape)

    loss = -probs[Tensor.arange(num), ys].log().mean()  + 0.01 * (W**2).mean() # fancy indexing
    print("Loss:", loss.numpy())

    opt.zero_grad() #This is the "real" way, still's slow
    print("Gradients zeroed")
    
    loss.backward()
    print("Backward pass completed")
    
    opt.step()
    print("Optimizer step taken")
    print("Updated W shape:", W.shape)
