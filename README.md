Right now this repo is a place for me to put artifacts from following Neural Networks: Zero to Hero in Tinygrad.import os
import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor, dtypes
with open('sandbox/names.txt', 'r') as f:
    words = f.read().splitlines()
Tensor.manual_seed(42)

N_array = np.zeros((27, 27), dtype=np.float16)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

i = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N_array[ix1, ix2] += 1
        i += 1

N = Tensor(N_array, dtype=dtypes.float16)

for i in range(20):
    ix = 0
    out = []
    while True:
        row = N[ix]
        row_probs = row / row.sum()
        ix = row_probs.multinomial(num_samples=1).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))