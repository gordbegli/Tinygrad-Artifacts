4.1

Cap. G80 is limited to 512 threads per block.

---

4.2

5 or less. Mf is using shared memory without syncing, so you need to constrain the block width to inhibit the creation of more than 1 warp.

---

4.3

Call syncthreads before reading from shared memory.

---

5.1

No. Since each element is accessed only once in order to complete the operation, moving data from global memory to shared memory will not reduce global memory bandwidth consumption.

---

5.2

n/a (can't draw)

---

5.3

The first barrier stops threads from executing on old data.
The second barrier stops threads from messing with shared memory until all computations using it are complete.

---

5.4

If the global memory bandwidth is small, then it may still be performant to cache reused values in shared memory to prevent a bottleneck when moving items from global memory to registers (even if those registers can store all of the info).

---

6.1

kernel from figure six point two
```cpp
#define SECTION_SIZE 512  // Elements reduced per block

// — Figure 6.2 Complete Sum Reduction Kernel :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
__global__ void reduce_f62_orig(const float* __restrict__ d_in,
                                float*       __restrict__ d_out)
{
    // full‑size shared array (one slot per thread)
    __shared__ float partialSum[SECTION_SIZE];

    // thread index within the block
    unsigned int t   = threadIdx.x;
    // compute global index into input array
    unsigned int idx = blockIdx.x * blockDim.x + t;

    // load one element from global into shared memory
    partialSum[t] = d_in[idx];
    __syncthreads();

    // binary‑tree reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        if (t % (2 * stride) == 0)
            partialSum[t] += partialSum[t + stride];
    }

    // write the block’s result back to global memory
    if (t == 0)
        d_out[blockIdx.x] = partialSum[0];
}
```

kernel from figure six point four
```cpp
#define SECTION_SIZE 512  // Elements reduced per block

// — Figure 6.4 Complete Sum Reduction Kernel (revised) :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
__global__ void reduce_f62_compact(const float* __restrict__ d_in,
                                   float*       __restrict__ d_out)
{
    // full‑size shared array (same size, but fewer active threads)
    __shared__ float partialSum[SECTION_SIZE];

    unsigned int t   = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + t;

    // load one element from global into shared memory
    partialSum[t] = d_in[idx];
    __syncthreads();

    // reduction with half‐width strides to avoid thread‐modulo divergence
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }

    // write the block’s result back to global memory
    if (t == 0)
        d_out[blockIdx.x] = partialSum[0];
```

[notes](https://chatgpt.com/share/68014784-cb60-8011-b373-3c29ba80efad)

---

6.2

The optimization is the same, but the second kernel would be more efficient post-optimization because it's using bit shifts instead of mod.

---

6.3

```english
<<<launch kernel>>> {

load a portion of the input array into a variable that resides in shared memory

while there are pairs in the input tile
  sum pairs
  store their sum into the left half of shared memory

store the first value in shared memory (now the sum) into the appropriate index in the output matrix based on block index, thread index, and block size
}
```

---

6.4

```english
load input into global memory

do a tree reduction on the input, using the kernel from 6.3 

return the reduced value
```

---

6.5

n/a (can't draw)

---

6.6

```english
<<<kernel that takes two inputs tiles, block_dim = 3x3, thread dim= >>> {





}
```


















