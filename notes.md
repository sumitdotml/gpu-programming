# NOTES

Table of contents

- [<< bit shift operator](#<<-bit-shift-operator)
- [float literals (`1.0f`) versus double literals (`1.0`)](#float-literals-10f-versus-double-literals-10)
- [Pointers and the `*` symbol](#pointers-and-the--symbol)
- [Regular variables vs pointers (`float p` vs `float *p`)](#regular-variables-vs-pointers-float-p-vs-float-p)
- [delete vs `delete[]`](#delete-vs-delete)
- [CUDA threads and blocks](#cuda-threads-and-blocks)
- [Strided loop pattern](#strided-loop-pattern)
- [Mental model for 1D grids/blocks](#mental-model-for-1d-grids-blocks)

#### `<<` bit shift operator <a name="<<-bit-shift-operator"></a>

This is the left bit shift operator. It shifts the binary representation of a number to the left by a specified number of positions.

How it works:

1 << 20 means "take the number 1 and shift it left by 20 bit positions."

1 in binary: 0000000000000000000001

Shift left 20 positions:

                100000000000000000000
                ^
                This "1" moved 20 places left

Each left shift by 1 doubles the number (multiplies by 2):

| Expression | Binary      | Decimal         |
| ---------- | ----------- | --------------- |
| 1 << 0     | 1           | 1               |
| 1 << 1     | 10          | 2               |
| 1 << 2     | 100         | 4               |
| 1 << 3     | 1000        | 8               |
| 1 << 10    | 10000000000 | 1,024 (1K)      |
| 1 << 20    | ...         | 1,048,576 (~1M) |

So 1 << 20 = 2²⁰ = 1,048,576

This is approximately 1 million, which is why the comment says "1M elements."

Why use this instead of just writing 1000000?

- Powers of 2 are common in computing (memory sizes, buffer sizes, etc.)
- 1 << 20 makes it clear you want exactly 2²⁰
- It's a common idiom in systems programming

---

#### float literals (`1.0f`) versus double literals (`1.0`) <a name="float-literals-10f-versus-double-literals-10"></a>

For the following:

```cpp
  for (int i = 0; i < N; i++) {
  x[i] = 1.0f;
  y[i] = 2.0f;
  }
```

The f suffix indicates that these are float literals (32-bit), not double literals (64-bit).

In C/C++:

| Literal | Type   | Size   |
| ------- | ------ | ------ |
| 1.0     | double | 64-bit |
| 1.0f    | float  | 32-bit |

Why use f?

If my arrays are declared as float:
float *x = new float[N];
float *y = new float[N];

Using 1.0f and 2.0f ensures I'm assigning a float to a float, avoiding any implicit conversion from double to float.

After the above example loop:
x = [1.0, 1.0, 1.0, 1.0, ...] (1 million 1.0s)
y = [2.0, 2.0, 2.0, 2.0, ...] (1 million 2.0s)

---

#### Pointers and the `*` symbol <a name="pointers-and-the--symbol"></a>

The `*` symbol has two different meanings depending on context:

**In a declaration, means "this is a pointer"**

```cpp
float *x = new float[N];
```

Here, `*` is part of the type declaration, saying "x is a pointer to float."

**In an expression, means "dereference" (get value at address)**

```cpp
*x    // "get the value that x points to"
```

So when I write `float *x`, I'm declaring a pointer. But when I write `*x` in an expression, I'm accessing the value at that address.

Since `x` points to the first element of the array, `*x` gives me `x[0]`. To get other elements, I use pointer arithmetic:

| Array notation | Pointer arithmetic | Meaning                          |
| -------------- | ------------------ | -------------------------------- |
| `x[0]`         | `*(x + 0)` or `*x` | Value at address x               |
| `x[1]`         | `*(x + 1)`         | Value at address x + 1 position  |
| `x[5]`         | `*(x + 5)`         | Value at address x + 5 positions |
| `x[i]`         | `*(x + i)`         | Value at address x + i positions |

In fact, `x[i]` is just shorthand for `*(x + i)`; the compiler converts it internally.

---

#### Regular variables vs pointers (`float p` vs `float *p`) <a name="regular-variables-vs-pointers-float-p-vs-float-p"></a>

**Regular variable:**

```cpp
float p = 1.0f;
```

- Stores the value directly
- Lives on the stack (automatic memory)
- Automatically cleaned up when it goes out of scope

**Pointer:**

```cpp
float *p = new float;
*p = 1.0f;
```

- Stores a memory address pointing to where the value lives
- The actual value lives on the heap (dynamic memory)
- I must manually `delete` it when I'm done

When to use which:

| Regular variable           | Pointer                        |
| -------------------------- | ------------------------------ |
| Size known at compile time | Size determined at runtime     |
| Small data                 | Large data (arrays, objects)   |
| Short-lived                | Needs to outlive current scope |
| Automatic cleanup          | Manual memory management       |

---

#### `delete` vs `delete[]` <a name="delete-vs-delete"></a>

`delete[]` is specifically for arrays. For single objects, I use `delete` (without the brackets).

The rule is they must match:

| Allocation | Deallocation |
| ---------- | ------------ |
| `new`      | `delete`     |
| `new[]`    | `delete[]`   |

Examples:

```cpp
// Single object
float *p = new float;       // allocate one float
delete p;                   // free it

// Array
float *x = new float[N];    // allocate N floats
delete[] x;                 // free all of them
```

---

#### CUDA threads and blocks <a name="cuda-threads-and-blocks"></a>

Threads are the smallest unit of execution on a GPU. CUDA organizes them in a hierarchy:

```
Grid
 └── Blocks
      └── Threads
```

Visual:

```
Grid (my whole job)
┌─────────────────────────────────────────┐
│  Block 0     Block 1     Block 2        │
│ ┌───────┐   ┌───────┐   ┌───────┐       │
│ │ t t t │   │ t t t │   │ t t t │       │
│ │ t t t │   │ t t t │   │ t t t │       │
│ └───────┘   └───────┘   └───────┘       │
└─────────────────────────────────────────┘
  (t = thread)
```

When I launch a kernel with `add<<<numBlocks, threadsPerBlock>>>`:

- First number = how many blocks
- Second number = how many threads per block

For example:

| Launch            | Blocks | Threads per block | Total threads |
| ----------------- | ------ | ----------------- | ------------- |
| `<<<1, 1>>>`      | 1      | 1                 | 1             |
| `<<<1, 256>>>`    | 1      | 256               | 256           |
| `<<<4096, 256>>>` | 4096   | 256               | 1,048,576     |

Visual comparison:

```
<<<1, 1>>>
┌─────────┐
│ Block 0 │
│  ┌───┐  │
│  │ t │  │   1 thread total
│  └───┘  │
└─────────┘

<<<1, 256>>>
┌─────────────────────────────────────┐
│ Block 0                             │
│ ┌───┬───┬───┬───┬─────────────┬───┐ │
│ │ t │ t │ t │ t │ ... 252 ... │ t │ │   256 threads total
│ └───┴───┴───┴───┴─────────────┴───┘ │
└─────────────────────────────────────┘

<<<4096, 256>>>
┌───────────────────────────────────────────────────────────────────────────────┐
│ Block 0           Block 1           Block 2                      Block 4095   │
│ ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐           ┌───┬───┬───┐ │
│ │ t │...│...│ t │ │ t │...│...│ t │ │ t │...│...│ t │  ... 4093 │ t │...│ t │ │
│ └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┴───┘           └───┴───┴───┘ │
│    256 threads       256 threads       256 threads               256 threads  │
└───────────────────────────────────────────────────────────────────────────────┘
            4096 blocks × 256 threads = 1,048,576 threads total (1M)
```

With `<<<1, 256>>>`, I get 256 threads running simultaneously. Each thread gets a unique `threadIdx.x` (0 to 255), so they can each work on a different element.

```cu
int i = threadIdx.x;  // thread 0 gets i=0, thread 1 gets i=1, ... thread 255 gets i=255
y[i] = x[i] + y[i];   // each thread handles one element
```

But if I have 1M elements and only 256 threads, only elements 0-255 get processed. The rest are never touched.

To process all 1M elements, I either need more threads (like `<<<4096, 256>>>`), or I can use a strided loop.

---

#### Strided loop pattern <a name="strided-loop-pattern"></a>

The strided loop lets a small number of threads handle a large array. Instead of each thread doing one element, each thread loops through multiple elements.

```cu
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
```

With `<<<1, 256>>>`:

- `threadIdx.x` = 0 to 255 (each thread's ID)
- `blockDim.x` = 256 (total threads per block)

Each thread starts at its index and jumps by 256 each iteration:

```
Thread 0:   i = 0,   256,   512,   768, ...
Thread 1:   i = 1,   257,   513,   769, ...
Thread 2:   i = 2,   258,   514,   770, ...
...
Thread 255: i = 255, 511,   767,   1023, ...
```

So with 1M elements and 256 threads, each thread processes 4,096 elements.

This combines parallelism and sequencing:

- **Parallel:** All 256 threads run at the same time
- **Sequential:** Each thread loops through its ~4,096 elements one by one

```
Time →

Thread 0:   [i=0] → [i=256] → [i=512] → [i=768] → ...
Thread 1:   [i=1] → [i=257] → [i=513] → [i=769] → ...
Thread 2:   [i=2] → [i=258] → [i=514] → [i=770] → ...
   ↓           ↓        ↓         ↓         ↓
 (all running simultaneously, each doing its own sequence)
```

| Approach                              | Analogy                                                     |
| ------------------------------------- | ----------------------------------------------------------- |
| `<<<1, 1>>>`                          | 1 worker doing all 1M tasks sequentially                    |
| `<<<1, 256>>>` with strided loop      | 256 workers, each doing ~4K tasks sequentially, all at once |
| `<<<4096, 256>>>` with 1 element each | 1M workers, each doing 1 task                               |

More threads = more parallelism, but there's a limit to how many threads a GPU can run efficiently. The strided loop lets me balance parallelism with sequential work per thread.

---

#### Mental model for 1D grids/blocks <a name="mental-model-for-1d-grids-blocks"></a>

A useful way to picture this in my head (for 1D):

**A block is like a Python list of threads:**

```python
block = [t, t, t, t, ..., t]  # a list of threads
len(block) = blockDim.x       # total threads in block
block[i] = threadIdx.x = i    # each thread's index
```

**A grid is like a list of blocks (list of lists):**

```python
grid = [
    [t, t, t, ..., t],   # block 0 → blockIdx.x = 0
    [t, t, t, ..., t],   # block 1 → blockIdx.x = 1
    [t, t, t, ..., t],   # block 2 → blockIdx.x = 2
    ...
]

len(grid) = gridDim.x              # total blocks
len(grid[0]) = blockDim.x          # threads per block
grid[b][t] → blockIdx.x = b, threadIdx.x = t
```

So I have two indexes:

| Index         | What it identifies             | Analogy                         |
| ------------- | ------------------------------ | ------------------------------- |
| `blockIdx.x`  | Which block in the grid        | `grid[blockIdx.x]`              |
| `threadIdx.x` | Which thread within that block | `grid[blockIdx.x][threadIdx.x]` |

**Global index (to access my data array):**

When I need a single flat index to access my data, I combine them:

```cu
int global_index = blockIdx.x * blockDim.x + threadIdx.x;
```

Like flattening a 2D list into 1D. For a thread in Block 1 with `threadIdx.x = 3` and `blockDim.x = 256`:

```
blockIdx.x * blockDim.x = 1 * 256 = 256   ← "Block 1 starts at global index 256"
              + threadIdx.x = + 3         ← "I'm the 4th thread in my block"
                              -----
                               259        ← "So my global index is 259"
```

---
