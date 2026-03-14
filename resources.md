# Learning Resources

A structured path from CUDA basics to Flash Attention 4. Organized roughly in the order I plan to work through them, starting from where I am now -- comfortable with thread/block hierarchy, strided loops, and unified memory -- and working toward understanding modern attention kernels.

---

## 0. C/C++ for CUDA

CUDA is C++-- `nvcc` is a C++ compiler with GPU extensions. I don't need all of C++, but I need more than just C. The memory model and pointer discipline come from C; the syntax and patterns I'll actually write come from C++.

What I need to cover:

- Pointers and pointer arithmetic (every kernel takes raw pointers)
- `malloc`/`free` and manual memory management (maps directly to `cudaMalloc`/`cudaFree`)
- Arrays as pointers, structs, basic type casting
- C++ additions that show up in CUDA code: `auto`, references, structs with methods, basic templates (reading them, not writing complex ones)
- Compiling from the command line with `g++` / `nvcc`

Resources:

- [Beej's Guide to C Programming](https://beej.us/guide/bgc/) -- for pointer fundamentals: pointers (ch. 5, 11, 23), structs (ch. 8, 20), and memory allocation (ch. 12)
- [learncpp.com](https://www.learncpp.com/) -- for the C++ bits: references, `auto`, structs with member functions, templates basics
- [CS50 C lectures](https://cs50.harvard.edu/x/) -- if I want video walkthroughs, weeks 1-5 cover the C foundations

Exercises to do:

1. Allocate a float array with `malloc`, fill it with values, compute the sum, `free` it
2. Redo exercise 1 using `new`/`delete` instead of `malloc`/`free`
3. Write a function that takes a pointer to an array and its length, modifies elements in place. Then write a version using references.
4. Port exercise 1 to CUDA: replace `malloc` with `cudaMallocManaged`, write a kernel that fills the array

> Note: The goal is fluency with pointers, manual memory, and reading C++ syntax -- not mastering the language. Once exercise 4 feels natural, move on.

---

## 1. CUDA Fundamentals

Where I need to go next: memory hierarchy (global, shared, registers, L1/L2), coalesced access patterns, `__syncthreads()`, and profiling with `ncu`.

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) -- chapters on memory hierarchy are the priority
- [GPU MODE Lectures 1-5](https://github.com/gpu-mode/lectures) -- covers fundamentals, profiling, and memory
- [Christian Mills' GPU MODE Lecture Notes](https://christianjmills.com/series/notes/cuda-mode-notes.html) -- written summaries for each GPU MODE lecture, easier to revisit than videos
- [NVIDIA "Even Easier Introduction to CUDA"](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) -- already used this for the basics, but worth revisiting for memory management sections

Exercises to do:

1. Matrix transpose (naive, then shared memory tiled)
2. Parallel reduction (sum of an array) -- teaches warp shuffles and shared memory reduction trees
3. Matrix multiplication (naive, then tiled with shared memory)

---

## 2. Optimizing Matrix Multiplication

MatMul is the canonical GPU optimization exercise. Going from a naive implementation to near-cuBLAS performance teaches tiling, register blocking, memory coalescing, and bank conflicts -- all concepts that show up again in attention kernels.

- [Simon Boehm: "How to Optimize a CUDA Matmul Kernel"](https://siboehm.com/articles/22/CUDA-MMM) -- step-by-step from naive to near-cuBLAS, probably the single most useful tutorial for this stage
- [GPU MODE Lecture 6](https://github.com/gpu-mode/lectures) -- CUDA matmul
- ~~Lezcano: "CUDA beyond the basics"~~ -- site taken down (lezcano.github.io no longer exists, no archived copy available)

---

## 3. Triton (optional intermediate step)

Triton sits between raw CUDA and CuTe-DSL in abstraction level. It hides memory management details while still exposing tiling and blocking patterns. If writing raw CUDA kernels feels like too big a jump from Python, Triton is a good stepping stone before CuTe-DSL.

- [Triton documentation and tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [GPU MODE Triton lectures](https://github.com/gpu-mode/lectures)

Not strictly required -- I can go straight from CUDA fundamentals to CuTe-DSL if the concepts from sections 1-2 are solid.

---

## 4. Flash Attention (FA1 & FA2)

The core idea: tiling + online softmax to avoid materializing the full N×N attention matrix in HBM. This is where my transformer knowledge meets GPU programming.

- [Aleksa Gordic: "ELI5: Flash Attention"](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) -- start here for intuition before touching the papers
- [GPU MODE Lecture 12: Flash Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-012/) -- walks through the algorithm and how it maps to GPU hardware
- [Tri Dao: FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) -- read after building intuition from the above two
- [Gau-Nernst: "Writing Speed-of-Light Flash Attention for 5090 in CUDA C++"](https://gau-nernst.github.io/fa-5090/) -- practical walkthrough of implementing FA from scratch

Prerequisites before starting this section: comfortable with tiled matmul in shared memory, understand the GPU memory hierarchy well enough to reason about data movement costs.

---

## 5. CuTe Layout Algebra and CuTe-DSL

CuTe-DSL is a Python DSL -- part of CUTLASS 4 -- that compiles to PTX via MLIR. It replaces the need to learn CUTLASS C++ templates. FA4 is written entirely in CuTe-DSL.

I don't need to learn CUTLASS C++. But I do need to understand the conceptual framework that CuTe-DSL shares with CuTe C++:

- Layout algebra: shapes and strides, how `(2048,2048):(2048,1)` describes a row-major matrix
- TV (Thread-Value) layouts: how threads map to data elements through composed layout transformations
- Tiling and partitioning: breaking problems into tiles for shared memory, then per-thread register fragments
- Memory hierarchy orchestration: staging data through global → shared → registers, async copies, pipelining

These concepts are what make CuTe powerful. The Python syntax is just the surface.

Resources for CuTe-DSL:

- [NVIDIA CuTe DSL Overview & Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) -- official docs, designed for newcomers
- [Chris Choy: "CuTe DSL Basics"](https://chrischoy.org/posts/cutedsl-basics/) -- starts from a first kernel, self-contained
- [Simon Veitner: "Bridging Math and Code: CuTe Layout Algebra in CuTeDSL"](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/) -- learn layout algebra first
- [Simon Veitner: "An applied introduction to CuTeDSL"](https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/) -- assumes layout algebra, shows naive → vectorized → TV layout kernel with bandwidth numbers
- [Ian Barber: "CuTe-DSL"](https://ianbarber.blog/2025/07/04/cute-dsl/) -- practical perspective, compares to Triton
- [CUTLASS repo examples](https://github.com/NVIDIA/cutlass) -- `examples/python/CuTeDSL/` has GEMM and FMHA implementations
- [NVIDIA blog: "Achieve CUTLASS C++ Performance with Python APIs Using CuTe DSL"](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)

---

## 6. Flash Attention 3, 4, and Beyond

FA3 introduced pipelining and warp specialization on Hopper. FA4 -- written in CuTe-DSL -- pushes further with algorithm-kernel co-design for Blackwell, reporting up to 1605 TFLOPs/s on B200 (71% utilization).

- [Colfax Research: "CUTLASS and FlashAttention-3"](https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/) -- bridges CUTLASS concepts with FA3
- [FA4 paper](https://arxiv.org/abs/2603.05451) -- the source
- [Together AI: "FlashAttention-4" blog post](https://www.together.ai/blog/flashattention-4) -- overview from the team
- [Modal: "We reverse-engineered Flash Attention 4"](https://modal.com/blog/reverse-engineer-flash-attention-4) -- probably the most useful resource for understanding FA4's internals
- [Colfax Research: "FlexAttention in Flash Attention CuTe DSL"](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/) -- FlexAttention through CuTe-DSL
- [PyTorch blog: "FlexAttention + FlashAttention-4"](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) -- how FA4 fits into the PyTorch ecosystem
- [Tyler Crosse: "FlashAttention & LLM Inference on GPUs"](https://www.tylercrosse.com/ideas/2026/gpu-p6) -- contextualizes FA in the broader inference stack
- [DeepWiki: Flash Attention CuTe-DSL Implementation](https://deepwiki.com/Dao-AILab/flash-attention/2.4-cute-based-implementation) -- annotated walkthrough of the codebase

---

## 7. Communities

- [GPU MODE Discord](https://discord.gg/gpumode) -- already aware of this from resource-stream. Active community for GPU programming questions.

---

## Study Plan


| Step | Topic                    | Exercise                                           |
| ---- | ------------------------ | -------------------------------------------------- |
| 0    | C/C++ for CUDA           | Pointers, manual memory, port to cudaMallocManaged |
| 1    | Shared memory, profiling | Tiled matrix transpose                             |
| 2    | Reductions               | Parallel sum with warp shuffles                    |
| 3    | Matrix multiplication    | Naive → tiled → register-tiled matmul              |
| 4    | Online softmax           | Single-pass softmax kernel                         |
| 5    | Flash Attention 1/2      | Read paper + implement simplified version          |
| 6    | CuTe layout algebra      | Work through Veitner's layout algebra post         |
| 7    | CuTe-DSL kernels         | Transpose → GEMM in CuTe-DSL                       |
| 8    | FA3/FA4                  | Study the codebase with Modal's guide              |


> Note: I don't need a Blackwell GPU for most of this. FA1/FA2 concepts apply on any CUDA-capable GPU. Even studying FA4's design decisions teaches GPU thinking regardless of the hardware I'm running on.

> Note: CUTLASS C++templates are not a prerequisite. CuTe-DSL was designed to replace that path. The concepts (layout algebra, tiling, TV layouts) are the same -- the language is Python, not C++.

