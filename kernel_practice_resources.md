# Kernel Practice Resources

Last researched: 2026-07-07

This file records the external practice resources found while looking for incremental CUDA and GPU-kernel exercises. The current learning context is PMPP chapter 3: thread hierarchy, launch geometry, bounds checks, simple 1D/2D kernels, and the beginning of memory-access discipline.

The goal is to find places where I can practice writing kernels in increasing difficulty, preferably with correctness checks, small tasks, and a path toward ML kernels.

> Note: Platform details can change, so I should treat the links as the source of truth before depending on login, judge, GPU, or grading behavior.

---

## 1. How I Should Use This

The repo tutor contract still applies:

- I write the learning code.
- The assistant can explain, review, design tests, ask questions, and give hints.
- Full implementation answers require the exact override phrase from `AGENTS.md`.

The practical path I want is:

1. Use PMPP as the conceptual spine.
2. Use puzzle or judge resources for repetition.
3. Use ML-kernel resources once indexing, memory transfers, bounds checks, and simple timing feel ordinary.
4. Add custom PMPP chapter drills when external resources skip too many intermediate steps.

For now, a good drill is one that forces me to answer:

- Which data element does this thread own?
- What does `blockIdx.x * blockDim.x + threadIdx.x` mean here?
- Do extra threads safely do nothing?
- Is the memory access pattern simple enough to reason about?
- Can I check correctness with a CPU reference or known output?

---

## 2. Best Resources For Right Now

These are the best fit for PMPP chapter 3 and early kernel-writing practice.

| Resource | Stack | Why it fits now | Caveat |
| --- | --- | --- | --- |
| [GPU Puzzles](https://github.com/srush/GPU-Puzzles) | Numba CUDA / Python | Starts with map, zip, guards, 2D map, blocks, shared memory, pooling, dot product, and reductions. Good for isolating `threadIdx`, `blockIdx`, guards, and local indexing. | It is not raw CUDA C++, so I still need to translate the mental model back into `.cu` files. |
| [OLCF CUDA Training Series](https://github.com/olcf/cuda-training-series) and [course page](https://www.olcf.ornl.gov/cuda-training-series/) | CUDA C++ | The course page describes a 13-part CUDA series with example exercises after each lecture. Good for host/device memory, kernel launches, reductions, and profiling habits. | Some material assumes an HPC environment, but the concepts transfer. |
| [Tensara problems](https://tensara.org/problems) | CUDA C++, Triton, Mojo, PyPTX, CuTe DSL, cuTile | The problem list showed 84 tasks during this search, including vector add, ReLU, RMSNorm, reductions, pooling, softmax, layernorm, matmul, convolution, and attention. | It is more judge/leaderboard-like than textbook-like. Do correctness first, then performance. |
| [LeetGPU](https://leetgpu.com/) | CUDA C++ in browser | Useful for browser-based CUDA practice and challenge-style repetition. Search results and secondary coverage describe easy/medium/hard challenges and an online playground. | Contest framing can pull attention toward speed before the basics are solid. |
| [CUDA Online Judge / CudaForces](https://cudaforces.com/) and [Codeforces announcement](https://codeforces.com/blog/entry/149751) | CUDA C/C++ with CPU-emulated execution | The announcement describes a browser practice environment and a step-by-step curriculum from basic syntax to shared memory and atomics. | CPU emulation is good for syntax and logic, but real GPU performance conclusions need real GPU runs. |

My default next move: GPU Puzzles for indexing drills, then CUDA C++ versions of the same ideas inside this repo.

---

## 3. ML-Kernel Practice Track

These resources connect CUDA fundamentals to ML operators.

| Resource | Stack | Kernel themes | When to use |
| --- | --- | --- | --- |
| [Tensara problems](https://tensara.org/problems) | CUDA C++ and several GPU DSLs | Activations, reductions, normalization, pooling, softmax, layernorm, matmul, convolution, attention, quantization | Start with easy vector/activation/reduction tasks after PMPP chapter 3. Save softmax/layernorm for reductions. |
| [GPU Mode lectures](https://github.com/gpu-mode/lectures) | CUDA, PyTorch, Triton | The repo lists Lecture 2 as a PMPP chapters 1-3 recap, Lecture 3 as CUDA getting-started material, Lecture 8 as a CUDA performance checklist, and Lecture 9 as reductions. | Use as a companion when a PMPP chapter feels too abstract. |
| [GPU Mode reference kernels](https://github.com/gpu-mode/reference-kernels) | CUDA / Triton | Competition-style reference kernels and problem sets | Use after simple correctness is routine; this is more competitive than tutorial-like. |
| [Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles) | Triton / Python | Starts from first principles and builds toward FlashAttention and quantized neural-network kernels. | Use after CUDA basics, or when I want to compare CUDA thinking with Triton thinking. |
| [Official Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | Triton / PyTorch | Fused softmax, bandwidth-bound operations, reductions | Good after I understand row-wise reductions in CUDA. |
| [CMU 11-868 LLM Systems HW3](https://llmsystem.github.io/llmsystem2024spring/assets/files/11868_LLM_Systems_Assignment_3-6124c724006d6b286d91809ef238c40d.pdf) and [starter repo](https://github.com/llmsystem/llmsys_s24_hw3) | CUDA C++ / miniTorch | Softmax forward/backward and LayerNorm forward/backward with tests and expected speedup targets in the assignment writeup | Use after reductions, shared memory, warp-level primitives, and PyTorch extension basics. |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | PyTorch reference tasks -> CUDA / DSL kernels | Correct and efficient kernels for PyTorch programs | Later. Useful as an ML operator task bank, but too broad for PMPP chapter 3. |
| [llm.c](https://github.com/karpathy/llm.c) | C / CUDA | GPT-style training in C/CUDA, including CUDA training and profiling files | Study later as a real codebase, not as a first exercise source. |
| [TritonBench](https://github.com/thunlp/TritonBench) | Triton | Benchmark data and evaluation channels for Triton operators | Later-stage benchmark material, not a beginner drill set. |

---

## 4. Academic And Course-Style Practice

These are useful when I want assignment-sized practice rather than tiny drills.

| Resource | Stack | What it covers | Fit |
| --- | --- | --- | --- |
| [Purdue ECE 60827 CUDA1](https://github.com/purdue-aalp/ECE60827-CUDA1) | CUDA C/CMake | The public template introduces CUDA programming through SAXPY and Monte Carlo pi; it includes an `autograder` directory in the repo listing. | Good early CUDA C++ practice. |
| [Stanford CS149 Assignment 3](https://github.com/stanford-cs149/asst3) | CUDA C++ | The assignment repo includes `saxpy`, `scan`, and `render`; the handout starts with a CUDA SAXPY warmup and timing discussion. | Good once I want a larger assignment with timing and synchronization details. |
| [CMU 15-418/15-618 Assignment 2 starter](https://github.com/cmu15418s23/asst2) | CUDA C++ | Starter repo lists `saxpy`, `scan`, and `render`. | Similar shape to Stanford CS149; useful but not a small drill set. |
| [MIT 6.S894 Accelerated Computing](https://accelerated-computing.academy/fall25/) and [labs](https://accelerated-computing.academy/fall25/labs/) | CUDA and accelerator programming | The course page says weekly programming assignments focus on GPUs. The lab list includes massively-parallel Mandelbrot, wave simulation, matrix multiply tiling/reuse, tensor cores, run-length compression, and H100 matrix multiplication. | Strong later path for performance engineering. |
| [Aalto / University of Helsinki Programming Parallel Computers](https://ppc.cs.aalto.fi/) and [exercise list](https://ppc-exercises.cs.aalto.fi/en/course/hy2025) | C++ / CUDA | The open course material has exercises, submissions, ratings, and task families such as correlated pairs, image segmentation, median filter, sorting, and an LLM task. | Good if I want structured performance tasks with visible difficulty ratings. |
| [UIUC ECE 408 / CS483 course page](https://lumetta.web.engr.illinois.edu/408-Sum25/) | CUDA C++ | PMPP-aligned applied parallel programming. Public student mirrors commonly show vector add, matmul, convolution, reductions, histogram, scan, and SpMV-style topics. | Use as a topic map. Avoid solution repos when doing active exercises. |
| [University of Luxembourg GPU Programming course](https://github.com/ptal/gpu-programming-uni.lu) | CUDA / C / C++ | The repo contains introduction CUDA material, scan, matrix computation, CUDA streams/histogram, and memory-transaction exercises. It also separates demos, exercises, and solutions. | Good, but solutions are nearby, so avoid spoiling exercises. |
| [ETH-CSCS GPU Training](https://github.com/eth-cscs/gpu-training) | CUDA C++, OpenACC, Fortran | The repo topics include GPU memory/API, writing kernels, scaling with thread blocks, shared memory examples, stencils, dot product, and concurrent code. | Useful for basic CUDA muscle memory; some setup assumes CSCS systems. |
| [Colorado State CS475 assignments](https://www.cs.colostate.edu/~cs475/f16/home_assignments.php) | Parallel programming / CUDA | The assignment page lists labs including Introduction to GPU and CUDA, plotting/analyzing data, and wavefront parallelization. | Older but useful as an assignment index. |
| [UC Riverside CS/EE 217](https://www.danielwong.org/classes/csee217-f16) | CUDA / C++ | The course description covers CUDA memory/threading models and data-parallel programming patterns, with PMPP as the primary textbook. | Good PMPP-adjacent reference, though some lab details may be behind course systems. |

---

## 5. Matmul And Performance Stretch Track

This is the path after naive matrix multiplication works and I can explain the memory traffic.

| Resource | Stack | What to learn |
| --- | --- | --- |
| [Simon Boehm SGEMM worklog](https://siboehm.com/articles/22/CUDA-MMM) | CUDA C++ | Naive matmul, global-memory coalescing, shared-memory cache blocking, tiling, bank conflicts, autotuning, and warp tiling. |
| [SGEMM_CUDA repo](https://github.com/siboehm/SGEMM_CUDA) | CUDA C++ | Step-by-step kernels and benchmarking setup corresponding to the worklog. |
| [Cuda-Learn-By-Practice](https://github.com/Fridge003/Cuda-Learn-By-Practice) | CUDA C++ | Reported by the search agents as a practical repo with GEMM and reduction variants. Recheck before relying on details. |
| [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples) and [CUDA Samples docs](https://docs.nvidia.com/cuda/archive/11.5.0/cuda-samples/index.html) | CUDA C++ | Official examples. The docs say the samples are educational and not meant for performance measurements. |

Use this section only after correctness checks are boring. Performance work should follow the benchmark protocol in `tutor_harness.md`: input size, GPU, CUDA version when known, compiler flags, warmups, measured runs, baseline, and metric.

---

## 6. Inspiration And Long-Arc Curricula

These are useful for planning and motivation, but I should not treat them as exercise judges.

| Resource | Role |
| --- | --- |
| [CUDA 120 Days Challenge](https://github.com/AdepojuJeremy/CUDA-120-DAYS--CHALLENGE) | Day-by-day CUDA curriculum and capstone-style path. Good checklist material; bring my own tests. |
| [AkashKarnatak 100 days of CUDA](https://github.com/AkashKarnatak/100-days-of-cuda/) | Public learning log from vector add and PMPP chapters into softmax, layernorm, FlashAttention, scans, histograms, GEMM, and quantization. Good for seeing one possible arc. |
| [a-hamdi GPU 100 days](https://github.com/a-hamdi/GPU) | Another public learning log around PMPP and kernels. Use for inspiration, not answers. |
| [Infatoshi cuda-course](https://github.com/Infatoshi/cuda-course) | Broad CUDA course repo with sections on first kernels, APIs, faster matmul, Triton, PyTorch extensions, and a final project. |
| [Mojo GPU Puzzles](https://puzzles.modular.com/) | Puzzle-based GPU programming in Mojo. The site frames the course around incremental challenges and GPU mental models. Good later if I want another language's view of the same ideas. |
| [Learn CUDA in an Afternoon practical](https://agray3.github.io/files/learnCUDApractical.pdf) | Short practical worksheet with template and solution folders. Useful for a compact review day. |
| [UL HPC CUDA tutorial](https://ulhpc-tutorials.readthedocs.io/en/latest/cuda/) | CUDA C/C++ tutorial based on NVIDIA DLI material. Useful for refreshers on managed memory, grid-stride loops, bounds checks, and shared memory. |

---

## 7. Suggested PMPP Chapter 3 Practice Path

This is the current plan I want the tutor to push me through.

1. Re-derive 1D global indexing by hand.
   - Exercises: vector add, SAXPY, ReLU, scalar multiply.
   - Checks: odd sizes, `N < blockDim.x`, `N` not divisible by block size.

2. Re-derive 2D indexing by hand.
   - Exercises: grayscale image, 2D map, matrix transpose without shared memory.
   - Checks: non-square shapes, width not divisible by tile size, row-major flattening.

3. Compare 1D block + 2D grid versus 2D block + 2D grid.
   - Exercises: matrix elementwise add, image threshold, box blur.
   - Checks: same output from both mappings, explain which mapping is easier to read.

4. Start matrix multiplication gently.
   - Exercises: one thread per output cell, then a deliberately bad mapping, then a coalescing-aware mapping.
   - Checks: tiny `2x3 @ 3x2`, square `16x16`, non-multiple sizes.

5. Touch reductions only after the above is stable.
   - Exercises: row sum, max over row, simple softmax denominator.
   - Checks: CPU reference, small rows first, then longer rows.

6. Move to ML operators.
   - Easy: ReLU, sigmoid, GELU approximation, vector add, scalar multiply.
   - Medium: RMSNorm, softmax, LayerNorm, matrix-vector multiply.
   - Later: 2D convolution, tiled matmul, attention.

The first external sequence I should try is:

1. GPU Puzzles: map, zip, guards, 2D map, broadcast, blocks.
2. Translate the same ideas into CUDA C++ scratch files in `pmpp/`.
3. Tensara: vector addition, ReLU, sigmoid, RMS normalization, sum/max over dimension.
4. OLCF exercises for host/device memory and reductions.
5. PMPP chapter-specific drills created in this repo when the outside exercises jump too far.

---

## 8. What Counts As Done

For a practice exercise, "done" means:

- CPU reference exists or the expected output is small enough to inspect.
- Kernel handles sizes not divisible by block size.
- Errors are checked or at least surfaced during the run.
- I can explain the launch geometry in one paragraph.
- If I make a performance claim, I record the benchmark conditions.

The artifact format should stay simple:

```text
Implemented: <kernel and launch geometry>
Checked: <CPU reference / tiny input / judge result>
Measured: <kernel time or not measured>
Learned: <one indexing or memory lesson>
Next: <one follow-up>
```
