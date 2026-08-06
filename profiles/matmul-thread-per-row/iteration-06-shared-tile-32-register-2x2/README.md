# Iteration 06: shared-tile-32-register-2x2

## 1. One change

Used a 32 x 32 shared-memory tile while each thread computes a 2 x 2 output tile; block size remains 16 x 16.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2` |
| Workload | 1024 x 1024 x 1024 FP32 matrix multiply |
| Launch geometry | grid `(32, 32, 1)`, block `(16, 16, 1)` |
| Timed launches per process sample | 10 |
| Warm-up policy | no warm-up before timing |
| Source SHA-256 | `73ca8d3b6d47dac0bc2deabf3ccf1cd5fb0eccceaa5d99e1d7b87b64f2daf19b` |
| Helper SHA-256 | `01c0e16734b3fbd08dafa6db9b69f75636ffe2f2a8b5d598996ee29dc2341da3` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Average kernel time (ms) |
|---:|---:|
| 1 | 5.528 |
| 2 | 0.481 |
| 3 | 0.475 |
| 4 | 0.482 |
| 5 | 0.480 |
| **Median** | **0.481** |
| Mean | 1.489 |
| Sample standard deviation | 2.258 |
| Derived kernel-only throughput | 4464.623 GFLOP/s |
| Speedup vs iteration 00 median | 108.626x |

CUDA events measure only the ten kernel launches: allocation, transfers, host initialization, validation, and printing are outside the interval. All five runs passed the four-position CPU sample validation and printed the same first/last output values.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-06-shared-tile-32-register-2x2 \
  -c 'nvcc -O2 /mnt/iteration-06-shared-tile-32-register-2x2/matmul_thread_per_row.cu -I/mnt/iteration-06-shared-tile-32-register-2x2 -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`. The GFLOP/s figure uses `2 * 1024^3` operations divided by the median; it is not end-to-end throughput.
