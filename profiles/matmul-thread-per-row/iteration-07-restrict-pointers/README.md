# Iteration 07: restrict-pointers

## 1. One change

Added `__restrict__` qualifiers to the three kernel pointer parameters.

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
| Source SHA-256 | `1beb0207d48b9fca1cbeddbfea2e798fe22835d070d31505c6fefaa936a568fe` |
| Helper SHA-256 | `01c0e16734b3fbd08dafa6db9b69f75636ffe2f2a8b5d598996ee29dc2341da3` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Average kernel time (ms) |
|---:|---:|
| 1 | 4.557 |
| 2 | 0.526 |
| 3 | 0.523 |
| 4 | 0.523 |
| 5 | 0.665 |
| **Median** | **0.526** |
| Mean | 1.359 |
| Sample standard deviation | 1.789 |
| Derived kernel-only throughput | 4082.669 GFLOP/s |
| Speedup vs iteration 00 median | 99.333x |

CUDA events measure only the ten kernel launches: allocation, transfers, host initialization, validation, and printing are outside the interval. All five runs passed the four-position CPU sample validation and printed the same first/last output values.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-07-restrict-pointers \
  -c 'nvcc -O2 /mnt/iteration-07-restrict-pointers/matmul_thread_per_row.cu -I/mnt/iteration-07-restrict-pointers -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`. The GFLOP/s figure uses `2 * 1024^3` operations divided by the median; it is not end-to-end throughput.
