# Iteration 09: warmup-before-timing

## 1. One change

Added one completed kernel warm-up launch before recording CUDA events. The timed kernel is otherwise iteration 06.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2` |
| Workload | 1024 x 1024 x 1024 FP32 matrix multiply |
| Launch geometry | grid `(32, 32, 1)`, block `(16, 16, 1)` |
| Timed launches per process sample | 10 |
| Warm-up policy | one completed warm-up launch before timing |
| Source SHA-256 | `b2f8022810afab3d27a415f6ceb565bcc22c9e43a5df1137314ca7f4d6d46dbc` |
| Helper SHA-256 | `01c0e16734b3fbd08dafa6db9b69f75636ffe2f2a8b5d598996ee29dc2341da3` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Average kernel time (ms) |
|---:|---:|
| 1 | 0.430 |
| 2 | 0.427 |
| 3 | 0.426 |
| 4 | 0.427 |
| 5 | 0.427 |
| **Median** | **0.427** |
| Mean | 0.427 |
| Sample standard deviation | 0.002 |
| Derived kernel-only throughput | 5029.236 GFLOP/s |
| Speedup vs iteration 00 median | 122.363x |

CUDA events measure only the ten kernel launches: allocation, transfers, host initialization, validation, and printing are outside the interval. All five runs passed the four-position CPU sample validation and printed the same first/last output values.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-09-warmup-before-timing \
  -c 'nvcc -O2 /mnt/iteration-09-warmup-before-timing/matmul_thread_per_row.cu -I/mnt/iteration-09-warmup-before-timing -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`. The GFLOP/s figure uses `2 * 1024^3` operations divided by the median; it is not end-to-end throughput.
