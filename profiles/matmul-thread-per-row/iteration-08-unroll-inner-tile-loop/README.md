# Iteration 08: unroll-inner-tile-loop

## 1. One change

Added `#pragma unroll` to the 32-step inner tile loop.

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
| Source SHA-256 | `94bf0286994c05ba48f07a7707729828de1646afdea4b81161372fbea5c5c37d` |
| Helper SHA-256 | `01c0e16734b3fbd08dafa6db9b69f75636ffe2f2a8b5d598996ee29dc2341da3` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Average kernel time (ms) |
|---:|---:|
| 1 | 4.474 |
| 2 | 0.541 |
| 3 | 0.544 |
| 4 | 0.562 |
| 5 | 0.553 |
| **Median** | **0.553** |
| Mean | 1.335 |
| Sample standard deviation | 1.755 |
| Derived kernel-only throughput | 3883.334 GFLOP/s |
| Speedup vs iteration 00 median | 94.483x |

CUDA events measure only the ten kernel launches: allocation, transfers, host initialization, validation, and printing are outside the interval. All five runs passed the four-position CPU sample validation and printed the same first/last output values.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-08-unroll-inner-tile-loop \
  -c 'nvcc -O2 /mnt/iteration-08-unroll-inner-tile-loop/matmul_thread_per_row.cu -I/mnt/iteration-08-unroll-inner-tile-loop -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`. The GFLOP/s figure uses `2 * 1024^3` operations divided by the median; it is not end-to-end throughput.
