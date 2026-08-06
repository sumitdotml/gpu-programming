# Iteration 05: shared-tile-32

## 1. One change

Increased the shared-memory tile and block edge from 16 to 32.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2` |
| Workload | 1024 x 1024 x 1024 FP32 matrix multiply |
| Launch geometry | grid `(32, 32, 1)`, block `(32, 32, 1)` |
| Timed launches per process sample | 10 |
| Warm-up policy | no warm-up before timing |
| Source SHA-256 | `fb2c606c8e6c8a47f06c902d373c3ce4c5c24625ff87d2979b035e9819f456f6` |
| Helper SHA-256 | `01c0e16734b3fbd08dafa6db9b69f75636ffe2f2a8b5d598996ee29dc2341da3` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Average kernel time (ms) |
|---:|---:|
| 1 | 3.081 |
| 2 | 1.029 |
| 3 | 1.018 |
| 4 | 1.022 |
| 5 | 1.025 |
| **Median** | **1.025** |
| Mean | 1.435 |
| Sample standard deviation | 0.920 |
| Derived kernel-only throughput | 2095.106 GFLOP/s |
| Speedup vs iteration 00 median | 50.975x |

CUDA events measure only the ten kernel launches: allocation, transfers, host initialization, validation, and printing are outside the interval. All five runs passed the four-position CPU sample validation and printed the same first/last output values.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-05-shared-tile-32 \
  -c 'nvcc -O2 /mnt/iteration-05-shared-tile-32/matmul_thread_per_row.cu -I/mnt/iteration-05-shared-tile-32 -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`. The GFLOP/s figure uses `2 * 1024^3` operations divided by the median; it is not end-to-end throughput.
