# Iteration 02: thread-per-element-16

## 1. One change

Thread-per-output-element mapping with a 16 x 16 block. This replaces iteration 01's one-thread-per-row kernel.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2` |
| Workload | 1024 x 1024 x 1024 FP32 matrix multiply |
| Launch geometry | grid `(64, 64, 1)`, block `(16, 16, 1)` |
| Timed launches per process sample | 10 |
| Warm-up policy | no warm-up before timing |
| Source SHA-256 | `e4f332a79ffb961b4338f2026e2f505c000509a24a95878f0273c959e458b9be` |
| Helper SHA-256 | `01c0e16734b3fbd08dafa6db9b69f75636ffe2f2a8b5d598996ee29dc2341da3` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Average kernel time (ms) |
|---:|---:|
| 1 | 3.975 |
| 2 | 1.070 |
| 3 | 1.064 |
| 4 | 1.066 |
| 5 | 1.063 |
| **Median** | **1.066** |
| Mean | 1.648 |
| Sample standard deviation | 1.301 |
| Derived kernel-only throughput | 2014.525 GFLOP/s |
| Speedup vs iteration 00 median | 49.014x |

CUDA events measure only the ten kernel launches: allocation, transfers, host initialization, validation, and printing are outside the interval. All five runs passed the four-position CPU sample validation and printed the same first/last output values.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-02-thread-per-element-16 \
  -c 'nvcc -O2 /mnt/iteration-02-thread-per-element-16/matmul_thread_per_row.cu -I/mnt/iteration-02-thread-per-element-16 -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`. The GFLOP/s figure uses `2 * 1024^3` operations divided by the median; it is not end-to-end throughput.
