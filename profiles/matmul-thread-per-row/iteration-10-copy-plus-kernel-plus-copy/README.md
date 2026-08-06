# Iteration 10: copy plus kernel plus copy timing

## 1. One change

This iteration adds a second CUDA-event metric. It measures ten repetitions of M host-to-device copy, N host-to-device copy, kernel launch, and output device-to-host copy. Device allocations and host-side setup remain outside both measurements. One warm-up copy and kernel launch occurs before either timer starts.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2` |
| Workload | 1024 x 1024 x 1024 FP32 matrix multiply |
| Kernel geometry | grid `(32, 32, 1)`, block `(16, 16, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `e0663930661003128dd233a7adcf8635687ed466b2bbf9be59c061cecd67a688` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.430 | 2.704 |
| 2 | 0.427 | 1.661 |
| 3 | 0.427 | 1.680 |
| 4 | 0.426 | 1.681 |
| 5 | 0.427 | 1.973 |
| **Median** | **0.427** | **1.681** |
| Mean | 0.427 | 1.940 |
| Sample standard deviation | 0.002 | 0.446 |

All five runs passed the four-position CPU sample validation. The copy-inclusive number is GPU-stream elapsed time, not host wall-clock latency, and it excludes device allocation.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-10-copy-plus-kernel-plus-copy \
  -c 'nvcc -O2 /mnt/iteration-10-copy-plus-kernel-plus-copy/matmul_thread_per_row.cu -I/mnt/iteration-10-copy-plus-kernel-plus-copy -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
