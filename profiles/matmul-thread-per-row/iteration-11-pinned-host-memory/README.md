# Iteration 11: pinned host memory

## 1. One change

Replaced pageable host buffers with CUDA pinned host allocations. The end-to-end timer remains the same as iteration 10.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2` |
| Workload | 1024 x 1024 x 1024 FP32 matrix multiply |
| Kernel geometry | grid `(32, 32, 1)`, block `(16, 16, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `7b7f262cbee9904bed85acb058dfc07656c5ff7a07f2d9328e356b891b71f5ee` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.486 | 0.995 |
| 2 | 0.482 | 0.990 |
| 3 | 0.481 | 0.990 |
| 4 | 0.481 | 0.990 |
| 5 | 0.481 | 0.988 |
| **Median** | **0.481** | **0.990** |
| Mean | 0.482 | 0.991 |
| Sample standard deviation | 0.002 | 0.003 |

All five runs passed the four-position CPU sample validation. Allocation and host-side setup are outside both event intervals.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-11-pinned-host-memory \
  -c 'nvcc -O2 /mnt/iteration-11-pinned-host-memory/matmul_thread_per_row.cu -I/mnt/iteration-11-pinned-host-memory -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
