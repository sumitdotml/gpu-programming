# Iteration 22: TF32 four warps across columns

## 1. One change

Tested four WMMA warps that compute adjacent column tiles within each block.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(16, 64, 1)`, block `(128, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `7f680985823165cea9f05a7d72c250740e36168524dfe8fc41f9228a03109e68` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.268 | 0.792 |
| 2 | 0.267 | 0.775 |
| 3 | 0.267 | 0.776 |
| 4 | 0.267 | 0.778 |
| 5 | 0.267 | 0.785 |
| **Median** | **0.267** | **0.778** |
| Mean | 0.267 | 0.781 |
| Sample standard deviation | 0.000 | 0.007 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-22-tf32-four-warps-across-columns \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-22-tf32-four-warps-across-columns/matmul_thread_per_row.cu -I/mnt/iteration-22-tf32-four-warps-across-columns -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
