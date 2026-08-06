# Iteration 21: TF32 two warps across columns

## 1. One change

Mapped the two WMMA warps in a 64-thread block to adjacent output columns instead of adjacent rows.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(32, 64, 1)`, block `(64, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `5857ea05a8bb35791bc3dca77002f83c7227930dd883d84267158c043a008513` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.194 | 1.162 |
| 2 | 0.193 | 1.164 |
| 3 | 0.193 | 1.160 |
| 4 | 0.193 | 1.163 |
| 5 | 0.193 | 1.163 |
| **Median** | **0.193** | **1.163** |
| Mean | 0.193 | 1.162 |
| Sample standard deviation | 0.000 | 0.002 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-21-tf32-two-warps-across-columns \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-21-tf32-two-warps-across-columns/matmul_thread_per_row.cu -I/mnt/iteration-21-tf32-two-warps-across-columns -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
