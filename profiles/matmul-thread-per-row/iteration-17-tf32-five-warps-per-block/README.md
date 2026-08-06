# Iteration 17: TF32 five warps per block

## 1. One change

Tested five independent WMMA warps in each 160-thread block.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(64, 13, 1)`, block `(160, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `c1cc85b3ad22603dd6ad58026eba5adaa3ba8d261b96291aeec24b01c8fc2513` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.252 | 1.212 |
| 2 | 0.245 | 1.207 |
| 3 | 0.249 | 1.209 |
| 4 | 0.254 | 1.208 |
| 5 | 0.245 | 1.205 |
| **Median** | **0.249** | **1.208** |
| Mean | 0.249 | 1.208 |
| Sample standard deviation | 0.004 | 0.003 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-17-tf32-five-warps-per-block \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-17-tf32-five-warps-per-block/matmul_thread_per_row.cu -I/mnt/iteration-17-tf32-five-warps-per-block -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
