# Iteration 15: TF32 two warps per block

## 1. One change

Reduced the TF32 WMMA block from four warps to two independent warps.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(64, 32, 1)`, block `(64, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `0631bcac98953458b8918dd6c779093845e70cbe08e1c711809b1be0b141bbb9` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.242 | 0.751 |
| 2 | 0.242 | 0.751 |
| 3 | 0.240 | 0.752 |
| 4 | 0.239 | 0.752 |
| 5 | 0.239 | 0.749 |
| **Median** | **0.240** | **0.751** |
| Mean | 0.240 | 0.751 |
| Sample standard deviation | 0.002 | 0.001 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-15-tf32-two-warps-per-block \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-15-tf32-two-warps-per-block/matmul_thread_per_row.cu -I/mnt/iteration-15-tf32-two-warps-per-block -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
