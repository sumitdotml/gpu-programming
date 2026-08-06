# Iteration 19: TF32 seven warps per block

## 1. One change

Completed the warp-count sweep with seven independent WMMA warps in each block.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(64, 10, 1)`, block `(224, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `2d02969107f8c3e349a090c96ea7269af2003871a87a9dd9f6aa31fadfb1ea6b` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.282 | 0.786 |
| 2 | 0.277 | 0.784 |
| 3 | 0.277 | 0.782 |
| 4 | 0.273 | 0.776 |
| 5 | 0.275 | 0.783 |
| **Median** | **0.277** | **0.783** |
| Mean | 0.277 | 0.782 |
| Sample standard deviation | 0.003 | 0.004 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-19-tf32-seven-warps-per-block \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-19-tf32-seven-warps-per-block/matmul_thread_per_row.cu -I/mnt/iteration-19-tf32-seven-warps-per-block -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
