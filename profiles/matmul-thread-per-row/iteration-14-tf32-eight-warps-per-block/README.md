# Iteration 14: TF32 eight warps per block

## 1. One change

Packed eight independent WMMA tiles into each 256-thread block.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(64, 8, 1)`, block `(256, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `6d2b2a7c7e58998463f449a0f31ebb77c004a77251674e1379b62dbc0e716004` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.292 | 0.817 |
| 2 | 0.296 | 0.804 |
| 3 | 0.296 | 0.802 |
| 4 | 0.295 | 0.806 |
| 5 | 0.297 | 0.809 |
| **Median** | **0.296** | **0.806** |
| Mean | 0.295 | 0.808 |
| Sample standard deviation | 0.002 | 0.006 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-14-tf32-eight-warps-per-block \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-14-tf32-eight-warps-per-block/matmul_thread_per_row.cu -I/mnt/iteration-14-tf32-eight-warps-per-block -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
