# Iteration 16: TF32 three warps per block

## 1. One change

Tested three independent WMMA warps in each 96-thread block.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(64, 22, 1)`, block `(96, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `89884c0fe6937a141a8e73f9d0f902f8ea64ec0d34ad351c5bab4815f2034673` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.257 | 0.774 |
| 2 | 0.254 | 0.765 |
| 3 | 0.252 | 0.766 |
| 4 | 0.257 | 0.764 |
| 5 | 0.255 | 0.766 |
| **Median** | **0.255** | **0.766** |
| Mean | 0.255 | 0.767 |
| Sample standard deviation | 0.002 | 0.004 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-16-tf32-three-warps-per-block \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-16-tf32-three-warps-per-block/matmul_thread_per_row.cu -I/mnt/iteration-16-tf32-three-warps-per-block -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
