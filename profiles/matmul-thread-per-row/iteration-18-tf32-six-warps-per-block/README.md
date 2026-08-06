# Iteration 18: TF32 six warps per block

## 1. One change

Tested six independent WMMA warps in each 192-thread block.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(64, 11, 1)`, block `(192, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `ca3fd96b41a39de513714f6da2b3bcfef8f07ca9382ce5936df2e28b744c9dda` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.287 | 0.785 |
| 2 | 0.284 | 0.799 |
| 3 | 0.280 | 0.787 |
| 4 | 0.275 | 0.784 |
| 5 | 0.278 | 0.788 |
| **Median** | **0.280** | **0.787** |
| Mean | 0.281 | 0.789 |
| Sample standard deviation | 0.005 | 0.006 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-18-tf32-six-warps-per-block \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-18-tf32-six-warps-per-block/matmul_thread_per_row.cu -I/mnt/iteration-18-tf32-six-warps-per-block -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
