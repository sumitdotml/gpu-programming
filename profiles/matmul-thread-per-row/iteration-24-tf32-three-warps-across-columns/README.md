# Iteration 24: TF32 three warps across columns

## 1. One change

Tested three WMMA warps computing adjacent column tiles within a 96-thread block.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(22, 64, 1)`, block `(96, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `cf2082a6d3935ca2f938201286b4af28a2c20997c505ced5588cd5abd525b0c1` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.342 | 1.300 |
| 2 | 0.338 | 1.302 |
| 3 | 0.338 | 1.328 |
| 4 | 0.337 | 1.297 |
| 5 | 0.337 | 1.311 |
| **Median** | **0.338** | **1.302** |
| Mean | 0.338 | 1.308 |
| Sample standard deviation | 0.002 | 0.013 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-24-tf32-three-warps-across-columns \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-24-tf32-three-warps-across-columns/matmul_thread_per_row.cu -I/mnt/iteration-24-tf32-three-warps-across-columns -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
