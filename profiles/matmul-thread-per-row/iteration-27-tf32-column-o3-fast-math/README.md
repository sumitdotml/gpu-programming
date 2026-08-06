# Iteration 27: TF32 column mapping with fast math

## 1. One change

Added `--use_fast_math` to the `-O3` build of the two-warp adjacent-column mapping.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O3 --use_fast_math -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(32, 64, 1)`, block `(64, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `5857ea05a8bb35791bc3dca77002f83c7227930dd883d84267158c043a008513` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.235 | 0.760 |
| 2 | 0.232 | 0.745 |
| 3 | 0.234 | 0.744 |
| 4 | 0.233 | 0.740 |
| 5 | 0.232 | 0.760 |
| **Median** | **0.233** | **0.745** |
| Mean | 0.233 | 0.750 |
| Sample standard deviation | 0.001 | 0.009 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-27-tf32-column-o3-fast-math \
  -c 'nvcc -O3 --use_fast_math -arch=sm_86 /mnt/iteration-27-tf32-column-o3-fast-math/matmul_thread_per_row.cu -I/mnt/iteration-27-tf32-column-o3-fast-math -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
