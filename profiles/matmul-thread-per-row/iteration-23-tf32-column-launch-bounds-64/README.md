# Iteration 23: TF32 column mapping with launch bounds

## 1. One change

Added `__launch_bounds__(64)` to the two-warp, adjacent-column mapping.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(32, 64, 1)`, block `(64, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `359ecad84577cd7852319e262caa72f276f84594e208b6429c69bfb0c75aca3b` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.246 | 0.754 |
| 2 | 0.239 | 0.745 |
| 3 | 0.238 | 0.742 |
| 4 | 0.237 | 0.746 |
| 5 | 0.237 | 0.744 |
| **Median** | **0.238** | **0.745** |
| Mean | 0.239 | 0.746 |
| Sample standard deviation | 0.004 | 0.005 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-23-tf32-column-launch-bounds-64 \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-23-tf32-column-launch-bounds-64/matmul_thread_per_row.cu -I/mnt/iteration-23-tf32-column-launch-bounds-64 -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
