# Iteration 20: TF32 launch bounds

## 1. One change

Added `__launch_bounds__(64)` to the two-warp TF32 kernel so the compiler knows its maximum block size.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(64, 32, 1)`, block `(64, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `6f607c4c47f1737d65d22b901a8542aa1a9f7bd127bcab0085ee3f029a369faa` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.240 | 0.758 |
| 2 | 0.240 | 0.751 |
| 3 | 0.240 | 0.757 |
| 4 | 0.240 | 0.750 |
| 5 | 0.240 | 0.748 |
| **Median** | **0.240** | **0.751** |
| Mean | 0.240 | 0.753 |
| Sample standard deviation | 0.000 | 0.004 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-20-tf32-launch-bounds-64 \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-20-tf32-launch-bounds-64/matmul_thread_per_row.cu -I/mnt/iteration-20-tf32-launch-bounds-64 -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
