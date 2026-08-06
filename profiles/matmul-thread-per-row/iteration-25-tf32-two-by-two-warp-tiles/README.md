# Iteration 25: TF32 two-by-two warp tiles

## 1. One change

Mapped four warps per block to a 2 x 2 output-tile square.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(32, 32, 1)`, block `(128, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `6ed5228a8b39081f810685c0b806498475828a2f2f4a13c121b35e7b96420064` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.266 | 0.765 |
| 2 | 0.272 | 0.770 |
| 3 | 0.261 | 0.764 |
| 4 | 0.271 | 0.761 |
| 5 | 0.264 | 0.765 |
| **Median** | **0.266** | **0.765** |
| Mean | 0.267 | 0.765 |
| Sample standard deviation | 0.005 | 0.003 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-25-tf32-two-by-two-warp-tiles \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-25-tf32-two-by-two-warp-tiles/matmul_thread_per_row.cu -I/mnt/iteration-25-tf32-two-by-two-warp-tiles -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
