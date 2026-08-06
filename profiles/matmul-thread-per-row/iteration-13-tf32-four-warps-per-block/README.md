# Iteration 13: TF32 four warps per block

## 1. One change

Packed four independent 16 x 16 WMMA tiles into each 128-thread block. Each warp computes one tile.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(64, 16, 1)`, block `(128, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `19005a425eab42b45c1974000bd639661f0074f0c7890760116b33242cf05313` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.270 | 0.783 |
| 2 | 0.272 | 0.783 |
| 3 | 0.271 | 0.775 |
| 4 | 0.271 | 0.777 |
| 5 | 0.267 | 0.781 |
| **Median** | **0.271** | **0.781** |
| Mean | 0.270 | 0.780 |
| Sample standard deviation | 0.002 | 0.004 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-13-tf32-four-warps-per-block \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-13-tf32-four-warps-per-block/matmul_thread_per_row.cu -I/mnt/iteration-13-tf32-four-warps-per-block -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
