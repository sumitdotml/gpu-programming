# Iteration 28: TF32 column mapping with PTXAS O3

## 1. One change

Added `-Xptxas -O3` to the `-O3` build of the two-warp adjacent-column mapping.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O3 -Xptxas -O3 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply using TF32 WMMA |
| Kernel geometry | grid `(32, 64, 1)`, block `(64, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `5857ea05a8bb35791bc3dca77002f83c7227930dd883d84267158c043a008513` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.236 | 0.744 |
| 2 | 0.232 | 0.742 |
| 3 | 0.231 | 0.742 |
| 4 | 0.231 | 0.742 |
| 5 | 0.230 | 0.745 |
| **Median** | **0.231** | **0.742** |
| Mean | 0.232 | 0.743 |
| Sample standard deviation | 0.002 | 0.001 |

All five runs passed the four-position CPU sample validation with its `1e-4` relative tolerance. Allocation and host-side setup are excluded.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-28-tf32-column-ptxas-o3 \
  -c 'nvcc -O3 -Xptxas -O3 -arch=sm_86 /mnt/iteration-28-tf32-column-ptxas-o3/matmul_thread_per_row.cu -I/mnt/iteration-28-tf32-column-ptxas-o3 -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
