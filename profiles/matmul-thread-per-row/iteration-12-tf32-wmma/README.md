# Iteration 12: TF32 WMMA Tensor Cores

## 1. One change

This variant replaces the shared-memory FP32 kernel with one 16 x 16 TF32 WMMA operation per warp. FP32 inputs and outputs remain in memory, while the Tensor Core operation uses TF32. It requires `-arch=sm_86` for this A10G profile.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2 -arch=sm_86` |
| Workload | 1024 x 1024 x 1024 matrix multiply |
| Kernel geometry | grid `(64, 64, 1)`, block `(32, 1, 1)` |
| Timed repetitions per process run | 10 |
| Source SHA-256 | `2c2d3954ed5937277587d8a7b2cbe8267b65a487e1dfcd6e161a2700d105dc3b` |

The complete build inputs are the `.cu` file and `cuda_helpers.h` in this directory.

## 3. CUDA-event measurements

| Process run | Kernel-only average (ms) | Copy-plus-kernel average (ms) |
|---:|---:|---:|
| 1 | 0.358 | 1.317 |
| 2 | 0.358 | 1.321 |
| 3 | 0.359 | 1.330 |
| 4 | 0.359 | 1.318 |
| 5 | 0.358 | 1.315 |
| **Median** | **0.358** | **1.318** |
| Mean | 0.358 | 1.320 |
| Sample standard deviation | 0.001 | 0.006 |

All five runs passed four CPU sample checks using the source's relative tolerance of `1e-4`. The TF32 output samples differ from the FP32 baseline: first output is `360013472.0` instead of `360014880.0`, and last output is `2508516352.0` instead of `2508542976.0`.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-12-tf32-wmma \
  -c 'nvcc -O2 -arch=sm_86 /mnt/iteration-12-tf32-wmma/matmul_thread_per_row.cu -I/mnt/iteration-12-tf32-wmma -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

The unedited terminal output is `a10g-five-runs.log`.
