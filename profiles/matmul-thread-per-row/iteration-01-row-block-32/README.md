# Iteration 01: 32 threads per row block

## 1. One change from iteration 00

`row_threads_per_block` changed from 4 to 32. The kernel still assigns one thread to each output row and retains the same 1024 x 1024 x 1024 workload, CUDA-event timing harness, and output samples.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Device string printed in measured runs | NVIDIA A10 |
| Compute capability | 8.6 |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler | `nvcc -O2` |
| Grid / block | `(1, 32, 1)` / `(1, 32, 1)` |
| Threads launched | 1024 |
| Timed launches per sample | 10 |
| Source SHA-256 | `56c48da7048e4dfe66a3f24bad5330a38355013d70d1bf8af3fcad1cb3e1eae0` |

The complete build inputs are `matmul_thread_per_row.cu` and `cuda_helpers.h` in this directory.

## 3. Measurements

| Process run | Average kernel time (ms) |
|---:|---:|
| 1 | 65.866 |
| 2 | 63.572 |
| 3 | 63.562 |
| 4 | 63.565 |
| 5 | 63.571 |
| **Median** | **63.571** |
| Mean | 64.027 |
| Sample standard deviation | 1.029 |
| Mean (runs 2–5) | 63.568 |
| Sample standard deviation (runs 2–5) | 0.005 |

The median is 1.217x slower than iteration 00's 52.249 ms median. Under the same `2 * 1024^3` operation convention, it is 33.781 GFLOP/s. All runs printed the same two output samples as iteration 00: `first=360014880.000000`, `last=2508542976.000000`.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-01-row-block-32 \
  -c 'nvcc -O2 /mnt/iteration-01-row-block-32/matmul_thread_per_row.cu -I/mnt/iteration-01-row-block-32 -o /tmp/matmul.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul.out; done'
```

`a10g-five-runs.log` preserves the terminal output.

## 5. Interpretation

This is a negative result: increasing only the row-block size did not improve this kernel. It remains a kernel-only CUDA-event measurement and only checks two output positions, so it does not establish full numerical correctness.
