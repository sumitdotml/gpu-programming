# Iteration 00: 1024-cubed baseline

## 1. Purpose and target

This is the baseline for a tentative 100x throughput-improvement target: reduce the median kernel time from 52.249 ms to at most 0.52249 ms while preserving the same 1024 x 1024 x 1024 computation and validating output. The target is a planning target, not a result.

## 2. Exact inputs

| Item | Value |
|---|---|
| GPU request | Modal `A10G` |
| Device string printed in five measured runs | NVIDIA A10G |
| Compute capability printed in five measured runs | 8.6 |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Compiler command | `nvcc -O2` |
| Matrix shapes | M: 1024 x 1024; N: 1024 x 1024; output: 1024 x 1024 |
| Kernel mapping | one thread computes one output row |
| Grid | `(1, 256, 1)` |
| Block | `(1, 4, 1)` |
| Threads launched | 1024 |
| Timed launches per sample | 10 |
| Source SHA-256 | `5387bd4c1256d84c2c80dd8d2835cfe53d4b4114bdd36fb3c152aad562a5a82e` |
| Helper SHA-256 | `01c0e16734b3fbd08dafa6db9b69f75636ffe2f2a8b5d598996ee29dc2341da3` |

`matmul_thread_per_row.cu` and `cuda_helpers.h` in this directory are the complete build inputs for this measurement. Compared with the repository's original example, this copy changes dimensions to 1024, replaces full-matrix printing with two output samples, and adds CUDA-event timing around ten kernel launches.

## 3. Measurements

CUDA events bracket the ten kernel launches only. Allocation, host-device copies, initialization, and output printing are excluded.

| Process run | Average kernel time (ms) |
|---:|---:|
| 1 | 70.177 |
| 2 | 52.273 |
| 3 | 52.249 |
| 4 | 52.225 |
| 5 | 52.129 |
| **Median (all five)** | **52.249** |
| Mean (all five) | 55.811 |
| Sample standard deviation (all five) | 8.031 |
| Mean (runs 2–5) | 52.219 |
| Sample standard deviation (runs 2–5) | 0.063 |

The first measurement is slower than the other four; it is retained rather than discarded. The stable-run subset is reported separately, not substituted for the all-run headline.

The mathematical workload is `2 * 1024^3 = 2,147,483,648` floating-point operations under the usual multiply-plus-add convention. Dividing that by the all-run median gives **41.101 GFLOP/s**. This is a derived kernel-only figure, not end-to-end application throughput.

All five runs printed `first=360014880.000000` and `last=2508542976.000000`. This checks repeatability of two output positions only; this iteration does not include a CPU reference comparison.

## 4. Reproduction

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local profiles/matmul-thread-per-row/iteration-00-baseline-1024 \
  -c 'nvcc -O2 /mnt/iteration-00-baseline-1024/matmul_thread_per_row.cu -I/mnt/iteration-00-baseline-1024 -o /tmp/matmul_thread_per_row.out && for run in 1 2 3 4 5; do echo "RUN=$run"; /tmp/matmul_thread_per_row.out; done'
```

The unedited terminal output is retained in `a10g-five-runs.log`.

## 5. Limits

This timing harness is required because the prior Nsight Compute and Nsight Systems collection attempts did not produce usable reports in this Modal environment. CUDA-event timing cannot provide occupancy, memory-traffic, or instruction-level metrics. Later iterations must compare the same timing scope and add stronger correctness validation before treating an optimization as valid.
