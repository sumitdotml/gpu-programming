# A10G profile attempt - 2026-08-06

## 1. Scope

- Target: `pmpp/matmul_thread_per_row.cu`
- Revision: `9af501f4b65fd04dd421d23263a776dcceb9219b`
- Source SHA-256: `0182d055d4ac41ec4c2df4dea83130d4fe1c8bbf02075020feab62aedffdca1b`
- Helper SHA-256: `01c0e16734b3fbd08dafa6db9b69f75636ffe2f2a8b5d598996ee29dc2341da3`
- Source changes: none.

## 2. Successful execution

| Field | Observed value |
|---|---:|
| Modal GPU request | `A10G` |
| Device reported by the program | NVIDIA A10 |
| Compute capability | 8.6 |
| Driver version | 580.95.05 |
| CUDA version reported by `nvidia-smi` | 13.0 |
| Compilation | `nvcc -O2` |
| Container image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Output shape | 5 x 4 |
| Kernel launch geometry | grid `(1, 2, 1)`, block `(1, 4, 1)` |
| Threads launched | 8 |
| Active output rows | 5 |

The program completed successfully and printed the expected 5 x 4 result matrix recorded in the terminal run.

Reproduce the successful execution (the `--no-pty` flag is required in this non-interactive session):

```sh
modal shell --no-pty --gpu A10G --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --add-local pmpp/matmul_thread_per_row.cu --add-local pmpp/cuda_helpers.h \
  -c 'nvcc -O2 /mnt/matmul_thread_per_row.cu -o /tmp/matmul_thread_per_row.out && /tmp/matmul_thread_per_row.out'
```

## 3. Kernel profiling result

No kernel-duration, occupancy, throughput, or memory-traffic number was obtained. They must not be inferred from this run.

Two collection attempts were made against the same source with the CUDA 13.0 development image:

| Tool | Version | Result |
|---|---|---|
| Nsight Compute (`ncu --set basic`) | 2025.3.0.0 | Failed before collection: `LibraryNotLoaded` from the counter measurement library. |
| Nsight Systems (`nsys profile --trace=cuda`) | 2026.1.3.425 | Failed while converting GPU timestamps with `InternalErrorException`; no `.nsys-rep` was produced. |

The program still executed and produced its output under both profiler wrappers, but neither tool emitted a usable profile report. The command-line exit status was zero despite the profiler errors, so success was determined from the tool diagnostics and the absent report rather than that status.

## 4. Interpretation limits

This is a correctness run plus a failed profiler-collection record, not a performance benchmark. The default problem performs only 60 fused multiply-add-style dot-product iterations across 20 outputs and launches eight threads, so even a successful one-launch timing would not characterize sustained matrix-multiplication performance. That operation count and launch size are derived from the checked source; the wording about performance characterization is an engineering interpretation.

## 5. Required code

None was added or copied here: the existing target and helper are the complete executable inputs, identified by the checksums above. The shell commands in this document are sufficient to reproduce the execution and the two profiler attempts.
