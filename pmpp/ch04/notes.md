# Chapter 4: Compute architecture and scheduling

Short takeaways from our discussions, not a transcript or a list of completed exercises.
Covered so far: Sections 4.1-4.6, registers from Section 4.7, and device queries from Section 4.8. Later topics will be added as we discuss them.

## 4.1 Architecture of a modern GPU

Remember: Threads are work; SMs and CUDA cores are hardware that carries out the work.

- An SM (streaming multiprocessor) contains execution units, control/scheduling machinery, and on-chip storage. The green squares in Figure 4.1 represent streaming processors, not threads.
- In this discussion, streaming processors correspond to CUDA cores; do not picture them as full CPU cores or permanent homes for individual threads.
- GPC is a hardware grouping of SMs. The book calls it a GPU Processing Cluster; NVIDIA uses Graphics Processing Cluster.
- Global memory is the device memory holding our matrices. Memory controllers connect memory requests to that storage; the SM's on-chip storage is a separate part of the diagram.

## 4.2 Thread block scheduling

Remember: A block is a group of work assigned to one SM, not a physical piece of the SM.

- All threads of a block are assigned to the same SM. Multiple blocks can reside on one SM when resources permit.
- Assigned together does not mean all threads execute an instruction simultaneously. Execution resources are shared over time.
- More blocks can wait to be assigned as resident blocks finish. Do not assume a particular block order.
- A hardware "processing block" inside an SM is not a CUDA thread block. Call it an SM subpartition to keep the two ideas separate.
- An optional thread block cluster groups blocks for co-scheduling within one GPC on supporting hardware. It is not the same thing as a GPC or an ordinary block.

## 4.3 Synchronization and transparent scalability

Remember: A barrier coordinates a phase of work within a particular scope; it does not schedule blocks onto hardware.

- `__syncthreads()` is a block-wide barrier, not a grid-wide barrier. Participating threads wait at the same barrier until the required threads arrive; it also provides memory-ordering guarantees within the block.
- Do not put a block barrier on a path taken by only some participating threads. Separate calls in opposing branches are not automatically the same barrier.
- Cluster synchronization has cluster scope; distributed shared memory exposes the participating blocks' shared memory within that cluster. Grid synchronization requires special support and launch restrictions.
- Transparent scalability: independent blocks can be scheduled on GPUs with different amounts of hardware without changing the kernel's basic work description.
- An intrinsic is a function-like operation the compiler recognizes specially, rather than merely compiling a normal supplied function body. `__syncthreads()` is an example. `+` is an operator, not an intrinsic function.

## 4.4 Warps and SIMD hardware

Remember: I write one thread's work; the GPU groups threads into warps and issues common instructions to participating lanes.

- The kernel function describes one thread's recipe. A kernel launch creates many threads following that recipe, each with its own indices and state. In our matmul, each valid thread computes one output element.
- A warp has 32 lane positions on the NVIDIA hardware discussed here. A lane is a logical position, not a CUDA core. A partial warp has unused positions; hardware does not create extra useful threads to fill them.
- Warps are formed within each block and never combine threads from different blocks. Two 16-thread blocks produce two partial warps, not one full warp.
- In a 2D block, x changes fastest: `linear_id = threadIdx.y * blockDim.x + threadIdx.x`. Consecutive groups of 32 linear IDs form warps, even if a group crosses a row boundary.
- SIMT describes individual-thread execution and branching. SIMD describes one instruction operating on multiple data values. SIMT code uses SIMD-like execution machinery; these are not competing labels.
- CUDA cores perform arithmetic; they do not "hold" entire warps. Subpartitions maintain resident warps and use schedulers and execution pipelines to carry out their instructions.

Practice: [Warp drill](warp-drill.md).

## 4.5 Control divergence

Remember: Different data is expected. Different execution paths within one warp cause control divergence.

- An `if` does not automatically diverge: divergence occurs when threads in the same warp choose different paths. Different warps choosing different paths is not within-warp divergence.
- For a divergent branch, each path's instructions take effect only for its participating threads. The other lanes are inactive for those instructions, not deleted.
- Loops can diverge when threads need different iteration counts. Boundary checks can also diverge; they are still necessary for correctness.
- Extra paths and inactive lanes can reduce efficiency, but divergence does not imply an exact 2x slowdown.
- Modern independent thread scheduling can interleave paths. Do not assume warp membership supplies synchronization; use the appropriate barrier when cooperation requires it.
- This section mainly teaches recognition and consequences, not a comprehensive mitigation toolkit. Synchronization ensures correctness; it does not eliminate divergence.

## 4.6 Warp scheduling and latency tolerance

Remember: Many resident warps provide alternatives; schedulers choose ready instructions to keep the hardware busy.

- Resident means the warp is on the SM with its state available. Ready means its next instruction can proceed. Selected means a scheduler chooses that instruction to issue.
- When one warp waits for a load result or another dependency, another ready warp can supply useful work. The original operation is not faster; its waiting time is occupied by other work. That is latency hiding.
- Fine-grained multithreading selects among warps at instruction-issue opportunities, not only after a warp finishes its entire kernel execution. Selection is not necessarily round-robin.
- Execution state means the information needed to resume: instruction positions, register values, and execution-control information. In matmul, a thread's current accumulator and loop progress illustrate the values that must be preserved (their exact compiled storage is compiler-dependent).
- Zero-overhead scheduling: resident warps already have their own allocated register storage and on-chip execution state. Selecting B does not overwrite A's state, so selecting A again needs no save/restore round trip to memory. The arithmetic hardware is shared; the resident state remains available. This refers to switching among resident warps, not GPU-wide context switches or preemption, and does not mean instructions are instantaneous or the SM can never stall.
- Multiple schedulers allow parallel issue, while each scheduler also interleaves work from its assigned warps over time. Hopper has four warp schedulers per SM; this is not a guarantee of four completed warp instructions every cycle.
- More resident warps increase the chance of finding ready work. If all are waiting, the hardware can still idle; available resources limit how many warps can reside on an SM.

## 4.7 Resource partitioning and occupancy (registers so far)

Remember: Registers are the SM's small, fast working storage; resident threads need their own allocated register space.

- A register stores bits used as an instruction's working values, such as operands, intermediate results, and addresses. The compiler generally places small thread-local working values in registers; source variables do not map one-to-one to physical registers.
- In matmul, the accumulator illustrates a value a thread keeps updating while input matrices and the output array remain in global memory. Exact register allocation requires inspecting compiled code; we have not measured our kernel's register usage here.
- H100 has 65,536 32-bit registers per SM: 4 bytes per register, totaling 256 KiB. That is an SM-wide pool, not 65,536 registers per thread or CUDA core, and it is not the shared-memory pool.
- Simplified budget: 2,048 resident threads using 32 registers each require 65,536 registers. Using 64 registers each gives a register-budget ceiling of 1,024 threads. Actual residency also depends on allocation granularity, block sizes, and other resource limits.
- Some values may be placed in thread-private local memory instead (including register spills); declaring a kernel variable does not guarantee register storage. Fewer resident warps can reduce opportunities to hide latency, but occupancy alone is not a performance measurement.

## 4.8 Querying device properties

Remember: Query the GPU actually running the program; the book's H100 numbers are examples, not constants for every device.

- Host code calls `cudaGetDeviceCount()` and `cudaGetDeviceProperties()`; the latter fills a `cudaDeviceProp` struct. Our standalone [device probe](device_query.cu) prints the relevant fields and units without launching a kernel.
- Distinguish per-device, per-SM, and per-block capacities. In particular, `regsPerBlock` is a block limit; `regsPerMultiprocessor` is the SM's register pool.
- `maxThreadsDim` limits each block axis, while `maxThreadsPerBlock` limits their product. A large permitted grid does not imply all its blocks can reside on the GPU simultaneously.
- Device limits do not tell us a particular kernel's actual occupancy or measured performance. Occupancy and maximum cluster-size APIs require a kernel and its resource/launch configuration. The probe reports cluster-launch support, not a kernel's maximum cluster size.
- `clockRate` is a deprecated reported property, not a live clock measurement. The probe includes it to connect to the book and labels that limitation.

Run the same file with different Modal GPU selections:

```sh
make modal SRC=pmpp/ch04/device_query.cu
make modal SRC=pmpp/ch04/device_query.cu GPU=H100
```

Observed in our 2026-10-10 Modal run (default `RTX-PRO-6000`, image `nvidia/cuda:12.8.1-devel-ubuntu24.04`, flags `-O2 -Xcompiler=-Wall,-Wextra`): NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, 188 SMs, 1,536 threads / 48 warps / 24 blocks maximum per SM, and 65,536 32-bit registers per SM. CUDA runtime reported 12.8 and driver-supported CUDA version 13.0. Compilation and execution succeeded; NVCC emitted its standard pre-sm_75 target deprecation warning.

User-reported H100 run on 2026-10-10 (the H100 command above, same image, flags `-O2`): NVIDIA H100 80GB HBM3, compute capability 9.0, 132 SMs, 2,048 threads / 64 warps / 32 blocks maximum per SM, and 65,536 32-bit registers per SM. Shared memory per SM was 228 KiB versus 100 KiB in our RTX PRO 6000 run; opt-in shared memory per block was 227 KiB versus 99 KiB. The H100 output reported 79.18 GiB global memory and 50 MiB L2 cache, CUDA runtime 12.8, and driver-supported CUDA version 13.0. The assistant read the supplied output but did not independently run this probe on H100. More SMs or more resident warps alone do not establish which GPU will run a particular kernel faster.

## Sources and scope

These pointers summarize our explanations using PMPP, fifth edition, Sections 4.1-4.8, read during the discussion. NVIDIA references checked during the discussion support the implementation-specific clarifications:

- [CUDA guide: SIMT and hardware multithreading](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#simt-architecture)
- [CUDA guide: Hopper / compute capability 9.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#compute-capability-9-0)
- [Nsight Compute: SMs and subpartitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#streaming-multiprocessor)
- [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Hopper tuning guide: register capacity and occupancy](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy)
- [CUDA programming model: registers and resource allocation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA 12.8.1 runtime: device-property fields and units](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/structcudaDeviceProp.html)

Architecture examples are not universal hardware guarantees. These notes deliberately do not treat the book's "GPCs started with Hopper" wording as a historical fact.
