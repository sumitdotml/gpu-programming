# Chapter 4: Compute architecture and scheduling

Short takeaways from our discussions, not a transcript or a list of completed exercises.
First reading and all nine exercises completed with tutor guidance on 2026-10-11.
These notes cover our discussions of Sections 4.1-4.8 and the distinctions reinforced by the exercises. Chapter 5 is next; it has not been started yet.

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
- The barrier separates phases, not entire tasks: all participating threads finish the preceding phase before any proceed past the barrier. Being in one warp does not replace a required barrier; `__syncwarp()` can synchronize participating warp threads when that scope is sufficient.
- Do not put a block barrier on a path taken by only some participating threads. Separate calls in opposing branches are not automatically the same barrier.
- Cluster synchronization has cluster scope; distributed shared memory exposes the participating blocks' shared memory within that cluster. Grid synchronization requires special support and launch restrictions.
- Transparent scalability: independent blocks can be scheduled on GPUs with different amounts of hardware without changing the kernel's basic work description.
- An intrinsic is a function-like operation the compiler recognizes specially, rather than merely compiling a normal supplied function body. `__syncthreads()` is an example. `+` is an operator, not an intrinsic function.

## 4.4 Warps and SIMD hardware

Remember: I write one thread's work; the GPU groups threads into warps and issues common instructions to participating lanes.

- The kernel function describes one thread's recipe. A kernel launch creates many threads following that recipe, each with its own indices and state. In our matmul, each valid thread computes one output element.
- A warp has 32 lane positions on the NVIDIA hardware discussed here. A lane is a logical position, not a CUDA core. A partial warp has unused positions; hardware does not create extra useful threads to fill them.
- Warps are formed within each block and never combine threads from different blocks. Two 16-thread blocks produce two partial warps, not one full warp.
- We declare block dimensions, not warps. For this 32-lane model, warps per block are `ceil((blockDim.x * blockDim.y * blockDim.z) / 32)`. Count per block first, then multiply by the number of blocks; rounding the whole grid's thread count can hide partial warps in individual blocks.
- In a 2D block, x changes fastest: `linear_id = threadIdx.y * blockDim.x + threadIdx.x`. Consecutive groups of 32 linear IDs form warps, even if a group crosses a row boundary.
- SIMT describes individual-thread execution and branching. SIMD describes one instruction operating on multiple data values. SIMT code uses SIMD-like execution machinery; these are not competing labels.
- CUDA cores perform arithmetic; they do not "hold" entire warps. Subpartitions maintain resident warps and use schedulers and execution pipelines to carry out their instructions.

Practice: [Warp drill](warp-drill.md).

## 4.5 Control divergence

Remember: Different data is expected. Different execution paths within one warp cause control divergence.

- An `if` does not automatically diverge: divergence occurs when threads in the same warp choose different paths. Different warps choosing different paths is not within-warp divergence.
- For a divergent branch, each path's instructions take effect only for its participating threads. The other lanes are inactive for those instructions, not deleted.
- Loops can diverge when threads need different iteration counts. Boundary checks can also diverge; they are still necessary for correctness.
- A launched thread with an out-of-range data index still exists and evaluates the boundary check; it skips the guarded data work. This differs from an unused lane in a partial warp caused by the block's thread count. A warp where everyone skips the guarded work is not divergent at that check; mixed decisions within a warp are what matter.
- For the exercises, an "active warp" at a statement means at least one of its threads executes that statement. It does not mean all such warps execute simultaneously. SIMD efficiency here is participating lanes divided by 32, for that statement.
- `threadIdx` starts again in every block; a branch based on it repeats the same participation pattern across identical blocks. A branch based on the global index needs analysis of those global values.
- To analyze loop divergence, fix one loop-counter value and ask which threads still satisfy their loop tests. Threads can share the same loop body but stop after different numbers of iterations.
- Extra paths and inactive lanes can reduce efficiency, but divergence does not imply an exact 2x slowdown.
- Modern independent thread scheduling can interleave paths. Do not assume warp membership supplies synchronization; use the appropriate barrier when cooperation requires it.
- This section mainly teaches recognition and consequences, not a comprehensive mitigation toolkit. Synchronization ensures correctness; it does not eliminate divergence.

## 4.6 Warp scheduling and latency tolerance

Remember: Many resident warps provide alternatives; schedulers choose ready instructions to keep the hardware busy.

- Translate resident as "currently assigned to this SM, with resources reserved." Ready means its next instruction can proceed. Selected means a scheduler chooses that instruction to issue. These are not separate kinds of warp or exclusive levels: a ready or selected warp is still resident, as is one waiting for data.
- The kernel launch requests blocks and threads. Blocks are admitted onto SMs as capacity becomes available, reserving registers, shared memory if needed, and thread/block slots. This happens automatically; it is separate from `cudaMalloc` for matrix storage. The whole grid need not be resident at once.
- When one warp waits for a load result or another dependency, another ready warp can supply useful work. The original operation is not faster; its waiting time is occupied by other work. That is latency hiding.
- Fine-grained multithreading selects among warps at instruction-issue opportunities, not only after a warp finishes its entire kernel execution. Selection is not necessarily round-robin.
- Execution state means the information needed to resume: instruction positions, register values, and execution-control information. In matmul, a thread's current accumulator and loop progress illustrate the values that must be preserved (their exact compiled storage is compiler-dependent).
- Zero-overhead scheduling: resident warps already have their own allocated register storage and on-chip execution state. Selecting B does not overwrite A's state, so selecting A again needs no save/restore round trip to memory. The arithmetic hardware is shared; the resident state remains available. This refers to switching among resident warps, not GPU-wide context switches or preemption, and does not mean instructions are instantaneous or the SM can never stall.
- Multiple schedulers allow parallel issue, while each scheduler also interleaves work from its assigned warps over time. Hopper has four warp schedulers per SM; this is not a guarantee of four completed warp instructions every cycle.
- More resident warps increase the chance of finding ready work. If all are waiting, the hardware can still idle; available resources limit how many warps can reside on an SM.

## 4.7 Resource partitioning and occupancy

Remember: Occupancy measures how full one SM's resident-warp capacity is. Every resource limit must allow the proposed assignment.

- A register stores bits used as an instruction's working values, such as operands, intermediate results, and addresses. The compiler generally places small thread-local working values in registers; source variables do not map one-to-one to physical registers.
- In matmul, the accumulator illustrates a value a thread keeps updating while input matrices and the output array remain in global memory. Exact register allocation requires inspecting compiled code; we have not measured our kernel's register usage here.
- H100 has 65,536 32-bit registers per SM: 4 bytes per register, totaling 256 KiB. That is an SM-wide pool, not 65,536 registers per thread or CUDA core, and it is not the shared-memory pool.
- Simplified budget: 2,048 resident threads using 32 registers each require 65,536 registers. Using 64 registers each gives a register-budget ceiling of 1,024 threads. Actual residency also depends on allocation granularity, block sizes, and other resource limits.
- Some values may be placed in thread-private local memory instead (including register spills); declaring a kernel variable does not guarantee register storage. Fewer resident warps can reduce opportunities to hide latency, but occupancy alone is not a performance measurement.
- Occupancy is `resident warps on one SM / maximum resident warps that SM supports`. It is not a total across all SMs and does not count only warps executing right now. For full-warp block sizes, the equivalent thread ratio is `resident threads / maximum resident threads per SM`.
- Occupancy and SIMD efficiency have different denominators: the SM's resident-warp capacity versus a warp's 32 lanes. An SM can have full occupancy while instructions use only some lanes in each warp.
- Resource limits are ceilings, not targets that must all be filled. Full occupancy does not require using every register or every block slot. Registers and other limits can prevent the SM from filling its warp capacity.
- For the paper exercises, propose a whole number of blocks, then check block count, total threads, and total registers against every stated limit. A maximum block count does not require using that many blocks; thread limits per block and per SM are different constraints. Use each question's hypothetical limits rather than carrying numbers over from another question or from Modal.
- Actual hardware also has register-allocation granularity and shared-memory constraints. Paper arithmetic is a simplified model; use compiled register usage and occupancy tools when investigating a real kernel. Higher occupancy is not automatically higher performance.

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

## Exercise recap and next step

We worked through the exercises in this order: 2, 3, 9, 1, 5, 4, 6, 7, 8. The answers were discussed with tutor prompts; this records completion, not independent mastery or timed testing.

- Check launch legality before analyzing occupancy or performance.
- Count launched threads separately from threads doing useful data work.
- For barrier waiting, distinguish elapsed time from time summed across threads. Each thread's wait is the last arrival time minus its own arrival time.
- Keep reinforcing occupancy versus SIMD efficiency, and check every resource budget before accepting a configuration.

The [Exercise 1 listing](exercise-01.cu) is the book's kernel and host function, copied for inspection with the user's explicit override. It has not been compiled or run as an exercise program here.

Next: Chapter 5, starting with Section 5.1. My learning recommendation is to keep our existing matmul in mind and ask how much data must be supplied for its arithmetic. Revisit Chapter 4's concepts as we apply them; do not treat exercise completion as proof that they no longer need practice.

## Sources and scope

These pointers summarize our explanations using PMPP, fifth edition, Sections 4.1-4.8, read during the discussion. NVIDIA references checked during the discussion support the implementation-specific clarifications:

- [CUDA guide: SIMT and hardware multithreading](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#simt-architecture)
- [CUDA guide: Hopper / compute capability 9.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#compute-capability-9-0)
- [Nsight Compute: SMs and subpartitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#streaming-multiprocessor)
- [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Hopper tuning guide: register capacity and occupancy](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy)
- [CUDA programming model: registers and resource allocation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA 12.8.1 runtime: device-property fields and units](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/structcudaDeviceProp.html)
- [CUDA best practices: occupancy and resource limits](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)

Architecture examples are not universal hardware guarantees. These notes deliberately do not treat the book's "GPCs started with Hopper" wording as a historical fact.
