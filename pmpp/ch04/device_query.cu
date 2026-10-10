#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
    exit(EXIT_FAILURE);
  }
}

int main(void) {
  int device_count = 0;
  checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    fprintf(stderr, "No CUDA devices are visible.\n");
    return EXIT_FAILURE;
  }

  int runtime_version = 0;
  int driver_version = 0;
  checkCuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  checkCuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  printf("Visible CUDA devices: %d\n", device_count);
  printf("CUDA runtime version: %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
  printf("CUDA driver-supported version: %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop = {};
    checkCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    printf("\n=== Device %d: %s ===\n", device, prop.name);
    printf("Compute capability (major.minor): %d.%d\n", prop.major, prop.minor);
    printf("SM count (multiProcessorCount): %d\n", prop.multiProcessorCount);
    printf("Threads per warp (warpSize): %d\n", prop.warpSize);
    printf("Legacy reported clock (clockRate): %d kHz = %.1f MHz\n", prop.clockRate, prop.clockRate / 1000.0);

    printf("\nPer-block and grid launch limits\n");
    printf("Threads per block (maxThreadsPerBlock): %d\n", prop.maxThreadsPerBlock);
    printf("Block dimensions (maxThreadsDim), x/y/z: %d / %d / %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Grid dimensions (maxGridSize), x/y/z: %d / %d / %d blocks\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("32-bit registers per block (regsPerBlock): %d\n", prop.regsPerBlock);
    printf("Shared memory per block (sharedMemPerBlock): %zu bytes = %.1f KiB\n", prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
    printf("Opt-in shared memory per block (sharedMemPerBlockOptin): %zu bytes = %.1f KiB\n", prop.sharedMemPerBlockOptin,
           prop.sharedMemPerBlockOptin / 1024.0);

    printf("\nPer-SM residency and storage limits\n");
    printf("Resident threads per SM (maxThreadsPerMultiProcessor): %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Resident warps per SM (threads per SM / warp size): %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("Resident blocks per SM (maxBlocksPerMultiProcessor): %d\n", prop.maxBlocksPerMultiProcessor);
    printf("32-bit registers per SM (regsPerMultiprocessor): %d = %.1f KiB\n", prop.regsPerMultiprocessor, prop.regsPerMultiprocessor * 4.0 / 1024.0);
    printf("Shared memory per SM (sharedMemPerMultiprocessor): %zu bytes = %.1f KiB\n", prop.sharedMemPerMultiprocessor,
           prop.sharedMemPerMultiprocessor / 1024.0);

    printf("\nDevice-wide storage and supported features\n");
    printf("Total global memory (totalGlobalMem): %zu bytes = %.2f GiB\n", prop.totalGlobalMem, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("L2 cache (l2CacheSize): %d bytes = %.2f MiB\n", prop.l2CacheSize, prop.l2CacheSize / (1024.0 * 1024.0));
    printf("Cluster launch (clusterLaunch): %s\n", prop.clusterLaunch ? "supported" : "not supported");
    printf("Cooperative launch (cooperativeLaunch): %s\n", prop.cooperativeLaunch ? "supported" : "not supported");
  }

  printf("\nThese are capacity limits and not measured performance or a kernel's actual occupancy.\n");
  printf("Block dimensions must satisfy both the per-axis limits and the total threads-per-block limit.\n");
  printf("clockRate is a deprecated property and not a live clock measurement.\n");
  printf("Kernel occupancy and maximum cluster size require kernel-specific occupancy queries.\n");
  return EXIT_SUCCESS;
}
