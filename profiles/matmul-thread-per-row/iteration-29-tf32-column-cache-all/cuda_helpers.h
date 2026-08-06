#ifndef PMPP_CUDA_HELPERS_H
#define PMPP_CUDA_HELPERS_H

#include <cuda_runtime.h>
#include <stdio.h>

static inline void printGpuInfo(void) {
  int device = 0;
  cudaDeviceProp prop;

  if (cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    printf("GPU: %s\n", prop.name);
    printf("compute capability: %d.%d\n", prop.major, prop.minor);
  }
}

#endif
