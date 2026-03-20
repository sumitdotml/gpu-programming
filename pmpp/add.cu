#include <stdio.h>

int vecAdd(float *A_h, float *B_h, float *C_h, int n) {
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_d;
  // Part 1: Allocate device memory for A, B, and C
  // Copy A and B to device memory

  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // Part 2: Call kernel – to launch a grid of threads
  // to perform the actual vector addition

  // Part 3: Copy C from the device memory
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  // Free device vectors
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
  int global_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_i < n) {
    C[global_i] = A[global_i] + B[global_i];
  }
}
