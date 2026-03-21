#include <stdio.h>
#include <stdlib.h>

__global__ void vecAddKernel(float *A, float *B, float *C, int n);

int main() {
  int n = 1 << 20; // 1M elements
  int size = n * sizeof(float);

  // memory allocation on host
  float *A_h = (float *)malloc(size);
  float *B_h = (float *)malloc(size);
  float *C_h = (float *)malloc(size);

  // filling up these with dummies
  for (int i = 0; i < n; ++i) {
    A_h[i] = 1.0f;
    B_h[i] = 2.0f;
  }

  // device pointers declaration
  float *A_d, *B_d, *C_d;

  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // calling the kernel to launch a grid of threads to perform vector addition
  vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

  // copying c from the device memory
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  printf("Printing the first 20 results\n");
  for (int i = 0; i < 20; ++i) {
    printf("C_h at %i: %f\n", i, C_h[i]);
  }

  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}

__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int global_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_i < n) {
    C[global_i] = A[global_i] + B[global_i];
  }
}
