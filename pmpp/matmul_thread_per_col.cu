#include "cuda_helpers.h"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

static void checkCuda(cudaError_t result, const char *operation);
__global__ void MatMulKernel(const float *M_d, const float *N_d, float *out_d, size_t num_rows, size_t num_inner, size_t num_cols);
void matmul(const float *M_h, const float *N_h, float *out_h, size_t num_rows, size_t num_inner, size_t num_cols);

int main(void) {
  printGpuInfo();

  size_t num_rows = 5;
  size_t num_inner = 3;
  size_t num_cols = 4;

  float *M_h = (float *)malloc(num_rows * num_inner * sizeof(float));
  float *N_h = (float *)malloc(num_inner * num_cols * sizeof(float));
  float *out_h = (float *)malloc(num_rows * num_cols * sizeof(float));

  // filling with dummies
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_inner; ++j) {
      M_h[i * num_inner + j] = i + j + 2;
    }
  }
  for (size_t i = 0; i < num_inner; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      N_h[i * num_cols + j] = i + j + 3;
    }
  }

  matmul(M_h, N_h, out_h, num_rows, num_inner, num_cols);

  printf("\nPrinting the output matrix:\n");
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      printf("%f ", out_h[num_cols * i + j]);
    }
    printf("\n");
  }
  free(M_h);
  free(N_h);
  free(out_h);
  return 0;
}

/*
  Shapes:
    M   is num_rows  x num_inner
    N   is num_inner x num_cols
    out is num_rows  x num_cols

  1. Calculate the thread's global column index col.
  2. Guard against col being outside the output matrix.
  3. Visit every output row.
  4. For every row, initialize a fresh accumulator.
  5. Walk through shared index k and accumulate the dot product.
  6. Store the result at output position [row][col].
*/

__global__ void MatMulKernel(const float *M_d, const float *N_d, float *out_d, size_t num_rows, size_t num_inner, size_t num_cols) {
  // global col index
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;

  if (col < num_cols) {
    for (size_t row = 0; row < num_rows; ++row) {
      float accumulator = 0.0f;
      for (size_t k = 0; k < num_inner; ++k) {
        accumulator += M_d[row * num_inner + k] * N_d[k * num_cols + col];
      }
      out_d[row * num_cols + col] = accumulator;
    }
  }
}

void matmul(const float *M_h, const float *N_h, float *out_h, size_t num_rows, size_t num_inner, size_t num_cols) {
  const size_t col_threads_per_block = 3; // just an impromptu choice
  const size_t col_block_count = (num_cols + col_threads_per_block - 1) / col_threads_per_block;
  const size_t M_bytes = sizeof(float) * num_rows * num_inner;
  const size_t N_bytes = sizeof(float) * num_inner * num_cols;
  const size_t out_bytes = sizeof(float) * num_rows * num_cols;
  float *M_d, *N_d, *out_d;

  checkCuda(cudaMalloc((void **)&M_d, M_bytes), "allocating memory for M on the device");
  checkCuda(cudaMalloc((void **)&N_d, N_bytes), "allocating memory for N on the device");
  checkCuda(cudaMalloc((void **)&out_d, out_bytes), "allocating memory for out on the device");

  dim3 gridDim(col_block_count, 1, 1);
  dim3 blockDim(col_threads_per_block, 1, 1);

  checkCuda(cudaMemcpy(M_d, M_h, M_bytes, cudaMemcpyHostToDevice), "copying data from host to device for M");
  checkCuda(cudaMemcpy(N_d, N_h, N_bytes, cudaMemcpyHostToDevice), "copying data from host to device for N");

  MatMulKernel<<<gridDim, blockDim>>>(M_d, N_d, out_d, num_rows, num_inner, num_cols);
  checkCuda(cudaGetLastError(), "launching MatMulKernel");

  checkCuda(cudaMemcpy(out_h, out_d, out_bytes, cudaMemcpyDeviceToHost), "copying data from the calculated matmul result from device to host");
  checkCuda(cudaFree(M_d), "freeing M_d memory");
  checkCuda(cudaFree(N_d), "freeing N_d memory");
  checkCuda(cudaFree(out_d), "freeing out_d memory");
}

static void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
    exit(1);
  }
}
