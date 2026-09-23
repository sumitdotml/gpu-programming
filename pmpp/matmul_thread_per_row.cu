/*
 * Modal run:
 * make modal SRC=pmpp/matmul_thread_per_row.cu
 * image: nvidia/cuda:12.8.1-devel-ubuntu24.04
 * GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
 * compute capability: 12.0
 *
 * Printing the output matrix:
 * 38.000000 47.000000 56.000000 65.000000
 * 50.000000 62.000000 74.000000 86.000000
 * 62.000000 77.000000 92.000000 107.000000
 * 74.000000 92.000000 110.000000 128.000000
 * 86.000000 107.000000 128.000000 149.000000
 */

#include "cuda_helpers.h"

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

  float *M_h = (float *)malloc(sizeof(float) * num_rows * num_inner);
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

  1. Calculate the thread’s global row index row.
  2. Guard against row being outside the output matrix.
  3. Visit every output column.
  4. For each column, initialize a fresh accumulator.
  5. Walk through shared index k and accumulate the dot product.
  6. Store the result at output position [row][col].
*/

__global__ void MatMulKernel(const float *M_d, const float *N_d, float *out_d, size_t num_rows, size_t num_inner, size_t num_cols) {
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < num_rows) {
    for (size_t col = 0; col < num_cols; ++col) {
      float accumulator = 0.0f;
      for (size_t k = 0; k < num_inner; ++k) {
        // offset + k for M_d, offset + col for N_d
        accumulator += M_d[row * num_inner + k] * N_d[k * num_cols + col];
      }
      out_d[row * num_cols + col] = accumulator;
    }
  }
}

void matmul(const float *M_h, const float *N_h, float *out_h, size_t num_rows, size_t num_inner, size_t num_cols) {
  const size_t row_threads_per_block = 4;
  const size_t row_block_count = (num_rows + row_threads_per_block - 1) / row_threads_per_block;
  const size_t M_bytes = num_rows * num_inner * sizeof(float);
  const size_t N_bytes = num_inner * num_cols * sizeof(float);
  const size_t out_bytes = num_rows * num_cols * sizeof(float);
  float *M_d, *N_d, *out_d;
  checkCuda(cudaMalloc((void **)&M_d, M_bytes), "allocating M on the device");
  checkCuda(cudaMalloc((void **)&N_d, N_bytes), "allocating N on the device");
  checkCuda(cudaMalloc((void **)&out_d, out_bytes), "allocating the output on the device");

  dim3 dimBlock(1, row_threads_per_block, 1);
  dim3 dimGrid(1, row_block_count, 1);

  checkCuda(cudaMemcpy(M_d, M_h, M_bytes, cudaMemcpyHostToDevice), "copying M to the device");
  checkCuda(cudaMemcpy(N_d, N_h, N_bytes, cudaMemcpyHostToDevice), "copying N to the device");

  MatMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, out_d, num_rows, num_inner, num_cols);
  checkCuda(cudaGetLastError(), "launching MatMulKernel");

  checkCuda(cudaMemcpy(out_h, out_d, out_bytes, cudaMemcpyDeviceToHost), "copying the output to the host");
  checkCuda(cudaFree(M_d), "freeing M on the device");
  checkCuda(cudaFree(N_d), "freeing N on the device");
  checkCuda(cudaFree(out_d), "freeing the output on the device");
}

static void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
    exit(1);
  }
}
