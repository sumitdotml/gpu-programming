#include <stdio.h>
#include <stdlib.h>

/*
  1. Calculate the thread’s global row index i.
  2. Guard against i being outside the output matrix.
  3. Visit every output column j.
  4. For each j, initialize a fresh accumulator.
  5. Walk through shared index k and accumulate the dot product.
  6. Store the result at output position [i][j].
*/

__global__ void MatMulKernel(float *M_d, float *N_d, float *out_d, size_t row_M, size_t col_M, size_t col_N) {
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  // size_t col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < row_M) {
    for (size_t j = 0; j < col_N; ++j) {
      float accumulator = 0.0f;
      for (size_t k = 0; k < col_M; ++k) {
        // offset + k for M_d, offset + j for N_d
        accumulator += M_d[row * col_M + k] * N_d[k * col_N + j];
      }
      out_d[row * col_N + j] = accumulator;
    }
  }
}

void matmul(float *M_h, float *N_h, float *out_h, size_t row_M, size_t col_M, size_t col_N) {
  const size_t row_threads_per_block = 4;
  const size_t row_block_count = (row_M + row_threads_per_block - 1) / row_threads_per_block;
  float *M_d, *N_d, *out_d;
  cudaMalloc((void **)&M_d, row_M * col_M * sizeof(float));
  cudaMalloc((void **)&N_d, col_M * col_N * sizeof(float));
  cudaMalloc((void **)&out_d, row_M * col_N * sizeof(float));

  dim3 dimBlock(1, row_threads_per_block, 1);
  dim3 dimGrid(1, row_block_count, 1);

  cudaMemcpy(M_d, M_h, row_M * col_M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N_h, col_M * col_N * sizeof(float), cudaMemcpyHostToDevice);

  MatMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, out_d, row_M, col_M, col_N);
  cudaMemcpy(out_h, out_d, row_M * col_N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(out_d);
}

int main(void) {
  size_t row_M = 2;
  size_t col_M = 3;
  size_t col_N = 4;

  float *M_h = (float *)malloc(sizeof(float) * row_M * col_M);
  float *N_h = (float *)malloc(col_M * col_N * sizeof(float));
  float *out_h = (float *)malloc(row_M * col_N * sizeof(float));

  // filling with dummies
  for (size_t i = 0; i < row_M; ++i) {
    for (size_t j = 0; j < col_M; ++j) {
      M_h[i * col_M + j] = i + j + 2;
    }
  }
  for (size_t i = 0; i < col_M; ++i) {
    for (size_t j = 0; j < col_N; ++j) {
      N_h[i * col_N + j] = i + j + 3;
    }
  }
  matmul(M_h, N_h, out_h, row_M, col_M, col_N);
  printf("\nPrinting the output matrix:\n");
  for (size_t i = 0; i < row_M; ++i) {
    for (size_t j = 0; j < col_N; ++j) {
      printf("%f ", out_h[col_N * i + j]);
    }
    printf("\n");
  }
  free(M_h);
  free(N_h);
  free(out_h);
  return 0;
}
