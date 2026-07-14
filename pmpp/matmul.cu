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


__global__ void MatMulKernel(float *M_d, float *N_d, float *out_d, size_t row_M, size_t col_M, size_t col_N){
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  // size_t col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < row_M){
      for (size_t j = 0; j < col_N; ++j){
        float accumulator = 0.0f;
        for (size_t k = 0; k < col_M; ++k){
          // offset + k for M_d, offset + j for N_d
          accumulator += M_d[row * col_M + k] * N_d[k * col_N + j];
        }
        out_d[row * col_N + j] = accumulator;
      }
    }
}

void matmul(float *M_h, float *N_h, float *out_h, size_t row_M, size_t col_M, size_t col_N) {
  // will do guard later
  float *M_d, *N_d, *out_d;
  cudaMalloc((void **)&M_d, row_M * col_M * sizeof(float));
  cudaMalloc((void**)&N_d, col_M * col_N * sizeof(float));
  cudaMalloc((void **)&out_d, row_M * col_N * sizeof(float));

}
