#include <stdio.h>
#include <stdlib.h>

static void checkCuda(cudaError_t result, const char *operation);
__global__ void MatMulKernel(const float *M_d, const float *N_d, float *out_d, size_t row_M, size_t col_M, size_t col_N);
void matmul(const float *M_h, const float *N_h, float *out_h, size_t row_M, size_t col_M, size_t col_N);

int main(void) {
  size_t row_M = 5;
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

/*
  1. Calculate the thread’s global row index i.
  2. Guard against i being outside the output matrix.
  3. Visit every output column j.
  4. For each j, initialize a fresh accumulator.
  5. Walk through shared index k and accumulate the dot product.
  6. Store the result at output position [i][j].
*/

__global__ void MatMulKernel(const float *M_d, const float *N_d, float *out_d, size_t row_M, size_t col_M, size_t col_N) {
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;

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

void matmul(const float *M_h, const float *N_h, float *out_h, size_t row_M, size_t col_M, size_t col_N) {
  const size_t row_threads_per_block = 4;
  const size_t row_block_count = (row_M + row_threads_per_block - 1) / row_threads_per_block;
  const size_t M_bytes = row_M * col_M * sizeof(float);
  const size_t N_bytes = col_M * col_N * sizeof(float);
  const size_t out_bytes = row_M * col_N * sizeof(float);
  float *M_d, *N_d, *out_d;
  checkCuda(cudaMalloc((void **)&M_d, M_bytes), "allocating M on the device");
  checkCuda(cudaMalloc((void **)&N_d, N_bytes), "allocating N on the device");
  checkCuda(cudaMalloc((void **)&out_d, out_bytes), "allocating the output on the device");

  dim3 dimBlock(1, row_threads_per_block, 1);
  dim3 dimGrid(1, row_block_count, 1);

  checkCuda(cudaMemcpy(M_d, M_h, M_bytes, cudaMemcpyHostToDevice), "copying M to the device");
  checkCuda(cudaMemcpy(N_d, N_h, N_bytes, cudaMemcpyHostToDevice), "copying N to the device");

  MatMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, out_d, row_M, col_M, col_N);
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
