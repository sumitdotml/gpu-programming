#include <cstdlib>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

static void checkCuda(cudaError_t result, const char *operation);
__global__ void MatMulKernel(float *M_d, float *N_d, float *out_d, size_t num_rows, size_t num_inner, size_t num_cols);
void matmul(float *M_h, float *N_h, float *out_h, size_t num_rows, size_t num_inner, size_t num_cols);

int main(void) {
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

__global__ void MatMulKernel(float *M_d, float *N_d, float *out_d, size_t num_rows, size_t num_inner, size_t num_cols) {
  size_t row_index = blockDim.y * blockIdx.y + threadIdx.y;
  size_t col_index = blockDim.x * blockIdx.x + threadIdx.x;

  float accumulator = 0.0f;
  if (row_index < num_rows && col_index < num_cols) {
    for (size_t k = 0; k < num_inner; ++k) {
      accumulator += M_d[row_index * num_inner + k] * N_d[k * num_cols + col_index];
    }
    out_d[row_index * num_cols + col_index] = accumulator;
  }
}

void matmul(float *M_h, float *N_h, float *out_h, size_t num_rows, size_t num_inner, size_t num_cols) {
  size_t threads_per_width = 3;  // across the "grid" of out
  size_t threads_per_height = 3; // down the "grid" of out
  size_t block_count_x = (num_cols + threads_per_width - 1) / threads_per_width;
  size_t block_count_y = (num_rows + threads_per_height - 1) / threads_per_height;
  size_t M_bytes = num_rows * num_inner * sizeof(float);
  size_t N_bytes = num_inner * num_cols * sizeof(float);
  size_t out_bytes = num_rows * num_cols * sizeof(float);

  float *M_d, *N_d, *out_d;

  // allocating memory in device
  checkCuda(cudaMalloc((void **)&M_d, M_bytes), "malloc of M_d");
  checkCuda(cudaMalloc((void **)&N_d, N_bytes), "malloc of N_d");
  checkCuda(cudaMalloc((void **)&out_d, out_bytes), "malloc of out_d");

  dim3 gridDim(block_count_x, block_count_y, 1);
  dim3 blockDim(threads_per_width, threads_per_height, 1);

  checkCuda(cudaMemcpy(M_d, M_h, M_bytes, cudaMemcpyHostToDevice), "cudaMemcpyHtoD");
  checkCuda(cudaMemcpy(N_d, N_h, N_bytes, cudaMemcpyHostToDevice), "cudaMemcpyHtoD");

  MatMulKernel<<<gridDim, blockDim>>>(M_d, N_d, out_d, num_rows, num_inner, num_cols);
  checkCuda(cudaGetLastError(), "launching MatMulKernel");

  checkCuda(cudaMemcpy(out_h, out_d, out_bytes, cudaMemcpyDeviceToHost), "cudaMemcpyDtoH");

  checkCuda(cudaFree(M_d), "cudaFree");
  checkCuda(cudaFree(N_d), "cudaFree");
  checkCuda(cudaFree(out_d), "cudaFree");
}

static void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
    exit(1);
  }
}
