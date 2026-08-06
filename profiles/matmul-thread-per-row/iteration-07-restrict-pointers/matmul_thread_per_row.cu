#include "cuda_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void checkCuda(cudaError_t result, const char *operation);
__global__ void MatMulKernel(const float *__restrict__ M_d, const float *__restrict__ N_d, float *__restrict__ out_d, size_t row_M, size_t col_M, size_t col_N);
static float referenceElement(size_t row, size_t col, size_t col_M);
static void validateSamples(const float *out_h, size_t row_M, size_t col_M, size_t col_N);
void matmul(const float *M_h, const float *N_h, float *out_h, size_t row_M, size_t col_M, size_t col_N);

int main(void) {
  printGpuInfo();
  const size_t row_M = 1024;
  const size_t col_M = 1024;
  const size_t col_N = 1024;
  float *M_h = (float *)malloc(sizeof(float) * row_M * col_M);
  float *N_h = (float *)malloc(sizeof(float) * col_M * col_N);
  float *out_h = (float *)malloc(sizeof(float) * row_M * col_N);
  if (M_h == NULL || N_h == NULL || out_h == NULL) {
    fprintf(stderr, "host allocation failed\n");
    free(M_h);
    free(N_h);
    free(out_h);
    return 1;
  }
  for (size_t i = 0; i < row_M; ++i)
    for (size_t j = 0; j < col_M; ++j)
      M_h[i * col_M + j] = i + j + 2;
  for (size_t i = 0; i < col_M; ++i)
    for (size_t j = 0; j < col_N; ++j)
      N_h[i * col_N + j] = i + j + 3;
  matmul(M_h, N_h, out_h, row_M, col_M, col_N);
  validateSamples(out_h, row_M, col_M, col_N);
  free(M_h);
  free(N_h);
  free(out_h);
  return 0;
}

__global__ void MatMulKernel(const float *__restrict__ M_d, const float *__restrict__ N_d, float *__restrict__ out_d, size_t row_M, size_t col_M, size_t col_N) {
  constexpr int tile_size = 32;
  constexpr int thread_tile = 16;
  __shared__ float M_tile[tile_size][tile_size];
  __shared__ float N_tile[tile_size][tile_size];
  const size_t row = blockIdx.y * tile_size + threadIdx.y;
  const size_t col = blockIdx.x * tile_size + threadIdx.x;
  float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
  for (size_t tile_start = 0; tile_start < col_M; tile_start += tile_size) {
    for (int local_row = threadIdx.y; local_row < tile_size; local_row += thread_tile) {
      for (int local_col = threadIdx.x; local_col < tile_size; local_col += thread_tile) {
        const size_t M_row = blockIdx.y * tile_size + local_row;
        const size_t M_col = tile_start + local_col;
        const size_t N_row = tile_start + local_row;
        const size_t N_col = blockIdx.x * tile_size + local_col;
        M_tile[local_row][local_col] = M_row < row_M && M_col < col_M ? M_d[M_row * col_M + M_col] : 0.0f;
        N_tile[local_row][local_col] = N_row < col_M && N_col < col_N ? N_d[N_row * col_N + N_col] : 0.0f;
      }
    }
    __syncthreads();
    for (int k = 0; k < tile_size; ++k) {
      const float m0 = M_tile[threadIdx.y][k];
      const float m1 = M_tile[threadIdx.y + thread_tile][k];
      const float n0 = N_tile[k][threadIdx.x];
      const float n1 = N_tile[k][threadIdx.x + thread_tile];
      acc00 += m0 * n0;
      acc01 += m0 * n1;
      acc10 += m1 * n0;
      acc11 += m1 * n1;
    }
    __syncthreads();
  }
  if (row < row_M && col < col_N) out_d[row * col_N + col] = acc00;
  if (row < row_M && col + thread_tile < col_N) out_d[row * col_N + col + thread_tile] = acc01;
  if (row + thread_tile < row_M && col < col_N) out_d[(row + thread_tile) * col_N + col] = acc10;
  if (row + thread_tile < row_M && col + thread_tile < col_N) out_d[(row + thread_tile) * col_N + col + thread_tile] = acc11;
}

static float referenceElement(size_t row, size_t col, size_t col_M) {
  float accumulator = 0.0f;
  for (size_t k = 0; k < col_M; ++k)
    accumulator += (float)(row + k + 2) * (float)(k + col + 3);
  return accumulator;
}

static void validateSamples(const float *out_h, size_t row_M, size_t col_M, size_t col_N) {
  const size_t samples[][2] = {{0, 0}, {1, 17}, {511, 513}, {1023, 1023}};
  for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i) {
    const size_t row = samples[i][0], col = samples[i][1];
    const float expected = referenceElement(row, col, col_M);
    const float actual = out_h[row * col_N + col];
    if (fabsf(actual - expected) > 1.0e-4f * fabsf(expected)) {
      fprintf(stderr, "sample validation failed at (%zu, %zu): got %f expected %f\n", row, col, actual, expected);
      exit(1);
    }
  }
  printf("Sample validation passed for 4 positions; first=%f last=%f\n", out_h[0], out_h[row_M * col_N - 1]);
}

void matmul(const float *M_h, const float *N_h, float *out_h, size_t row_M, size_t col_M, size_t col_N) {
  const size_t tile_size = 32;
  const size_t threads_per_dim = 16;
  const dim3 dimBlock(threads_per_dim, threads_per_dim, 1);
  const dim3 dimGrid((col_N + tile_size - 1) / tile_size, (row_M + tile_size - 1) / tile_size, 1);
  const size_t M_bytes = row_M * col_M * sizeof(float), N_bytes = col_M * col_N * sizeof(float), out_bytes = row_M * col_N * sizeof(float);
  float *M_d, *N_d, *out_d;
  checkCuda(cudaMalloc((void **)&M_d, M_bytes), "allocating M on the device");
  checkCuda(cudaMalloc((void **)&N_d, N_bytes), "allocating N on the device");
  checkCuda(cudaMalloc((void **)&out_d, out_bytes), "allocating output on the device");
  checkCuda(cudaMemcpy(M_d, M_h, M_bytes, cudaMemcpyHostToDevice), "copying M to the device");
  checkCuda(cudaMemcpy(N_d, N_h, N_bytes, cudaMemcpyHostToDevice), "copying N to the device");
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start), "creating start event");
  checkCuda(cudaEventCreate(&stop), "creating stop event");
  checkCuda(cudaEventRecord(start), "recording start event");
  for (int iteration = 0; iteration < 10; ++iteration)
    MatMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, out_d, row_M, col_M, col_N);
  checkCuda(cudaGetLastError(), "launching MatMulKernel");
  checkCuda(cudaEventRecord(stop), "recording stop event");
  checkCuda(cudaEventSynchronize(stop), "synchronizing stop event");
  float elapsed_ms;
  checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "measuring kernel time");
  printf("Kernel average time over 10 launches: %.3f ms\n", elapsed_ms / 10.0f);
  checkCuda(cudaEventDestroy(start), "destroying start event");
  checkCuda(cudaEventDestroy(stop), "destroying stop event");
  checkCuda(cudaMemcpy(out_h, out_d, out_bytes, cudaMemcpyDeviceToHost), "copying output to the host");
  checkCuda(cudaFree(M_d), "freeing M on the device");
  checkCuda(cudaFree(N_d), "freeing N on the device");
  checkCuda(cudaFree(out_d), "freeing output on the device");
}

static void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
    exit(1);
  }
}
