#include <stdio.h>
#include <stdlib.h>

__global__ void matrixVectorMultiplyKernel(float *input_matrix, float *input_vector, float *output_vector, size_t dimension);
void matrixVectorMultiply(float *output_vector_h, float *input_matrix_h, float *input_vector_h, size_t dimension);

int main(void) {
  size_t dimension = 4;
  float *input_matrix_h = (float *)malloc(dimension * dimension * sizeof(float));
  float *input_vector_h = (float *)malloc(dimension * sizeof(float));
  float *output_vector_h = (float *)malloc(dimension * sizeof(float));

  // filling with dummies
  for (size_t i = 0; i < dimension; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      input_matrix_h[i * dimension + j] = i + j + 2;
    }
    input_vector_h[i] = i + 5;
  }

  matrixVectorMultiply(output_vector_h, input_matrix_h, input_vector_h, dimension);

  printf("Printing the results!\n");
  for (size_t i = 0; i < dimension; ++i) {
    printf("Element at output[%zu]: %f\n", i, output_vector_h[i]);
  }

  free(input_matrix_h);
  free(input_vector_h);
  free(output_vector_h);

  return 0;
}

// - multiplying a square, row-major float matrix by a float vector
// - producing one output-vector element from each matrix-row dot product
// - using one thread to calculate each output-vector element
// - writing a host stub that accepts output, matrix, vector, and side length
//
// using one dimension for both axes because the input matrix is square.
__global__ void matrixVectorMultiplyKernel(float *input_matrix, float *input_vector, float *output_vector, size_t dimension) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < dimension) {
    float dot_product = 0;
    for (size_t column = 0; column < dimension; ++column) {
      dot_product += input_matrix[row * dimension + column] * input_vector[column];
    }
    output_vector[row] = dot_product;
  }
}

void matrixVectorMultiply(float *output_vector_h, float *input_matrix_h, float *input_vector_h, size_t dimension) {
  float *input_matrix_d, *input_vector_d, *output_vector_d;
  cudaMalloc((void **)&input_matrix_d, dimension * dimension * sizeof(float));
  cudaMalloc((void **)&input_vector_d, dimension * sizeof(float));
  cudaMalloc((void **)&output_vector_d, dimension * sizeof(float));
  cudaMemcpy(input_matrix_d, input_matrix_h, dimension * dimension * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_vector_d, input_vector_h, dimension * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid((dimension + 3) / 4, 1, 1);
  dim3 dimBlock(4, 1, 1);

  matrixVectorMultiplyKernel<<<dimGrid, dimBlock>>>(input_matrix_d, input_vector_d, output_vector_d, dimension);

  cudaMemcpy(output_vector_h, output_vector_d, dimension * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(input_matrix_d);
  cudaFree(input_vector_d);
  cudaFree(output_vector_d);
}
