void vecAdd(float *A, float *B, float *C, int n) {
  int size = n * sizeof(float);
  float *d_A, *d_B, *d_C;
  // Part 1: Allocate device memory for A, B, and C
  // Copy A and B to device memory

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  // Part 2: Call kernel – to launch a grid of threads
  // to perform the actual vector addition

  // Part 3: Copy C from the device memory
  // Free device vectors
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
