__global__ void foo_kernel(int *a, int *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x < 40 || threadIdx.x >= 104) {
    b[i] = a[i] + 1;
  }
  if (i % 2 == 0) {
    a[i] = b[i] * 2;
  }
  for (unsigned int j = 0; j < 5 - (i % 3); ++j) {
    b[i] += j;
  }
}
void foo(int *a_d, int *b_d) {
  unsigned int N = 1024;
  foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d);
}
