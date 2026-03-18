// compute vector C = A + B

#include <stdio.h>
#include <stdlib.h>

void vecAdd(float *A_h, float *B_h, float *C_h, int n);

int main(void){
    int n = 5;

    float *A_h = (float*) malloc(n * sizeof(float));
    float *B_h = (float*) malloc(n * sizeof(float));
    float *C_h = (float*) malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i){
        A_h[i] = i + 1;
        B_h[i] = i * 2;
    }

    vecAdd(A_h, B_h, C_h, n);

    for (int i = 0; i < n; ++ i){
        printf("%f\n", C_h[i]);
    }

    free(A_h);
    free(B_h);
    free(C_h);
    return 0;
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n){
    for (int i = 0; i < n; ++i){
        C_h[i] = A_h[i] + B_h[i];
    }
}