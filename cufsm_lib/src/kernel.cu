#include "kernel.cuh"

void cuda_add(int *a, int *b, int *c, int N) {
    int *d_a, *d_b, *d_c;
    int size = 5 * sizeof(int);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cu_add<<<1, 5>>>(d_a, d_b, d_c, 5);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
}

void pytorch_add(float *a, float *b, float *c, int N) {
    cu_add<<<1, 5>>>(a, b, c, N);
}

__global__ void cu_add(int *a, int *b, int *c, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < N) {
        c[index] = a[index] + b[index];
        index += blockDim.x * gridDim.x;
    }
}

__global__ void cu_add(float *a, float *b, float *c, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < N) {
        c[index] = a[index] + b[index];
        index += blockDim.x * gridDim.x;
    }
}
