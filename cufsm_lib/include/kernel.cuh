#pragma once

#include <cuda_runtime.h>

void cuda_add(int *a, int *b, int *c, int N);
void pytorch_add(float *a, float *b, float *c, int N);

// Simple CUDA kernel example
__global__ void cu_add(int *a, int *b, int *c, int N);
__global__ void cu_add(float *a, float *b, float *c, int N);