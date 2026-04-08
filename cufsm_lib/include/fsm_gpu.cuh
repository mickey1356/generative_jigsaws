#pragma once

#include <cuda_runtime.h>

void fsm_gpu(float *T, const float *src, const float *f, int N, int L,
             int iters);
void fsm_adjoint_gpu(const float *T, const float *grad_T, const float *src,
                     const float *f, float *grad_f, float *grad_src, int N,
                     int L, int iters);

__global__ void init_T(float *T, const float *src, const float *f, int N, int L,
                       float h);
__global__ void gpu_eikonal_update(float *T, const float *f, const float *src,
                                   int N, int L, float h);
__global__ void gpu_adjoint_update(const float *T, float *lambda,
                                   const float *src, const float *f,
                                   float *grad_f, int N, int L, float h);
__global__ void finalize_grad_src(const float *lambda, const float *src,
                                  const float *f, float *grad_src, int N, int L,
                                  float h);