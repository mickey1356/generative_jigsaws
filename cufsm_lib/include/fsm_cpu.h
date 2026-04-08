#pragma once

void fsm_cpu(float *T, const float *src, const float *f, int N, int L,
             int iters);

void cpu_eikonal_update(float *T, const float *f, int i, int s, int L, float h);

void fsm_cpu_grad(float *dT, const float *T, int N, int L);

void fsm_adjoint_cpu(const float *T, const float *grad_T, const float *src,
                     const float *f, float *grad_f, float *grad_src, int N,
                     int L, int iters);