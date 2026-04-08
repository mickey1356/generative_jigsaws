#include "fsm_gpu.cuh"

#include "constants.h"

void fsm_gpu(float *T, const float *src, const float *f, int N, int L,
             int iters) {
    /*
    Solve the Eikonal equation (|grad T| = f) using the Fast Sweeping Method on
    GPU Inputs src : input array of source points (N x 2) in the range [-1, 1] f :
    input array of slowness values (L x L) iters : number of iterations to perform
    Returns
          T : output array of travel times[N * L * L]
    */

    // compute the step size (we assume the grid spans [-1, 1] in both dimensions)
    float h = 2.0f / (L - 1);

    // intialize T to be inf (or distance from source at 4 corners)
    init_T<<<512, 128>>>(T, src, f, N, L, h);

    cudaDeviceSynchronize();

    for (int it = 0; it < iters; it++) {
        // parallelize over sources and each grid point
        gpu_eikonal_update<<<512, 128>>>(T, f, src, N, L, h);
    }

    cudaDeviceSynchronize();
}

__global__ void init_T(float *T, const float *src, const float *f, int N, int L,
                       float h) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * L * L;
    while (index < total_size) {
        int s = index / (L * L);
        int grid_idx = index % (L * L);
        int r = grid_idx / L;
        int c = grid_idx % L;

        float u = ((src[s * 2 + 0] + 1.0f) / 2.0f) * (L - 1);
        float v = ((src[s * 2 + 1] + 1.0f) / 2.0f) * (L - 1);

        int i0 = (int)floorf(u);
        int j0 = (int)floorf(v);

        float val = INF;
        // Check if this grid point is one of the 4 corners
        if ((r == j0 || r == j0 + 1) && (c == i0 || c == i0 + 1)) {
            if (r >= 0 && r < L && c >= 0 && c < L) {
                float du = u - c;
                float dv = v - r;
                float dist = sqrtf(du * du + dv * dv) * h;
                val = dist * f[r * L + c];
            }
        }

        T[index] = val;
        index += blockDim.x * gridDim.x;
    }
}

__global__ void gpu_eikonal_update(float *T, const float *f, const float *src,
                                   int N, int L, float h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * L * L;
    while (idx < total_size) {
        int s = idx / (L * L);
        int grid_idx = idx % (L * L);
        int i = grid_idx / L;
        int j = grid_idx % L;

        float u = ((src[s * 2 + 0] + 1.0f) / 2.0f) * (L - 1);
        float v = ((src[s * 2 + 1] + 1.0f) / 2.0f) * (L - 1);

        int i0 = (int)floorf(u);
        int j0 = (int)floorf(v);

        // Skip source corners
        if ((i == j0 || i == j0 + 1) && (j == i0 || j == i0 + 1)) {
            idx += blockDim.x * gridDim.x;
            continue;
        }

        // Get neighboring T values with boundary checks
        float T_left = (j > 0) ? T[s * L * L + i * L + (j - 1)] : INF;
        float T_right = (j < L - 1) ? T[s * L * L + i * L + (j + 1)] : INF;
        float T_up = (i > 0) ? T[s * L * L + (i - 1) * L + j] : INF;
        float T_down = (i < L - 1) ? T[s * L * L + (i + 1) * L + j] : INF;

        // Eikonal update
        float a = fminf(T_left, T_right);
        float b = fminf(T_up, T_down);
        float f_ij = f[i * L + j];

        float new_T;
        if (fabs(a - b) >= f_ij * h) {
            new_T = fminf(a, b) + f_ij * h;
        } else {
            new_T = (a + b + sqrtf(2.0f * f_ij * f_ij * h * h - (a - b) * (a - b))) /
                    2.0f;
        }

        T[s * L * L + i * L + j] = fminf(T[s * L * L + i * L + j], new_T);
        idx += blockDim.x * gridDim.x;
    }
}

void fsm_adjoint_gpu(const float *T, const float *grad_T, const float *src,
                     const float *f, float *grad_f, float *grad_src, int N,
                     int L, int iters) {
    float h = 2.0f / (L - 1);

    // Allocate and initialize lambda on device
    float *d_lambda;
    cudaMalloc(&d_lambda, N * L * L * sizeof(float));
    cudaMemcpy(d_lambda, grad_T, N * L * L * sizeof(float),
               cudaMemcpyDeviceToDevice);

    for (int it = 0; it < iters; it++) {
        gpu_adjoint_update<<<512, 128>>>(T, d_lambda, src, f, grad_f, N, L, h);
    }

    // Finalize grad_src from the accumulated adjoints at corners
    finalize_grad_src<<<512, 128>>>(d_lambda, src, f, grad_src, N, L, h);

    cudaDeviceSynchronize();
    cudaFree(d_lambda);
}

__global__ void gpu_adjoint_update(const float *T, float *lambda,
                                   const float *src, const float *f,
                                   float *grad_f, int N, int L, float h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * L * L;

    while (idx < total_size) {
        int s = idx / (L * L);
        int grid_idx = idx % (L * L);
        int i = grid_idx / L;
        int j = grid_idx % L;

        float u = ((src[s * 2 + 0] + 1.0f) / 2.0f) * (L - 1);
        float v = ((src[s * 2 + 1] + 1.0f) / 2.0f) * (L - 1);

        int i0 = (int)floorf(u);
        int j0 = (int)floorf(v);

        // Skip source corners
        if ((i == j0 || i == j0 + 1) && (j == i0 || j == i0 + 1)) {
            idx += blockDim.x * gridDim.x;
            continue;
        }

        // Atomic exchange to "take" the current sensitivity charge
        float l_ij = atomicExch(&lambda[idx], 0.0f);

        if (l_ij != 0) {
            // Get neighboring T values
            float T_l = (j > 0) ? T[s * L * L + i * L + (j - 1)] : INF;
            float T_r = (j < L - 1) ? T[s * L * L + i * L + (j + 1)] : INF;
            float T_u = (i > 0) ? T[s * L * L + (i - 1) * L + j] : INF;
            float T_d = (i < L - 1) ? T[s * L * L + (i + 1) * L + j] : INF;

            float a = fminf(T_l, T_r);
            int p_a = (T_l < T_r) ? (idx - 1) : (idx + 1);

            float b = fminf(T_u, T_d);
            int p_b = (T_u < T_d) ? (idx - L) : (idx + L);

            float f_ij = f[i * L + j];

            if (fabsf(a - b) >= f_ij * h) {
                // 1D Update
                if (a < b) {
                    atomicAdd(&lambda[p_a], l_ij);
                } else {
                    atomicAdd(&lambda[p_b], l_ij);
                }
                atomicAdd(&grad_f[i * L + j], l_ij * h);
            } else {
                // 2D Update
                float delta = 2.0f * f_ij * f_ij * h * h - (a - b) * (a - b);
                if (delta > 0) {
                    float sqrt_delta = sqrtf(delta);
                    float da = 0.5f * (1.0f - (a - b) / sqrt_delta);
                    float db = 0.5f * (1.0f + (a - b) / sqrt_delta);
                    float df = (f_ij * h * h) / sqrt_delta;

                    atomicAdd(&lambda[p_a], l_ij * da);
                    atomicAdd(&lambda[p_b], l_ij * db);
                    atomicAdd(&grad_f[i * L + j], l_ij * df);
                }
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void finalize_grad_src(const float *lambda, const float *src,
                                  const float *f, float *grad_src, int N, int L,
                                  float h) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < N) {
        float u = ((src[s * 2 + 0] + 1.0f) / 2.0f) * (L - 1);
        float v = ((src[s * 2 + 1] + 1.0f) / 2.0f) * (L - 1);

        int i0 = (int)floorf(u);
        int j0 = (int)floorf(v);

        float g_u = 0.0f;
        float g_v = 0.0f;

        for (int di = 0; di <= 1; di++) {
            for (int dj = 0; dj <= 1; dj++) {
                int r = j0 + dj;
                int c = i0 + di;
                if (r >= 0 && r < L && c >= 0 && c < L) {
                    int idx = s * L * L + r * L + c;
                    float l_corner = lambda[idx];

                    float du = u - c;
                    float dv = v - r;
                    float dist_pixel = sqrtf(du * du + dv * dv);

                    if (dist_pixel > 1e-6f) {
                        float slowness = f[r * L + c];
                        g_u += l_corner * slowness * h * (du / dist_pixel);
                        g_v += l_corner * slowness * h * (dv / dist_pixel);
                    }
                }
            }
        }

        grad_src[s * 2 + 0] = g_u * (L - 1) / 2.0f;
        grad_src[s * 2 + 1] = g_v * (L - 1) / 2.0f;
    }
}