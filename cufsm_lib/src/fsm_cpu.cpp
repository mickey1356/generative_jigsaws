#include "fsm_cpu.h"

#include <algorithm>
#include <cmath>

#include "constants.h"

void fsm_cpu(float *T, const float *src, const float *f, int N, int L,
             int iters) {
    /*
      Solve the Eikonal equation (|grad T| = f) using the Fast Sweeping Method on
    CPU Inputs src : input array of source points (N x 2) in the range [-1, 1] f :
    input array of slowness values (L x L) iters : number of iterations to perform
    Returns
          T : output array of travel times[N * L * L]
    */

    // compute the step size (we assume the grid spans [-1, 1] in both dimensions)
    float h = 2.0f / (L - 1);

#pragma omp parallel for
    for (int s = 0; s < N; s++) {
        // intialize T to be inf
        for (int i = 0; i < L * L; i++) {
            T[s * L * L + i] = INF;
        }

        float u = ((src[s * 2 + 0] + 1.0f) / 2.0f) * (L - 1);
        float v = ((src[s * 2 + 1] + 1.0f) / 2.0f) * (L - 1);

        int i0 = (int)std::floor(u);
        int j0 = (int)std::floor(v);

        int corners[4];
        int num_corners = 0;

        for (int di = 0; di <= 1; di++) {
            for (int dj = 0; dj <= 1; dj++) {
                int r = j0 + dj;
                int c = i0 + di;
                if (r >= 0 && r < L && c >= 0 && c < L) {
                    float du = u - c;
                    float dv = v - r;
                    float dist = std::sqrt(du * du + dv * dv) * h;
                    int idx = r * L + c;
                    T[s * L * L + idx] = dist * f[idx];
                    corners[num_corners++] = idx;
                }
            }
        }

        for (int it = 0; it < iters; it++) {
            // for each point i
            for (int i = 0; i < L * L; i++) {
                bool is_corner = false;
                for (int k = 0; k < num_corners; k++) {
                    if (i == corners[k]) {
                        is_corner = true;
                        break;
                    }
                }
                if (is_corner)
                    continue;

                cpu_eikonal_update(T, f, i, s, L, h);
            }
        }
    }
}

void cpu_eikonal_update(float *T, const float *f, int i, int s, int L,
                        float h) {
    // get indices of neighbors, accounting for boundaries
    float T_left = (i % L == 0) ? INF : T[s * L * L + i - 1];
    float T_right = (i % L == L - 1) ? INF : T[s * L * L + i + 1];
    float T_up = (i < L) ? INF : T[s * L * L + i - L];
    float T_down = (i >= L * (L - 1)) ? INF : T[s * L * L + i + L];

    float a = std::min(T_left, T_right);
    float b = std::min(T_up, T_down);

    float T_new;
    if (std::abs(a - b) >= f[i] * h) {
        T_new = std::min(a, b) + h * f[i];
    } else {
        T_new =
            (a + b + std::sqrt(2 * f[i] * f[i] * h * h - (a - b) * (a - b))) / 2.0f;
    }

    T[s * L * L + i] = std::min(T[s * L * L + i], T_new);
}

void fsm_adjoint_cpu(const float *T, const float *grad_T, const float *src,
                     const float *f, float *grad_f, float *grad_src, int N,
                     int L, int iters) {
    float h = 2.0f / (L - 1);

    // Allocate lambda buffer for each source
    float *lambda = new float[N * L * L];
    for (int i = 0; i < N * L * L; i++) {
        lambda[i] = grad_T[i];
    }

// Process each source independently
#pragma omp parallel for
    for (int s = 0; s < N; s++) {
        float u = ((src[s * 2 + 0] + 1.0f) / 2.0f) * (L - 1);
        float v = ((src[s * 2 + 1] + 1.0f) / 2.0f) * (L - 1);

        int i0 = (int)std::floor(u);
        int j0 = (int)std::floor(v);

        int corners[4];
        int num_corners = 0;
        for (int di = 0; di <= 1; di++) {
            for (int dj = 0; dj <= 1; dj++) {
                int r = j0 + dj;
                int c = i0 + di;
                if (r >= 0 && r < L && c >= 0 && c < L) {
                    corners[num_corners++] = r * L + c;
                }
            }
        }

        for (int it = 0; it < iters; it++) {
            // Sweeping order: reverse order of forward pass for better convergence
            // We use 'push-and-clear' logic: each node moves its currently
            // accumulated adjoint 'charge' to its parents. This prevents explosion.
            for (int i = L * L - 1; i >= 0; i--) {
                bool is_corner = false;
                for (int k = 0; k < num_corners; k++) {
                    if (i == corners[k]) {
                        is_corner = true;
                        break;
                    }
                }
                if (is_corner)
                    continue;

                float l_i = lambda[s * L * L + i];
                if (l_i == 0)
                    continue;

                // Clear the charge from current node as it's moved to parents
                lambda[s * L * L + i] = 0;

                int r = i / L;
                int c = i % L;

                // get neighbors to re-determine parents
                float T_l = (c == 0) ? INF : T[s * L * L + i - 1];
                float T_r = (c == L - 1) ? INF : T[s * L * L + i + 1];
                float T_u = (r == 0) ? INF : T[s * L * L + i - L];
                float T_d = (r == L - 1) ? INF : T[s * L * L + i + L];

                float a = std::min(T_l, T_r);
                int p_a = (T_l < T_r) ? (i - 1) : (i + 1);

                float b = std::min(T_u, T_d);
                int p_b = (T_u < T_d) ? (i - L) : (i + L);

                float f_i = f[i];

                if (std::abs(a - b) >= f_i * h) {
                    // 1D update
                    if (a < b) {
                        lambda[s * L * L + p_a] += l_i;
                    } else {
                        lambda[s * L * L + p_b] += l_i;
                    }
#pragma omp atomic
                    grad_f[i] += l_i * h;
                } else {
                    // 2D update
                    float delta = 2.0f * f_i * f_i * h * h - (a - b) * (a - b);
                    if (delta > 0) {
                        float sqrt_delta = std::sqrt(delta);
                        float da = 0.5f * (1.0f - (a - b) / sqrt_delta);
                        float db = 0.5f * (1.0f + (a - b) / sqrt_delta);
                        float df = (f_i * h * h) / sqrt_delta;

                        lambda[s * L * L + p_a] += l_i * da;
                        lambda[s * L * L + p_b] += l_i * db;
#pragma omp atomic
                        grad_f[i] += l_i * df;
                    }
                }
            }
        }

        // Finalize grad_src
        float g_u = 0.0f;
        float g_v = 0.0f;
        for (int k = 0; k < num_corners; k++) {
            int idx = corners[k];
            int r = idx / L;
            int c = idx % L;
            float du = u - c;
            float dv = v - r;
            float dist_pixel = std::sqrt(du * du + dv * dv);
            if (dist_pixel > 1e-6f) {
                float l_corner = lambda[s * L * L + idx];
                g_u += l_corner * f[idx] * h * (du / dist_pixel);
                g_v += l_corner * f[idx] * h * (dv / dist_pixel);
            }
        }
        grad_src[s * 2 + 0] = g_u * (L - 1) / 2.0f;
        grad_src[s * 2 + 1] = g_v * (L - 1) / 2.0f;
    }

    delete[] lambda;
}
