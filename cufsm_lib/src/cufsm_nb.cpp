#include <iostream>

#include <nanobind/nanobind.h>

#include <nanobind/ndarray.h>

#include "fsm_cpu.h"
#include "fsm_gpu.cuh"

namespace nb = nanobind;
using namespace nb::literals;

int add(int a, int b) {
    std::cout << "Adding " << a << " and " << b << std::endl;
    return a + b;
}

void arr_add(const float *a, const float *b, float *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

NB_MODULE(cufsm_nb, m) {
    m.def("add", &add, "a"_a, "b"_a);

    m.def(
        "fsm_cpu",
        [](nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> T,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig>
               src,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig>
               f,
           int iters) {
            // check shapes (assume f is L x L)
            int N = (int)src.shape(0);
            int L = (int)f.shape(0);

            if (L != (int)f.shape(1)) {
                throw std::runtime_error("f (" + std::to_string(f.shape(0)) + ", " +
                                         std::to_string(f.shape(1)) +
                                         ") must be square!");
            }

            if (!(T.shape(0) == src.shape(0) && T.shape(1) == f.shape(0) &&
                  T.shape(2) == f.shape(1))) {
                throw std::runtime_error(
                    "T (" + std::to_string(T.shape(0)) + ", " +
                    std::to_string(T.shape(1)) + ", " + std::to_string(T.shape(2)) +
                    ") has incorrect shape! "
                    "Expected (" +
                    std::to_string(src.shape(0)) + ", " + std::to_string(L) + ", " +
                    std::to_string(L) + ").");
            }

            fsm_cpu(T.data(), src.data(), f.data(), N, L, iters);
        },
        "T"_a, "src"_a, "f"_a, "iters"_a);

    m.def(
        "fsm_adjoint_cpu",
        [](nb::ndarray<const float, nb::pytorch, nb::shape<-1, -1, -1>,
                       nb::c_contig>
               T,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, -1, -1>,
                       nb::c_contig>
               grad_T,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig>
               src,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig>
               f,
           nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig>
               grad_f,
           nb::ndarray<float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig>
               grad_src,
           int iters) {
            int N = (int)src.shape(0);
            int L = (int)f.shape(0);

            if (L != (int)f.shape(1)) {
                throw std::runtime_error("f (" + std::to_string(f.shape(0)) + ", " +
                                         std::to_string(f.shape(1)) +
                                         ") must be square!");
            }

            if (!(T.shape(0) == src.shape(0) && T.shape(1) == f.shape(0) &&
                  T.shape(2) == f.shape(1))) {
                throw std::runtime_error(
                    "T (" + std::to_string(T.shape(0)) + ", " +
                    std::to_string(T.shape(1)) + ", " + std::to_string(T.shape(2)) +
                    ") has incorrect shape! "
                    "Expected (" +
                    std::to_string(src.shape(0)) + ", " + std::to_string(L) + ", " +
                    std::to_string(L) + ").");
            }

            // Basic shape checks
            if (grad_f.shape(0) != L || grad_f.shape(1) != L) {
                throw std::runtime_error("grad_f shape mismatch");
            }

            if (grad_src.shape(0) != N || grad_src.shape(1) != 2) {
                throw std::runtime_error("grad_src shape mismatch");
            }

            fsm_adjoint_cpu(T.data(), grad_T.data(), src.data(), f.data(),
                            grad_f.data(), grad_src.data(), N, L, iters);
        },
        "T"_a, "grad_T"_a, "src"_a, "f"_a, "grad_f"_a, "grad_src"_a, "iters"_a);

    m.def(
        "fsm_gpu",
        [](nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig,
                       nb::device::cuda>
               T,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig,
                       nb::device::cuda>
               src,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig,
                       nb::device::cuda>
               f,
           int iters) {
            // check shapes (assume f is L x L)
            int N = (int)src.shape(0);
            int L = (int)f.shape(0);

            if (L != (int)f.shape(1)) {
                throw std::runtime_error("f (" + std::to_string(f.shape(0)) + ", " +
                                         std::to_string(f.shape(1)) +
                                         ") must be square!");
            }

            if (!(T.shape(0) == src.shape(0) && T.shape(1) == f.shape(0) &&
                  T.shape(2) == f.shape(1))) {
                throw std::runtime_error(
                    "T (" + std::to_string(T.shape(0)) + ", " +
                    std::to_string(T.shape(1)) + ", " + std::to_string(T.shape(2)) +
                    ") has incorrect shape! "
                    "Expected (" +
                    std::to_string(src.shape(0)) + ", " + std::to_string(L) + ", " +
                    std::to_string(L) + ").");
            }

            fsm_gpu(T.data(), src.data(), f.data(), N, L, iters);
        },
        "T"_a, "src"_a, "f"_a, "iters"_a);

    m.def(
        "fsm_adjoint_gpu",
        [](nb::ndarray<const float, nb::pytorch, nb::shape<-1, -1, -1>,
                       nb::c_contig, nb::device::cuda>
               T,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, -1, -1>,
                       nb::c_contig, nb::device::cuda>
               grad_T,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig,
                       nb::device::cuda>
               src,
           nb::ndarray<const float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig,
                       nb::device::cuda>
               f,
           nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>, nb::c_contig,
                       nb::device::cuda>
               grad_f,
           nb::ndarray<float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig,
                       nb::device::cuda>
               grad_src,
           int iters) {
            int N = (int)src.shape(0);
            int L = (int)f.shape(0);

            if (L != (int)f.shape(1)) {
                throw std::runtime_error("f (" + std::to_string(f.shape(0)) + ", " +
                                         std::to_string(f.shape(1)) +
                                         ") must be square!");
            }

            if (!(T.shape(0) == src.shape(0) && T.shape(1) == f.shape(0) &&
                  T.shape(2) == f.shape(1))) {
                throw std::runtime_error(
                    "T (" + std::to_string(T.shape(0)) + ", " +
                    std::to_string(T.shape(1)) + ", " + std::to_string(T.shape(2)) +
                    ") has incorrect shape! "
                    "Expected (" +
                    std::to_string(src.shape(0)) + ", " + std::to_string(L) + ", " +
                    std::to_string(L) + ").");
            }

            fsm_adjoint_gpu(T.data(), grad_T.data(), src.data(), f.data(),
                            grad_f.data(), grad_src.data(), N, L, iters);
        },
        "T"_a, "grad_T"_a, "src"_a, "f"_a, "grad_f"_a, "grad_src"_a, "iters"_a);
}