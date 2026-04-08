#include <iostream>

#include "kernel.cuh"

int main() {
    std::cout << "Hello, CUDA Test!" << std::endl;


    int a[5] = {1, 2, 3, 4, 5};
    int b[5] = {10, 20, 30, 40, 50};
    int c[5] = {0};

    cuda_add(a, b, c, 5);

    std::cout << "Result: ";
    for (int i = 0; i < 5; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}