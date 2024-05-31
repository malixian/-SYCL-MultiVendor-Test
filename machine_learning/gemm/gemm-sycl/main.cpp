#include <iostream>
#include <chrono>
#include <CL/sycl.hpp>


using namespace cl::sycl;

const size_t matrixSize = 1024;
const int REPEAT = 100;


int main() {
    std::vector<float> matrixA(matrixSize * matrixSize, 1.0f);
    std::vector<float> matrixB(matrixSize * matrixSize, 1.0f);
    std::vector<float> result(matrixSize * matrixSize);

    
    sycl::queue q(sycl::gpu_selector_v);

    float *a = sycl::malloc_device<float>(matrixSize * matrixSize, q);
    q.memcpy(a, matrixA.data(), matrixSize * matrixSize * sizeof(float));

    float *b = sycl::malloc_device<float>(matrixSize * matrixSize, q);
    q.memcpy(b, matrixB.data(), matrixSize * matrixSize * sizeof(float));

    float *c = sycl::malloc_device<float>(matrixSize * matrixSize, q);

    q.wait();

    auto global_size = sycl::range<2>(matrixSize, matrixSize);
    auto local_size = sycl::range<2>(16, 16);

    auto start_time = std::chrono::steady_clock::now();
    
    for(int rid=0; rid<REPEAT; rid++)
    q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class matrixMultiplication>(nd_range<2>(global_size, local_size), [=](nd_item<2> it) {
        int col = it.get_global_id(1); 
        int row = it.get_global_id(0); 
        
        float sum = 0.0f;
        for (int i = 0; i < matrixSize; ++i) {
            sum += a[row * matrixSize + i] * b[i * matrixSize + col];
        }
        c[row * matrixSize + col] = sum;
        
    });}).wait();

    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Total kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    return 0;
}
