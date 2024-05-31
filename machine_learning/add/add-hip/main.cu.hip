#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

const size_t REPEAT = 100;

const size_t ITER = 10000;

#define CHECK(call) \
{ \
    const hipError_t error = call; \
    if (error != hipSuccess) { \
        std::cerr << "Error: " << __FILE__ << ", line " << __LINE__ << ": " \
                  << hipGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// 向量加法的CUDA核函数
__global__ void addVectors(float* c, const float* a, const float* b, int n) {
    for(int rid=0; rid<ITER; rid++) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
    
    
    
}

int main() {
    const int n = 102400; // 向量长度
    float h_a[n], h_b[n], h_c[n]; // 主机端向量（h_表示host）
    float *d_a, *d_b, *d_c; // 设备端向量（d_表示device）

    // 初始化主机端向量
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 分配GPU内存
    CHECK(hipMalloc((void**)&d_a, n * sizeof(float)));
    CHECK(hipMalloc((void**)&d_b, n * sizeof(float)));
    CHECK(hipMalloc((void**)&d_c, n * sizeof(float)));

    //auto start_time = std::chrono::steady_clock::now();
    // 将数据从主机复制到设备
    CHECK(hipMemcpy(d_a, h_a, n * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, h_b, n * sizeof(float), hipMemcpyHostToDevice));

    // 配置并启动核函数
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    auto start_time = std::chrono::steady_clock::now();
    
    for(int i=0; i<REPEAT; i++)
        addVectors<<<numBlocks, blockSize>>>(d_c, d_a, d_b, n);

    // 同步以确保核函数执行完毕
    CHECK(hipDeviceSynchronize());
    
    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Total kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    // 将结果从设备复制回主机
    CHECK(hipMemcpy(h_c, d_c, n * sizeof(float), hipMemcpyDeviceToHost));

    // 释放GPU内存
    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_c));

    // 打印结果
    std::cout << "Check PASS:\n";

    return 0;
}
