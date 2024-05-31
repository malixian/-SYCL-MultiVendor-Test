#include "hip/hip_runtime.h"
#include <iostream>
#include <vector>
#include <chrono>


const size_t matrixSize = 1024;
const int REPEAT = 100;


__global__ void matrixMultiplication(float *matrixA, float *matrixB, float *result)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < matrixSize; ++i) {
        sum += matrixA[row * matrixSize + i] * matrixB[i * matrixSize + col];
    }
    result[row * matrixSize + col] = sum;
    
}

int main() {
    std::vector<float> matrixA(matrixSize * matrixSize, 1.0f);
    std::vector<float> matrixB(matrixSize * matrixSize, 1.0f);
    std::vector<float> result(matrixSize * matrixSize);

    float *d_matrixA, *d_matrixB, *d_result;
    hipMalloc(&d_matrixA, matrixSize * matrixSize * sizeof(float));
    hipMalloc(&d_matrixB, matrixSize * matrixSize * sizeof(float));
    hipMalloc(&d_result, matrixSize * matrixSize * sizeof(float));

    auto start_time = std::chrono::steady_clock::now();
    hipMemcpy(d_matrixA, matrixA.data(), matrixSize * matrixSize * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_matrixB, matrixB.data(), matrixSize * matrixSize * sizeof(float), hipMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((matrixSize + blockSize.x - 1) / blockSize.x, (matrixSize + blockSize.y - 1) / blockSize.y);

    
    for(int rid=0; rid<REPEAT; rid++)
        matrixMultiplication<<<gridSize, blockSize>>>(d_matrixA, d_matrixB, d_result);
    hipDeviceSynchronize();

    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Total kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    hipMemcpy(result.data(), d_result, matrixSize * matrixSize * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_matrixA);
    hipFree(d_matrixB);
    hipFree(d_result);

    return 0;
}

