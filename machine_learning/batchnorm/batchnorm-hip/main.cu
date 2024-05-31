#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const int N = 102400; // 向量维度
const size_t REPEAT = 100;
const size_t ITER = 10000;

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ", line " << __LINE__ << ": " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

__global__ void batchnorm(float *x, float *mean, float *variance, float *scale, float *bias, float *output)
{
    for(int rid=0; rid<ITER; rid++) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < N)
        {
            output[tid] = scale[tid] * (x[tid] - mean[tid]) / sqrtf(variance[tid] + 1e-5) + bias[tid];
        }
    }
}

int main()
{

    float h_x[N], h_mean[N], h_variance[N], h_scale[N], h_bias[N], h_output[N]; // 主机端向量（h_表示host）
    float *x, *mean, *variance, *scale, *bias, *output;

    for (int i = 0; i < N; ++i)
    {
        h_x[i] = i;
        h_mean[i] = i * 2;
        h_variance[i] = i * 3;
        h_scale[i] = 1.0;
        h_bias[i] = 0.0;
    
    }
    
    

    CHECK(cudaMalloc((void**)&x, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mean, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&variance, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&scale, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&bias, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&output, N * sizeof(float)));

    

    // 将数据从主机复制到设备
    CHECK(cudaMemcpy(x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mean, h_mean, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(variance, h_variance, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(scale, h_scale, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(bias, h_bias, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    auto start_time = std::chrono::steady_clock::now();
    float kernel_milliseconds = 0;

    for(int r=0; r<REPEAT; r++)
    batchnorm<<<numBlocks, threadsPerBlock>>>(x, mean, variance, scale, bias, output);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Total kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    kernel_milliseconds += milliseconds;

    cudaFree(x);
    cudaFree(mean);
    cudaFree(variance);
    cudaFree(scale);
    cudaFree(bias);
    cudaFree(output);
    
    
    

    return 0;
}

