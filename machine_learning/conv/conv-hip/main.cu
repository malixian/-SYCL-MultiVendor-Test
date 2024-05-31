#include <iostream>
#include <vector>
#include <chrono>

const size_t width = 10240;
const size_t height = 10240;
const int kernelSize = 3;
const int REPEAT = 100;

__global__ void conv2d(float *output, float *input, float *kernel)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int sum = 0;
        for (int kRow = 0; kRow < kernelSize; ++kRow) {
            for (int kCol = 0; kCol < kernelSize; ++kCol) {
                int inputRow = row + kRow - kernelSize / 2;
                int inputCol = col + kCol - kernelSize / 2;
                if (inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width)
                    sum += input[inputRow * width + inputCol] * kernel[kRow * kernelSize + kCol];
            }
        }
        output[row * width + col] = sum;
    }
}

int main()
{
    // 创建输入、输出和卷积核数据
    std::vector<float> input_data(width * height, 1.0f);
    std::vector<float> kernel_data(kernelSize * kernelSize, 1.0f);
    std::vector<float> output_data(width * height);

    // 将数据从主机内存拷贝到设备内存
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, input_data.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel_data.data(), kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    auto start_time = std::chrono::steady_clock::now();

    // 定义 CUDA 线程块和网格大小
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 调用 CUDA kernel 进行卷积操作
    for(int rid=0; rid<REPEAT; rid++)
        conv2d<<<gridSize, blockSize>>>(d_output, d_input, d_kernel);
    cudaDeviceSynchronize();

    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Total kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    // 将结果从设备内存拷贝回主机内存
    cudaMemcpy(output_data.data(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}

