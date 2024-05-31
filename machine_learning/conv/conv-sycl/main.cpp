#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace cl::sycl;

const size_t width = 10240;
const size_t height = 10240;
const int kernelSize = 3;
const int REPEAT = 100;


int main() {
    std::vector<float> input_data(width * height, 1.0f);
    std::vector<float> kernel_data(kernelSize * kernelSize, 1.0f);
    std::vector<float> output_data(width * height);

    sycl::queue q(sycl::gpu_selector_v);

    float *input_acc = sycl::malloc_device<float>(width * height, q);
    q.memcpy(input_acc, input_data.data(), width * height * sizeof(float));

    float *kernel_acc = sycl::malloc_device<float>(kernelSize * kernelSize, q);
    q.memcpy(kernel_acc, kernel_data.data(), kernelSize * kernelSize * sizeof(float));

    float *output_acc = sycl::malloc_device<float>(width * height, q);

    q.wait();

    auto start_time = std::chrono::steady_clock::now();
    
    auto global_size = sycl::range<2>(width, height);
    auto local_size = sycl::range<2>(32, 32);

    for(int rid=0; rid<REPEAT; rid++)
    q.submit([&](handler &h) {
        h.parallel_for<class conv2dKernelFunc>(nd_range<2>(global_size, local_size), [=](nd_item<2> it) {
            int col = it.get_global_id(1); 
            int row = it.get_global_id(0); 
            
            float sum = 0.0f;
            for (int kRow = 0; kRow < kernelSize; ++kRow) {
                for (int kCol = 0; kCol < kernelSize; ++kCol) {
                    int inputRow = row + kRow - kernelSize / 2;
                    int inputCol = col + kCol - kernelSize / 2;
                    if (inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width)
                        sum += input_acc[inputRow * width + inputCol] * kernel_acc[kRow * kernelSize + kCol];
                }
            }
            output_acc[row * width + col] = sum;
        });
    }).wait();

    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Total kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    return 0;
}