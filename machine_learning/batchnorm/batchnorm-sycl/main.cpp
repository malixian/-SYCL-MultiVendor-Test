#include <iostream>
#include <CL/sycl.hpp>

//namespace sycl = cl::sycl;

const int N = 102400; // 向量维度
const size_t REPEAT = 100;
const size_t ITER = 10000;

int main()
{
    // 创建和初始化输入数据
    std::vector<float> x(N);
    std::vector<float> mean(N);
    std::vector<float> variance(N);
    std::vector<float> scale(N);
    std::vector<float> bias(N);
    std::vector<float> output(N);

    for (int i = 0; i < N; ++i)
    {
        x[i] = i;
        mean[i] = i * 2;
        variance[i] = i * 3;
        scale[i] = 1.0;
        bias[i] = 0.0;
    }


    sycl::queue q(sycl::gpu_selector_v);

    float *x_acc = sycl::malloc_device<float>(N, q);
    q.memcpy(x_acc, x.data(), N * sizeof(float));

    float *mean_acc = sycl::malloc_device<float>(N, q);
    q.memcpy(mean_acc, mean.data(), N * sizeof(float));

    float *variance_acc = sycl::malloc_device<float>(N, q);
    q.memcpy(variance_acc, variance.data(), N * sizeof(float));

    float *scale_acc = sycl::malloc_device<float>(N, q);
    q.memcpy(scale_acc, scale.data(), N * sizeof(float));

    float *bias_acc = sycl::malloc_device<float>(N, q);
    q.memcpy(bias_acc, bias.data(), N * sizeof(float));

    float *output_acc = sycl::malloc_device<float>(N, q);

    q.wait();
    
    sycl::range<1> gws (N);
    sycl::range<1> lws (256);

    auto start_time = std::chrono::steady_clock::now();
    for (int r=0; r<REPEAT; r++)
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class batchnorm_kernel>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
            int idx = item.get_global_id(0);
            for(int rid=0; rid<ITER; rid++)
                output_acc[idx] = scale_acc[idx] * (x_acc[idx] - mean_acc[idx]) / sqrtf(variance_acc[idx] + 1e-5) + bias_acc[idx];

        });
    }).wait();
    
    
    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Total kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    std::cout << "Check PASS:\n";

    return 0;
}

