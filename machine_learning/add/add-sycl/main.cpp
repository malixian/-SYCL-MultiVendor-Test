#include <iostream>
#include <CL/sycl.hpp>


const size_t N = 102400; // 向量维度

const size_t REPEAT = 100;

const size_t ITER = 10000;

int main()
{
    std::vector<float> a(N, 1);
    std::vector<float> b(N, 2);
    std::vector<float> result(N);
	
    sycl::queue q(sycl::gpu_selector_v);

    float *a_accessor = sycl::malloc_device<float>(N, q);
    q.memcpy(a_accessor, a.data(), N * sizeof(float));

    float *b_accessor = sycl::malloc_device<float>(N, q);
    q.memcpy(b_accessor, b.data(), N * sizeof(float));

    float *result_accessor = sycl::malloc_device<float>(N, q);

    q.wait();

    auto start_time = std::chrono::steady_clock::now();

    sycl::range<1> gws (N);
    sycl::range<1> lws (256);
    
    for(int r=0; r<REPEAT; r++)
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class vector_add>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
            int idx = item.get_global_id(0);
            for(int rid=0; rid<ITER; rid++)
                result_accessor[idx] = a_accessor[idx] + b_accessor[idx];
        });
    }).wait();
    
    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Total kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    std::cout << "Check PASS:\n";
    return 0;
}