// 用于测试使用向量加的执行时间

#include <CL/sycl.hpp>
#include <array>
#include <sys/time.h>
using namespace sycl;

constexpr int REPEAT = 100;
constexpr int N = 1024;


int main() {

    gpu_selector Selector;
    queue Q(Selector);

    // float *f = (float *)malloc(sizeof(float) * N);
    float *d = (float *)malloc(sizeof(float) * N);
    float *a = (float *)malloc(sizeof(float) * N);
    float *b = (float *)malloc(sizeof(float) * N);
    float *c = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 0.5;
        b[i] = 2.0;
        c[i] = 0;
    }

    cl::sycl::range<1> arr_range{N};

    // buffer<float, 1> bufferF((float *)f, arr_range);
    buffer<float, 1> bufferD((float *)d, arr_range);
    buffer<float, 1> bufferA((float *)a, arr_range);
    buffer<float, 1> bufferB((float *)b, arr_range);
    buffer<float, 1> bufferC((float *)c, arr_range);

    auto start_time = std::chrono::steady_clock::now();

    // Submit our job to the queue
    Q.submit([&](cl::sycl::handler &cgh) {
        //accessor accessorD(bufferD, cgh, read_only);
        accessor accessorA(bufferA, cgh, read_only);
        accessor accessorB(bufferB, cgh, read_only);
        accessor accessorC(bufferC, cgh, write_only);
        // Local Accessor for NRAM
        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            localAccA(cl::sycl::range<1>(N), cgh);
        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            localAccB(cl::sycl::range<1>(N), cgh);
        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            localAccC(cl::sycl::range<1>(N), cgh);

        cgh.parallel_for<class mm>(1, [=](id<1> i) {
            auto aPtr =
                reinterpret_cast<float_t *>(localAccA.get_pointer().get());
            auto bPtr =
                reinterpret_cast<float_t *>(localAccB.get_pointer().get());
            auto cPtr =
                reinterpret_cast<float_t *>(localAccC.get_pointer().get());
            for (int j = 0; j < N; ++j) {
                localAccA[j] = accessorA[j];
                localAccB[j] = accessorB[j];
            }

            for (int k = 0; k < REPEAT; ++k) {
#ifdef __SYCL_DEVICE_ONLY__
                __mlvm_stream_mul_f32(cPtr, aPtr, bPtr, N);
#endif
            }

            for (int j = 0; j < N; ++j) {
                accessorC[j] = localAccC[j];
                accessorC[j] = accessorA[j];
            } 
        });
    });

    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    printf("PASSED\n");
    return 0;
}