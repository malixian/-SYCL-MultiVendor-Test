#include <CL/sycl.hpp>
#include <array>
#include <sys/time.h>
using namespace sycl;

constexpr int N = 32;

constexpr int REPEAT = 100;

long long getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

int main() {

    gpu_selector Selector;
    queue Q(Selector);

    // float *f = (float *)malloc(sizeof(float) * N);
    float *d = (float *)malloc(sizeof(float) * N);
    float *a = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 0.5;
        d[i] = 0.0;
    }

    cl::sycl::range<1> arr_range{N};
    buffer<float, 1> bufferD((float *)d, arr_range);
    buffer<float, 1> bufferA((float *)a, arr_range);

    auto start_time = std::chrono::steady_clock::now();

    // Submit our job to the queue
    Q.submit([&](cl::sycl::handler &cgh) {
        //accessor accessorD(bufferD, cgh, read_only);
        accessor accessorA(bufferA, cgh, read_only);
        accessor accessorD(bufferD, cgh, write_only);
        // Local Accessor for NRAM
        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            localAccA(cl::sycl::range<1>(N), cgh);
        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            localAccD(cl::sycl::range<1>(N), cgh);

        cgh.parallel_for<class mm>(1, [=](id<1> i) {
            auto aPtr =
                reinterpret_cast<float_t *>(localAccA.get_pointer().get());
            auto dPtr =
                reinterpret_cast<float_t *>(localAccD.get_pointer().get());
            for (int j = 0; j < N; ++j) {
                localAccA[j] = accessorA[j];
                localAccD[j] = accessorD[j];
            }

            #ifdef __SYCL_DEVICE_ONLY__
            for(int i=0; i<REPEAT; i++)
            __mlvm_stream_active_relu_f32(dPtr, aPtr, N);
            #endif

            for (int j = 0; j < N; ++j) {
                accessorD[j] = localAccD[j];
            }
        });
    });

    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    host_accessor host_accD(bufferD, read_only);

    return 0;
}
