#include <CL/sycl.hpp>
#include <array>
#include <sys/time.h>
using namespace sycl;

constexpr int N = 32;
constexpr int REPEAT = 100;

int main() {

    gpu_selector Selector;
    queue Q(Selector);

    // float *f = (float *)malloc(sizeof(float) * N);
    float *d = (float *)malloc(sizeof(float) * N);
    float *a = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = i;
        d[i] = 0.0;
    }

    cl::sycl::range<1> arr_range{N};
    buffer<float, 1> bufferD((float *)d, arr_range);
    buffer<float, 1> bufferA((float *)a, arr_range);

    buffer<float, 1> bufferT((float *)a, arr_range);

    auto start_time = std::chrono::steady_clock::now();

    // Submit our job to the queue
    Q.submit([&](cl::sycl::handler &cgh) {
        accessor accessorT(bufferT, cgh, read_only);
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

            auto gaPtr =
                reinterpret_cast<float_t *>(accessorA.get_pointer().get());
            auto gdPtr =
                reinterpret_cast<float_t *>(accessorD.get_pointer().get());

            auto aPtr =
                reinterpret_cast<float_t *>(localAccA.get_pointer().get());
            auto dPtr =
                reinterpret_cast<float_t *>(localAccD.get_pointer().get());
            for (int j = 0; j < N; ++j) {
                localAccA[j] = accessorA[j];
            }

            #ifdef __SYCL_DEVICE_ONLY__
            //__mlvm_memcpy_1D_gdram_to_nram(aPtr, gaPtr, N);
            for(int i=0; i<REPEAT; i++)
            __mlvm_stream_argmax_f32(dPtr, aPtr, N);
            //__mlvm_memcpy_1D_nram_to_gdram(gdPtr, dPtr, N);
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
