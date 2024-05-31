#include <CL/sycl.hpp>
#include <array>
#include <sys/time.h>
#include <iostream>
using namespace std;
using namespace sycl;

#define M 256
#define N 256
#define K 256

#define INPUT_SIZE_1 M * K
#define INPUT_SIZE_2 K * N
#define OUTPUT_SIZE M * N

constexpr int REPEAT = 100;

int main() {

    gpu_selector Selector;
    queue Q(Selector);

    float *d = (float *)malloc(sizeof(float) * OUTPUT_SIZE);
    float *a = (float *)malloc(sizeof(float) * INPUT_SIZE_1);
    float *b = (float *)malloc(sizeof(float) * INPUT_SIZE_2);

    for (int i = 0; i < INPUT_SIZE_1; i++) {
        a[i] = 1;
    }

    for (int i = 0; i < INPUT_SIZE_2; i++) {
        b[i] = 1;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d[i] = 1;
    }

    cl::sycl::range<1> in_range_1{INPUT_SIZE_1};
    cl::sycl::range<1> in_range_2{INPUT_SIZE_2};
    cl::sycl::range<1> out_range{OUTPUT_SIZE};

    buffer<float, 1> bufferA((float *)a, in_range_1);
    buffer<float, 1> bufferB((float *)b, in_range_2);

    buffer<float, 1> bufferT((float *)d, out_range);
    buffer<float, 1> bufferD((float *)d, out_range);

    auto start_time = std::chrono::steady_clock::now();

    // Submit our job to the queue
    Q.submit([&](cl::sycl::handler &cgh) {
       accessor accessorT(bufferT, cgh, read_only);
       accessor accessorA(bufferA, cgh, read_only);
       accessor accessorB(bufferB, cgh, read_only);
       accessor accessorD(bufferD, cgh, write_only);

        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            localAccA(cl::sycl::range<1>(INPUT_SIZE_1), cgh);

        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::wram>
            localAccB(cl::sycl::range<1>(INPUT_SIZE_2), cgh);

        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            localAccC(cl::sycl::range<1>(OUTPUT_SIZE), cgh);
        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                            cl::sycl::access::target::local>
            localAccD(cl::sycl::range<1>(OUTPUT_SIZE), cgh);

        cgh.parallel_for<class mm>(1, [=](id<1> i) {

            auto laPtr =
                reinterpret_cast<float *>(localAccA.get_pointer().get());
            auto lbPtr =
                reinterpret_cast<float *>(localAccB.get_pointer().get());
            auto ldPtr =
                reinterpret_cast<float *>(localAccD.get_pointer().get());

            /* for (int j = 0; j < INPUT_SIZE_1; ++j) {
                localAccA[j] = accessorA[j];
            } */

            #ifdef __SYCL_DEVICE_ONLY__
            for(int i=0; i<REPEAT; i++)
            //__mlvm_memcpy_nram_to_wram(lbPtr, laPtr, INPUT_SIZE_2*sizeof(short));
            __mlvm_stream_conv_dilation_f32_f32_f32(ldPtr, laPtr, lbPtr, K, M,
                                          1, 1, 1, 1, 1, N, 1, 1, 1, 1);
            #endif

            /* for (int j = 0; j < OUTPUT_SIZE; ++j) {
                accessorD[j] = localAccD[j];
            } */


        });
    });

    host_accessor host_accD(bufferD, read_only);
    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);

    return 0;
}
