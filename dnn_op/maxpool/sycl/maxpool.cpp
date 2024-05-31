#include <CL/sycl.hpp>
#include <array>
#include <sys/time.h>
#include <iostream>
using namespace std;
using namespace sycl;

#define CHANNELS 64
#define HEIGHT 4
#define WIDTH 4
#define KERNEL_H 2
#define KERNEL_W 2
#define INPUT_SIZE CHANNELS * WIDTH * HEIGHT
#define OUTPUT_SIZE (CHANNELS * (HEIGHT / KERNEL_H) * (WIDTH / KERNEL_W))
#define STRIDE_H 1
#define STRIDE_W 1

constexpr int REPEAT = 100;

int main() {

    gpu_selector Selector;
    queue Q(Selector);

    float *d = (float *)malloc(sizeof(float) * OUTPUT_SIZE);
    float *a = (float *)malloc(sizeof(float) * INPUT_SIZE);

    for (int i = 0; i < INPUT_SIZE; i++) {
        a[i] = 0.5;
    }

    cl::sycl::range<1> in_range{INPUT_SIZE};
    cl::sycl::range<1> out_range{OUTPUT_SIZE};

    buffer<float, 1> bufferA((float *)a, in_range);
    buffer<float, 1> bufferD((float *)d, out_range);

    auto start_time = std::chrono::steady_clock::now();

    // Submit our job to the queue
    Q.submit([&](cl::sycl::handler &cgh) {
       accessor accessorA(bufferA, cgh, read_only);
       accessor accessorD(bufferD, cgh, write_only);

       cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            localAccA(cl::sycl::range<1>(INPUT_SIZE), cgh);

        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                            cl::sycl::access::target::local>
            localAccD(cl::sycl::range<1>(OUTPUT_SIZE), cgh);

        cgh.parallel_for<class mm>(1, [=](id<1> i) {
            auto gaPtr =
                reinterpret_cast<float *>(accessorA.get_pointer().get());
            auto gdPtr =
                reinterpret_cast<float *>(accessorD.get_pointer().get());

            auto laPtr =
                reinterpret_cast<float *>(localAccA.get_pointer().get());
            auto ldPtr =
                reinterpret_cast<float *>(localAccD.get_pointer().get());

            /* for (int j = 0; j < INPUT_SIZE; ++j) {
                localAccA[j] = accessorA[j];
            } */

            #ifdef __SYCL_DEVICE_ONLY__
            //__mlvm_memcpy_1D_gdram_to_nram(laPtr, gaPtr, INPUT_SIZE);
            for(int i=0; i<REPEAT; i++)
            __mlvm_stream_pool_max_f32(ldPtr, laPtr, CHANNELS, HEIGHT, WIDTH, KERNEL_H, KERNEL_W, STRIDE_H, STRIDE_W);
            //__mlvm_memcpy_1D_nram_to_gdram(gdPtr, ldPtr, OUTPUT_SIZE);
            #endif

            /* for (int j = 0; j < OUTPUT_SIZE; ++j) {
                accessorD[j] = localAccD[j];
            } */
        });
    });

    /* host_accessor host_accD(bufferD, read_only);
    for(int i=0; i<OUTPUT_SIZE; i++){
                std::cout<<host_accD[i]<<std::endl;
    } */

    auto end_time = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    printf("Kernel execution time %f (ms)\n", time / REPEAT * 1e-6f);
    return 0;
}
