#include <iostream>

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#define gpuErrChk(ans) { gpuAssert2((ans), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

using namespace cute;

int main() {
    Layout smem_layout = make_layout(make_shape(_4{}, _6{}), make_stride(_6{}, _1{}));
    printf("smem_layout: ");
    print_layout(smem_layout);
    printf("\n");

    Layout tile_layout = make_layout(make_shape(_4{}), make_stride(_1{}));
    Layout repetition_layout1 = make_layout(make_shape(_6{}), make_stride(_1{}));
    Layout reshape_layout1 = logical_product(tile_layout, repetition_layout1);
    // result = composition(lhs, rhs)
    // -> result(c) = lhs(rhs(c))
    Layout final_layout1 = composition(smem_layout, reshape_layout1);

    printf("tile_layout: ");
    print(tile_layout);
    printf("\n");
    printf("repetition_layout1: ");
    print(repetition_layout1);
    printf("\n");
    printf("reshape_layout1: ");
    print_layout(reshape_layout1);
    printf("\n");
    printf("final_layout1: ");
    print_layout(final_layout1);
    printf("\n");

    Layout repetition_layout2 = make_layout(make_shape(_2{}, _3{}), make_stride(_3{}, _1{}));
    Layout reshape_layout2 = logical_product(tile_layout, repetition_layout2);
    Layout final_layout2 = composition(smem_layout, reshape_layout2);

    printf("tile_layout: ");
    print(tile_layout);
    printf("\n");
    printf("repetition_layout2: ");
    print(repetition_layout2);
    printf("\n");
    printf("reshape_layout2: ");
    print_layout(reshape_layout2);
    printf("\n");
    printf("final_layout2: ");
    print_layout(final_layout2);
    printf("\n");

    return 0;
}