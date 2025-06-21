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
    // simple 2d matrix layout
    Layout row_major = make_layout(make_shape(2, 4), make_stride(4, 1));
    Layout col_major = make_layout(make_shape(2, 4), make_stride(1, 2));

    print_layout(row_major);
    print_layout(col_major);

    printf("row_major == col_major: ");
    print(row_major == col_major);
    printf("\n");

    // more complex layout
    auto M0 = _2{};
    auto N0 = _2{};
    auto M1 = _2{};
    auto N1 = _3{};
    Layout tile_layout = make_layout(make_shape(M0, N0), make_stride(_1{}, M0));
    Layout tiler_layout = make_layout(make_shape(M1, N1), make_stride(N1, _1{}));
    Layout final_layout = blocked_product(tile_layout, tiler_layout);
    // manually construct the layout
    Layout manual_layout = make_layout(make_shape(make_shape(M0, M1), make_shape(N0, N1)), make_stride(make_stride(_1{}, N1*M0*N0), make_stride(M0, M0*N0)));
    // have to use by-mode coalesce because we still want to preserve the 2d shape of the layout, i.e. (M, N) : (xxx, xxx)
    // Step<_1, _1> here means we want the resulting layout to have rank 2, because Step<_1, _1> has rank 2
    // the value inside Step<> doesn't matter, only the rank of the Step<> matters
    Layout manual_layout2 = coalesce(manual_layout, Step<_1, _1>{});

    print_layout(final_layout);
    print_layout(manual_layout);
    print_layout(manual_layout2);
    printf("final_layout == manual_layout: ");
    print(final_layout == manual_layout);
    printf("\n");
    printf("final_layout == manual_layout2: ");
    print(final_layout == manual_layout2);
    printf("\n");

    // test the shape tuple
    auto tiler_layout2 = make_shape(_4{}, _2{});
    auto tiler_layout3 = make_tile(Layout<_4, _1>{}, Layout<_2, _1>{});
    printf("tiler_layout2: \n");
    print(tiler_layout2);
    printf("\ntiler_layout3: \n");
    print(tiler_layout3);
    printf("\n");
    print(shape(tile_layout));
    
    // Allocate device memory using cudaMalloc
    using ElementType = int; // Choose your data type
    size_t total_elements = cosize(manual_layout); // Get total number of elements from the layout
    printf("total_elements: %zu\n", total_elements);
    size_t bytes = total_elements * sizeof(ElementType);
    ElementType* data = nullptr;
    gpuErrChk(cudaMalloc(&data, bytes));
    // Create a gmem tensor with the allocated memory
    auto gmem_tensor = make_tensor(make_gmem_ptr(data), manual_layout);

    // try to local_tile the layout
    //auto local_tile_tensor = local_tile(gmem_tensor, shape(tile_layout), make_coord(0, 0));
    auto local_tile_tensor = local_tile(gmem_tensor, tiler_layout2, make_coord(0, 1));
    printf("local_tile_layout: \n");
    print_layout(local_tile_tensor.layout());
    print(local_tile_tensor);
    
    // Free the allocated memory
    cudaFree(data);
    
    return 0;
}