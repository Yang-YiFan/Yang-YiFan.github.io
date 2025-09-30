#include <iostream>

#include <cute/tensor.hpp>
#include <cute/util/print_svg.hpp>

using namespace cute;

int main() {
    Layout tv_layout = make_layout(make_shape(make_shape(_4{}, _8{}), make_shape(_2{}, _2{})), make_stride(make_stride(_32{}, _1{}), make_stride(_16{}, _8{}))); // (T, V) -> (M, K)
    Layout a_layout = right_inverse(tv_layout); // (M, K) -> (T, V)
    
    printf("tv_layout: ");
    print(tv_layout);

    printf("\na_layout: ");
    print(a_layout);
    printf("\n");

    TiledMMA tiled_mma = make_tiled_mma(SM80_16x8x8_F32BF16BF16F32_TN{});
    print_svg(tiled_mma);

    return 0;
}