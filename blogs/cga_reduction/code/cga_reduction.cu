#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

// Use Thrust to handle host/device allocations
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cga_reduction.cuh"

using namespace cute;

template <class Tensor>
void
initialize_tensor(Tensor& tensor, cute::tuple<int, int> value_range = {-2, 2})
{
    using DataType = typename Tensor::element_type;
    auto [min, max] = value_range;
    for (int i = 0; i < cute::size(tensor); i++) {
        tensor(i) = DataType(int((max-min)*(rand() / double(RAND_MAX)) + min));
    }
}

// compare two tensors
template <class Tensor1, class Tensor2>
bool
compare_tensors(Tensor1 const& tensor1, Tensor2 const& tensor2, float epsilon = 1e-6)
{
    for (int i = 0; i < cute::size(tensor1); i++) {
        if (fabs((tensor1(i) - tensor2(i)) / tensor1(i)) > epsilon) {
            return false;
        }
    }
    return true;
}


template <class AccTensor, class ResultTensor>
void
cga_reduction_reference(
    AccTensor const& acc_tensor,
    ResultTensor const& result_tensor
)
{
    using DataType = typename AccTensor::element_type;

    for (int i = 0; i < cute::size(result_tensor); i++) {
        DataType sum = 0.0;
        for (int j = 0; j < cute::size<0>(acc_tensor); j++) {
            sum += acc_tensor(j, i);
        }
        result_tensor(i) = sum;
    }
}


int main() {
    int M = 1024;
    int constexpr NUM_SPLITS = 4;
    int constexpr CTA_M = 256;
    int constexpr NUM_THREADS = 32;

    using T = float;

    srand(time(NULL));

    thrust::host_vector<T> h_acc(M * NUM_SPLITS);
    thrust::host_vector<T> h_result(M);
    thrust::host_vector<T> h_result_ref(M);

    thrust::device_vector<T> d_acc(M * NUM_SPLITS);
    thrust::device_vector<T> d_result(M);

    // create tensor views
    Tensor h_acc_tensor = make_tensor(h_acc.data(), make_layout_gAcc<NUM_SPLITS>(M));
    Tensor h_result_tensor = make_tensor(h_result.data(), make_layout_gResult(M));
    Tensor h_result_ref_tensor = make_tensor(h_result_ref.data(), make_layout_gResult(M));

    // initialize tensors
    initialize_tensor(h_acc_tensor, {-4, 4});
    
    // copy input to device
    d_acc = h_acc;
    gpuErrChk(cudaDeviceSynchronize());

    cga_reduction_host<T, CTA_M, NUM_SPLITS, NUM_THREADS>(d_acc.data().get(), d_result.data().get(), M, false);

    // copy result to host
    h_result = d_result;
    gpuErrChk(cudaDeviceSynchronize());

    cga_reduction_reference(h_acc_tensor, h_result_ref_tensor);

    // compare result
    bool is_correct = compare_tensors(h_result_tensor, h_result_ref_tensor);
    printf("is_correct: %d\n", is_correct);

    //printf("h_acc_tensor: \n"); print_tensor(h_acc_tensor); printf("\n");
    //printf("h_result_tensor: \n"); print_tensor(h_result_tensor); printf("\n");
    //printf("h_result_ref_tensor: \n"); print_tensor(h_result_ref_tensor); printf("\n");

    return 0;
}
