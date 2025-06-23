#include <cuda_runtime.h>
#include <iostream>

#define gpuErrChk(ans) { gpuAssert2((ans), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void delay_kernel() {
    uint64_t timeout = 1000000000ULL; // 10e9 cycles, ~0.5s

    // prolog, will be overlapped with the previous kernel's mainloop2
    auto start = clock64();
    while (clock64() - start < timeout);

    asm volatile("griddepcontrol.wait;"); // block until the previous kernel's output is ready

    // mainloop1
    start = clock64();
    while (clock64() - start < timeout);

    asm volatile("griddepcontrol.launch_dependents;"); // launch the next kernel here

    // mainloop2, will be overlapped with the next kernel's prolog
    start = clock64();
    while (clock64() - start < timeout);
}

void launch_delay_kernel(cudaStream_t stream, bool pdl) {
    // enable pdl in kernel launch attributes
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl;

    // set kernel launch configuration
    cudaLaunchConfig_t config;
    config.gridDim = 1;
    config.blockDim = 32;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 1;

    // launch the kernel
    cudaLaunchKernelEx(&config, delay_kernel);
}

float benchmark_pdl(cudaStream_t stream, bool pdl) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));

    // Record start time and launch kernels
    gpuErrChk(cudaEventRecord(start, stream));
    
    launch_delay_kernel(stream, pdl);  // First kernel
    launch_delay_kernel(stream, pdl);  // Second kernel
    
    gpuErrChk(cudaEventRecord(stop, stream));
    gpuErrChk(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float elapsed_time;
    gpuErrChk(cudaEventElapsedTime(&elapsed_time, start, stop));
    
    // Cleanup events
    gpuErrChk(cudaEventDestroy(start));
    gpuErrChk(cudaEventDestroy(stop));
    
    return elapsed_time;
}

int main() {
    cudaStream_t stream;
    gpuErrChk(cudaStreamCreate(&stream));

    std::cout << "Testing PDL (Programmatic Dependent Launch) performance...\n\n";

    // Test with PDL enabled
    std::cout << "Launching two kernels with PDL enabled...\n";
    float pdl_time = benchmark_pdl(stream, true);
    std::cout << "Time with PDL: " << pdl_time << " ms\n\n";

    // Test with PDL disabled
    std::cout << "Launching two kernels with PDL disabled...\n";
    float no_pdl_time = benchmark_pdl(stream, false);
    std::cout << "Time without PDL: " << no_pdl_time << " ms\n\n";

    // Print comparison
    std::cout << "Performance comparison:\n";
    std::cout << "PDL enabled:  " << pdl_time << " ms\n";
    std::cout << "PDL disabled: " << no_pdl_time << " ms\n";
    std::cout << "Difference:   " << (no_pdl_time - pdl_time) << " ms\n";
    std::cout << "Speedup:      " << (no_pdl_time / pdl_time) << "x\n";

    // Cleanup
    gpuErrChk(cudaStreamDestroy(stream));

    return 0;
}