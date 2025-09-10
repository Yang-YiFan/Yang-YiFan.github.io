#CUDA_BUILD=cuda-35361325
#CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/$(CUDA_BUILD)
CUDA_HOME=/home/yifany-scratch/cuda-gpgpu-35361325
CUTLASS_HOME?=/home/yifany-scratch/dynamic-kernel-generator/cutlass

ARCH=sm_100a
CCBIN?=$(shell which g++)

CUTLASS_NVCC_FLAGS= \
		-I$(CUTLASS_HOME)/include \
		-I$(CUTLASS_HOME)/tools/util/include \
		-I$(CUTLASS_HOME)/tools/library/include \
		-DCUTLASS_FORCE_VECTOR_MBARRIER_INIT=1 \
		--expt-relaxed-constexpr \
		-uumn \

all: cga_reduction

cga_reduction: cga_reduction.cu cga_reduction.cuh
	$(CUDA_HOME)/bin/nvcc -ccbin $(CCBIN) -arch $(ARCH) -g $(CUTLASS_NVCC_FLAGS) -lineinfo cga_reduction.cu -o $@

dump_sass:
	$(CUDA_HOME)/bin/cuobjdump -xelf all cga_reduction
	$(CUDA_HOME)/bin/nvdisasm -gi -c cga_reduction.sm_100.cubin > cga_reduction.sass

clean:
	rm -f cga_reduction
