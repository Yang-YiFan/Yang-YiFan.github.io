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
		--expt-relaxed-constexpr \
		-uumn \

all: cute_layout

cute_layout: cute_layout.cpp
	$(CUDA_HOME)/bin/nvcc -ccbin $(CCBIN) -arch $(ARCH) -g $(CUTLASS_NVCC_FLAGS) -lineinfo cute_layout.cpp -o $@

clean:
	rm -f cute_layout
