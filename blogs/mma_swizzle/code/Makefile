include Makefile.paths

ARCH=sm_100a
CCBIN?=$(shell which g++)

CUTLASS_NVCC_FLAGS= \
		-I$(CUTLASS_HOME)/include \
		-I$(CUTLASS_HOME)/tools/util/include \
		-I$(CUTLASS_HOME)/tools/library/include \
		--expt-relaxed-constexpr \

all: mma_swizzle

mma_swizzle: mma_swizzle.cpp
	$(CUDA_HOME)/bin/nvcc -ccbin $(CCBIN) -arch $(ARCH) -g $(CUTLASS_NVCC_FLAGS) -lineinfo mma_swizzle.cpp -o $@

clean:
	rm -f mma_swizzle
