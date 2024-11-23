include Makefile.paths

CCBIN?=$(shell which g++)

CUTLASS_NVCC_FLAGS= \
	--generate-code=arch=compute_90a,code=[sm_90a] \
	-std=c++17 \
	-O3 \
	-Xcompiler=-Wno-psabi \
	-Xcompiler=-fno-strict-aliasing \
	-I${CUTLASS_HOME}/include \
	-I${CUTLASS_HOME}/tools/util/include \
	-isystem=$(CUDA_HOME)/include \
	--expt-relaxed-constexpr \

all : cute_tma

cute_tma: cute_tma.cu
	$(CUDA_HOME)/bin/nvcc -ccbin $(CCBIN) -g $(CUTLASS_NVCC_FLAGS) -lineinfo cute_tma.cu -o $@

.PHONY: clean

clean :
	@rm -f cute_tma
