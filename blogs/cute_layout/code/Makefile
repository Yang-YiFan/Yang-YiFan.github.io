include Makefile.paths

ARCH=sm_100a
CCBIN?=$(shell which g++)

CUTLASS_NVCC_FLAGS= \
		-I$(CUTLASS_HOME)/include \
		-I$(CUTLASS_HOME)/tools/util/include \
		-I$(CUTLASS_HOME)/tools/library/include \
		--expt-relaxed-constexpr \

all: cute_layout

cute_layout: cute_layout.cpp
	$(CUDA_HOME)/bin/nvcc -ccbin $(CCBIN) -arch $(ARCH) -g $(CUTLASS_NVCC_FLAGS) -lineinfo cute_layout.cpp -o $@

clean:
	rm -f cute_layout
