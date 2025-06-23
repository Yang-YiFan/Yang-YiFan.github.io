include Makefile.paths

ARCH=sm_100a
CCBIN?=$(shell which g++)

all: pdl

pdl: pdl.cu
	$(CUDA_HOME)/bin/nvcc -ccbin $(CCBIN) -arch $(ARCH) -g -lineinfo pdl.cu -o $@

clean:
	rm -f pdl
