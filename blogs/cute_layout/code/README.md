# CuTe Layout and Tensor

## Requirements

CUDA and Cutlass are required to build the c++ code.

Works on Hopper (sm_90a) and Blackwell (sm_100a).

## Usage

Create your own `Makefile.paths` (by copying [Makefile.paths.example](./Makefile.paths.example)) and set `CUDA_HOME` and `CUTLASS_HOME`.

```bash
# build
make
# run
./cute_layout
```