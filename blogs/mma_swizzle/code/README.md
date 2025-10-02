# Tensor Core MMA Swizzle Layout

## Requirements

CUDA and Cutlass are required to build the c++ code.

## Usage

Create your own `Makefile.paths` (by copying [Makefile.paths.example](./Makefile.paths.example)) and set `CUDA_HOME` and `CUTLASS_HOME`.

```bash
# build
make
# run
./mma_swizzle
```