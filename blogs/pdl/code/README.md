# Using Programmatic Dependent Launch (PDL) to Reduce End-to-End Latency

## Requirements

CUDA are required to build the c++ code.

Works on Hopper (sm_90a) and Blackwell (sm_100a).

## Usage

Create your own `Makefile.paths` (by copying [Makefile.paths.example](./Makefile.paths.example)) and set `CUDA_HOME`.

```bash
# build
make
# run
./pdl
```