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

You will see speedup by kernel overlap with PDL enabled.
```bash
Testing PDL (Programmatic Dependent Launch) performance...

Launching two kernels with PDL enabled...
Time with PDL: 2733.72 ms

Launching two kernels with PDL disabled...
Time without PDL: 3280.45 ms

Performance comparison:
PDL enabled:  2733.72 ms
PDL disabled: 3280.45 ms
Difference:   546.725 ms
Speedup:      1.19999x

```