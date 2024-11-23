# Using TMA Load and Prefetch in Cute

## Requirements

CUDA and Cutlass are required to build the code.

Only works on Hopper.

## Usage

Create your own `Makefile.paths` (by copying [Makefile.paths.example](./Makefile.paths.example)) and set `CUDA_HOME` and `CUTLASS_HOME`.

```bash
# build
make
# run
./cute_tma
```

You should see result like:
```bash
block: (0, 0), value: 0.000000, 0.000000
block: (1, 0), value: 1.000000, 1.000000
block: (0, 1), value: 1.000000, -1.000000
block: (1, 1), value: 2.000000, 0.000000
block: (2, 0), value: 2.000000, 2.000000
block: (3, 0), value: 3.000000, 3.000000
block: (2, 1), value: 3.000000, 1.000000
block: (3, 1), value: 4.000000, 2.000000
execution time 5.105248 ms
```
