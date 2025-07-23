# Using TMA Load, Prefetch, and Multicast Load in CuTe

## Requirements

CUDA and Cutlass are required to build the c++ code.

CuTe DSL >= 4.1.0 is required to run the python code.

Works on Hopper (sm_90a) and Blackwell (sm_100a).

## Usage

### C++

Create your own `Makefile.paths` (by copying [Makefile.paths.example](./Makefile.paths.example)) and set `CUDA_HOME` and `CUTLASS_HOME`.

```bash
# build
make
# run
./cute_tma
```

For TMA load, you should see result like:
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

Uncomment `cute_host_multicast<c...>(...);` to use TMA multicast, you should see result like:
```bash
block: (4, 3, 0), cluster: (2, 1, 0), cluster rank: 2, value: 3.000000, 1.000000
block: (5, 3, 0), cluster: (2, 1, 0), cluster rank: 3, value: 3.000000, 1.000000
block: (4, 2, 0), cluster: (2, 1, 0), cluster rank: 0, value: 3.000000, 1.000000
block: (5, 2, 0), cluster: (2, 1, 0), cluster rank: 1, value: 3.000000, 1.000000
block: (6, 3, 0), cluster: (3, 1, 0), cluster rank: 2, value: 4.000000, 2.000000
block: (6, 2, 0), cluster: (3, 1, 0), cluster rank: 0, value: 4.000000, 2.000000
block: (3, 3, 0), cluster: (1, 1, 0), cluster rank: 3, value: 2.000000, 0.000000
block: (2, 3, 0), cluster: (1, 1, 0), cluster rank: 2, value: 2.000000, 0.000000
block: (3, 2, 0), cluster: (1, 1, 0), cluster rank: 1, value: 2.000000, 0.000000
block: (7, 3, 0), cluster: (3, 1, 0), cluster rank: 3, value: 4.000000, 2.000000
block: (7, 2, 0), cluster: (3, 1, 0), cluster rank: 1, value: 4.000000, 2.000000
block: (2, 2, 0), cluster: (1, 1, 0), cluster rank: 0, value: 2.000000, 0.000000
block: (1, 3, 0), cluster: (0, 1, 0), cluster rank: 3, value: 1.000000, -1.000000
block: (0, 3, 0), cluster: (0, 1, 0), cluster rank: 2, value: 1.000000, -1.000000
block: (1, 0, 0), cluster: (0, 0, 0), cluster rank: 1, value: 0.000000, 0.000000
block: (0, 2, 0), cluster: (0, 1, 0), cluster rank: 0, value: 1.000000, -1.000000
block: (1, 2, 0), cluster: (0, 1, 0), cluster rank: 1, value: 1.000000, -1.000000
block: (0, 0, 0), cluster: (0, 0, 0), cluster rank: 0, value: 0.000000, 0.000000
block: (3, 0, 0), cluster: (1, 0, 0), cluster rank: 1, value: 1.000000, 1.000000
block: (2, 0, 0), cluster: (1, 0, 0), cluster rank: 0, value: 1.000000, 1.000000
block: (3, 1, 0), cluster: (1, 0, 0), cluster rank: 3, value: 1.000000, 1.000000
block: (1, 1, 0), cluster: (0, 0, 0), cluster rank: 3, value: 0.000000, 0.000000
block: (2, 1, 0), cluster: (1, 0, 0), cluster rank: 2, value: 1.000000, 1.000000
block: (0, 1, 0), cluster: (0, 0, 0), cluster rank: 2, value: 0.000000, 0.000000
block: (4, 0, 0), cluster: (2, 0, 0), cluster rank: 0, value: 2.000000, 2.000000
block: (5, 0, 0), cluster: (2, 0, 0), cluster rank: 1, value: 2.000000, 2.000000
block: (4, 1, 0), cluster: (2, 0, 0), cluster rank: 2, value: 2.000000, 2.000000
block: (5, 1, 0), cluster: (2, 0, 0), cluster rank: 3, value: 2.000000, 2.000000
block: (7, 0, 0), cluster: (3, 0, 0), cluster rank: 1, value: 3.000000, 3.000000
block: (6, 0, 0), cluster: (3, 0, 0), cluster rank: 0, value: 3.000000, 3.000000
block: (7, 1, 0), cluster: (3, 0, 0), cluster rank: 3, value: 3.000000, 3.000000
block: (6, 1, 0), cluster: (3, 0, 0), cluster rank: 2, value: 3.000000, 3.000000
execution time 6.843968 ms
```

### Python

```bash
python cute_tma.py
```
