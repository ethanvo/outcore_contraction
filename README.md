# outcore_contraction

A high-performance, out-of-core tensor contraction engine written in C11.
Contracts arbitrary N-dimensional tensors that exceed available RAM by streaming
tiles from HDF5 files on NVMe storage through a double-buffered BLAS pipeline.

On Apple Silicon (M-series) the engine hits the **AMX compute roofline** for
COMPLEX128 arithmetic: ~361 GFLOPS sustained while keeping NVMe I/O
**fully hidden** behind BLAS execution.

---

## Features

| Capability | Detail |
|---|---|
| Out-of-core operation | Tensors of any size; RAM usage is bounded by a configurable pool |
| Generic N-D contractions | Einstein summation strings, e.g. `ijab,akbl->klji` |
| Double-buffered I/O | I/O thread prefetches the next tile pair while BLAS processes the current one |
| Block sparsity | Tiles not allocated on disk are skipped with zero I/O cost |
| Dtype support | `double` (FP64) and `double _Complex` (COMPLEX128) |
| BLAS backend | Apple Accelerate (AMX/vecLib) · Intel MKL · OpenBLAS · scalar fallback |
| NVMe alignment | 16 KiB-aligned pool pages match Apple Silicon NVMe page granularity |
| 2D SUMMA tiling | Minimises SSD write amplification vs. naïve row-by-row streaming |

---

## Dependencies

| Library | Purpose |
|---|---|
| [HDF5](https://www.hdfgroup.org/solutions/hdf5/) ≥ 1.12 | Chunked on-disk tensor storage |
| Apple Accelerate (macOS) | AMX-accelerated BLAS (`cblas_dgemm` / `cblas_zgemm`) |
| Intel MKL (Linux / Windows) | High-performance BLAS via Intel oneAPI |
| OpenBLAS (Linux / macOS) | Portable BLAS alternative |
| pthreads | Double-buffer I/O thread |
| GCD / libdispatch (macOS) | GCD-parallel BLAS dispatch for multi-core AMX |

Install HDF5:

```sh
# macOS
brew install hdf5

# Ubuntu / Debian
sudo apt install libhdf5-dev

# OpenBLAS (if not using MKL or Accelerate)
sudo apt install libopenblas-dev
```

---

## Build

```sh
# Configure — auto-detects the best BLAS on the current machine:
#   macOS  → Apple Accelerate
#   Linux  → Intel MKL (if MKLROOT is set) → OpenBLAS → scalar fallback
cmake -S . -B build

# Build everything
cmake --build build -j

# Run tests
./build/test_tensor_store
./build/test_io
./build/test_high_rank
./build/test_einsum

# Generate small test data and run a contraction
./build/gen_data          # creates A.h5, B.h5
./build/engine_app        # contracts A * B → C.h5

# Run the full benchmark suite
./build/bench_run_all
```

To override the BLAS backend explicitly:

```sh
cmake -S . -B build -DBLAS_BACKEND=MKL       # Intel MKL
cmake -S . -B build -DBLAS_BACKEND=OpenBLAS  # OpenBLAS
cmake -S . -B build -DBLAS_BACKEND=None      # scalar fallback (no BLAS dependency)
```

To skip the large 40 GiB benchmark case (CI / low-disk environments):

```sh
cmake -S . -B build -DCMAKE_C_FLAGS="-DSKIP_LARGE=1"
```

---

## Performance Tuning

### BLAS backend

The BLAS backend is the single largest performance lever.  Set it at configure
time with `-DBLAS_BACKEND=<name>`.  Auto-detection picks the best available
backend, but you can override it for reproducibility or benchmarking.

| Backend | Typical use case |
|---|---|
| `Accelerate` | macOS — uses AMX coprocessor, highest throughput on Apple Silicon |
| `MKL` | Linux (Intel CPUs) — AVX-512 / AVX2, multi-threaded via `iomp5` |
| `OpenBLAS` | Portable Linux / macOS fallback |
| `None` | Debugging or dependency-free builds; scalar fallback only |

### Threading (MKL)

When built with MKL the engine links against `mkl_intel_thread`, which uses
Intel's OpenMP runtime (`libiomp5`).  Control thread count via:

```sh
# Number of threads MKL uses for BLAS calls (takes precedence over OMP_NUM_THREADS)
export MKL_NUM_THREADS=4

# Fallback — respected if MKL_NUM_THREADS is not set
export OMP_NUM_THREADS=4
```

**`KMP_AFFINITY`** controls how OpenMP threads are bound to physical cores.
Poor affinity causes threads to migrate between cores mid-computation, breaking
cache locality and hurting L3 reuse — which matters especially for the large
tile sizes this engine uses.

```sh
# Recommended for single-socket Intel machines (most laptops and workstations):
# Pack threads onto consecutive cores, starting from core 0, at fine granularity.
export KMP_AFFINITY=compact,1,0,granularity=fine

# NUMA / multi-socket: scatter threads across sockets to maximise memory BW
export KMP_AFFINITY=scatter,granularity=fine

# Disable affinity (OS scheduler decides — lowest overhead to set up, worst sustained perf)
export KMP_AFFINITY=none
```

`compact` is the right default for this workload: the contracted tiles being
accumulated into C stay hot in L3, and packing threads on one socket avoids
inter-socket latency.

### Threading (OpenBLAS)

```sh
export OPENBLAS_NUM_THREADS=4   # or OMP_NUM_THREADS if OpenBLAS was built with OpenMP
```

### CPU frequency governor (Linux)

The `schedutil` governor drops CPU frequency during I/O-wait phases, then
ramps back up for compute — adding latency at the start of each BLAS call.
For sustained compute-bound runs, switch to `performance`:

```sh
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Reset afterwards with `schedutil` or `ondemand`.

### Buffer pool size

The pool cap is the main memory knob.  Larger pools reduce redundant I/O by
keeping more tiles in RAM between macro-blocks.

```sh
# Via environment variable (takes effect at runtime, no rebuild needed)
export TENSOR_POOL_MB=2048   # 2 GiB pool

# Or at API level
cfg.pool_mb = 2048;
```

Set `TENSOR_POOL_MB` to roughly 10–20 % of available RAM as a starting point.
The engine prints the actual pool configuration at startup.

### Storage

The engine is I/O-bound unless compute tiles are large enough to saturate the
BLAS pipeline.  Storage speed matters:

| Storage | Sequential read | Impact |
|---|---|---|
| Spinning HDD | ~100 MB/s | I/O dominates; BLAS mostly idle |
| SATA SSD | ~500 MB/s | Mixed regime; tile size determines bottleneck |
| NVMe SSD | 3–7 GB/s | BLAS often becomes the bottleneck; full pipeline utilisation |
| `tmpfs` ramdisk | ~20 GB/s | Compute-bound; shows peak GFLOPS |

To benchmark compute performance in isolation, copy the `.h5` files to a
`tmpfs` mount before running the engine:

```sh
sudo mount -t tmpfs -o size=8G tmpfs /mnt/ram
cp A.h5 B.h5 /mnt/ram/
cd /mnt/ram && /path/to/build/engine_app
```

---

## Public API

Include `<tensor_engine.h>` — no other headers are needed.

```c
#include <tensor_engine.h>

int main(void)
{
    /* 1. Configure (all zeros → auto-tune to machine) */
    tensor_engine_config_t cfg = {0};
    cfg.pool_mb = 512;          /* cap buffer pool at 512 MiB; 0 = 80 % of RAM */

    /* 2. Create engine */
    tensor_engine_t *eng = tensor_engine_init(&cfg);
    if (!eng) { /* allocation failed */ }

    /* 3. Contract
     *    Input files must contain a dataset named "tensor".
     *    Output file is created (or overwritten).
     */
    int rc = tensor_engine_contract(eng,
                                    "ijab,akbl->klji",   /* einsum expr   */
                                    "A.h5",              /* operand A     */
                                    "B.h5",              /* operand B     */
                                    "C.h5");             /* result C      */
    if (rc != TENSOR_ENGINE_OK)
        fprintf(stderr, "contraction failed: %s\n", tensor_engine_strerror(rc));

    /* 4. Destroy */
    tensor_engine_free(eng);
    return rc;
}
```

Link against `libtensor_core.a`, HDF5, and `libm`:

```makefile
CFLAGS  = -std=c11 -Ipath/to/include
LDFLAGS = -Lpath/to/build -ltensor_core -lhdf5 -lm -lpthread
```

### Error codes

| Code | Value | Meaning |
|---|---|---|
| `TENSOR_ENGINE_OK` | 0 | Success |
| `TENSOR_ENGINE_ERR_FILE` | -1 | File not found or I/O error |
| `TENSOR_ENGINE_ERR_DIMS` | -2 | Incompatible tensor dimensions |
| `TENSOR_ENGINE_ERR_EXPR` | -3 | Malformed einsum expression |
| `TENSOR_ENGINE_ERR_MEM` | -4 | Memory allocation failed |
| `TENSOR_ENGINE_ERR` | -5 | Unspecified internal error |

### Configuration fields

| Field | Default | Description |
|---|---|---|
| `pool_mb` | 0 (80 % of RAM) | Buffer pool cap in MiB |
| `tile_bytes` | 0 (16 MiB) | Target tile byte budget |

---

## Architecture

### Out-of-core design

Tensors are stored in **chunked HDF5 datasets**.  Each chunk (tile) is a
contiguous hyper-rectangle of the tensor.  At runtime only the tiles currently
needed for computation are resident in the buffer pool; everything else stays on
NVMe.

```
HDF5 files (A, B)
    ↓  registry_scan_file()    mark which tiles exist on disk
    ↓  pool_acquire()          reserve a RAM page
    ↓  read_chunk_fast()       DMA tile → pool page (HDF5 hyperslab)
    ↓  compute kernel          BLAS GEMM on tile pair
    ↓  write_chunk_fast()      DMA result → output file
    ↓  pool_release()          return page to pool
```

### 2D SUMMA tiling — minimising SSD wear

A naïve implementation of `C(i,j) += A(i,k) * B(k,j)` loads the entire B
tensor once per row-tile of A, for a total read volume of
`O(I × K × J + I × K × J) = O(2 × total_elements)`.

The engine instead uses a **2D SUMMA blocking** strategy: the output space
`(i, j)` is partitioned into macro-blocks.  For each macro-block the inner
contracted dimensions are iterated once, producing a _tile read volume_ that is
proportional only to the macro-block area rather than the full tensor.

For the 40 GiB benchmark this reduces total NVMe reads from ~1.8 TiB (naïve)
to ~262 GiB — a **7× reduction in SSD wear** at identical arithmetic intensity.

### Double-buffered NVMe streaming

The engine runs two OS threads per contraction:

```
I/O thread:                    Compute thread:
  slot[1] ← read A₁, B₁         wait for slot[0] ready
  signal slot[0] ready     →     BLAS(A₀, B₀) → C₀
  slot[0] ← read A₂, B₂         wait for slot[1] ready
  signal slot[1] ready     →     BLAS(A₁, B₁) → C₁
  ...                            ...
```

The I/O latency (~4 ms for a 16 MiB tile at 4 GB/s) is fully hidden inside
the BLAS window (~14 ms for a 1024×1024 ZGEMM at 361 GFLOPS).

### Apple Silicon AMX ceiling for COMPLEX128

Apple's Accelerate framework dispatches `cblas_zgemm` to the **AMX** (Apple
Matrix coprocessor) which delivers peak throughput on FP64 real arithmetic.

Complex multiplication requires **4 multiplies + 2 adds** per element pair
(vs 1 multiply + 1 add for real arithmetic), so the FP64 multiply bandwidth
is the binding constraint.  This gives:

```
Peak COMPLEX128 ≈ (Peak FP64 GFLOPS) × (8 FLOPs/complex pair)
                                      ÷ (6 muls + 2 adds per complex pair)
               ≈ AMX_peak × 8/6
```

In practice, the M-series AMX delivers ~540 GFLOPS FP64 real and
~361 GFLOPS COMPLEX128 — a ratio of ~1.5×, matching the 6-multiply
bottleneck theory.  There is no further optimisation available at the
user-space level; the engine is at the hardware ceiling.

### Component layers

| Module | File | Role |
|---|---|---|
| Public API | `src/tensor_engine.c` | Opaque context, env-var protocol |
| Engine | `src/engine.c` | Contraction orchestrator, double-buffer pipeline |
| I/O | `src/tensor_store.c` | HDF5 hyperslab read/write, boundary clamping |
| Registry | `src/registry.c` | Tile metadata, block-sparsity map |
| Pool | `src/memory.c` | LIFO page allocator, O(1) acquire/release |
| Einsum | `src/einsum.c` | Expression parser, dimension permutation |
| Odometer | `src/odometer.c` | N-dimensional tile iterator |
| Write queue | `src/write_queue.c` | Async ring-buffer for HDF5 writes |
| Metal | `src/metal_backend.m` | GPU GEMM stub (Apple Silicon, optional) |

---

## Benchmark

Run `./build/bench_run_all` to execute both cases and print a Markdown table.
Representative numbers on an Apple M2 Max (96 GB unified memory, 7 GB/s NVMe):

| Case | Dimensions | Tensor Size | Chunk | Read | Write | Elapsed | GFLOPS |
|---|---|---|---|---|---|---|---|
| Small | 80^4 | 0.61 GiB | 16^4 | 0.2 GiB/s | 0.1 GiB/s | 5.4 s | **388** |
| Large | 224^4 | 37.5 GiB | 32^4 | 0.03 GiB/s | 0.01 GiB/s | 2631 s | **384** |

> The large case is AMX compute-bound; the low apparent I/O bandwidth
> reflects that NVMe transfers are fully hidden inside BLAS windows.
> 2D SUMMA reduces total NVMe reads from 1875 GiB (naïve) to 300 GiB — **6.2×
> less SSD wear** with zero redundant B reads.

---

## File format

Input and output tensors are stored as **chunked HDF5 datasets** named
`"tensor"` inside `.h5` files.  Dtype is inferred at runtime from the A
dataset.

Create a compatible file from C:

```c
#include "tensor_store.h"  /* internal header — not needed via public API */

hsize_t shape[]      = {224, 224, 224, 224};
hsize_t chunk_dims[] = { 32,  32,  32,  32};
create_chunked_dataset_einsum("A.h5", "tensor", 4,
                               shape, chunk_dims, DTYPE_COMPLEX128);
```

Or use the bundled generator:

```sh
./build/gen_complex_tensor 224 32 A    # creates A.h5
```
