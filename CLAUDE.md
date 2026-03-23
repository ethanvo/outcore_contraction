# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Configure (first time or after CMakeLists.txt changes)
cmake -S . -B build

# Build all targets
cmake --build build

# Run tests
./build/test_tensor_store   # HDF5 I/O unit tests incl. boundary clamping
./build/test_io             # Registry + I/O integration test (rank-3)
./build/test_high_rank      # Rank-4 and rank-5 mathematical verification

# Generate test data then run the engine
./build/gen_data            # Creates A.h5 and B.h5 in the working directory
./build/engine_app          # Contracts A * B → C.h5
```

The project uses C11 with `-Wall -Wextra`. There is no separate lint step.

## Architecture

This is an **out-of-core tensor contraction engine** — it contracts tensors that exceed available RAM by streaming chunks from HDF5 files.

**Core operation:** `A(i,k) * B(k,j) → C(i,j)` using the SUMMA algorithm, with block-sparsity: tile pairs where either operand doesn't exist on disk are skipped entirely.

### Component Layers (bottom to top)

**`src/tensor_store.c`** — HDF5 I/O layer
`calculate_chunk_dims()` targets a byte budget using the nth-root approach. `read_chunk_fast()` / `write_chunk_fast()` take an already-open `hid_t` to avoid per-call open/close overhead. Both use a **full nominal memspace** with a subregion selection at the origin for boundary tiles — data always lands at the correct row-major position for `chunk_dims`-strided access. `dset_open_no_cache()` opens with the HDF5 internal chunk cache disabled (`H5Pset_chunk_cache(..., 0, 0, 0.0)`) since `BufferPool` is the sole cache. `create_chunked_dataset()` sets `H5D_ALLOC_TIME_INCR` for block-sparse on-disk layout.

**`src/memory.c`** — Buffer pool
Single contiguous allocation split into fixed pages. LIFO free-stack gives O(1) `pool_acquire` / `pool_release`. Page IDs are `size_t`; `SIZE_MAX` is the "not yet acquired" sentinel.

**`src/registry.c`** — Tile metadata registry
`TileMetadata` and `TensorRegistry` use `[MAX_RANK]` arrays (MAX_RANK=8). `registry_create_from_dset()` reads rank, global_dims, and chunk_dims directly from the HDF5 dataset creation property list — this is the **authoritative** factory for existing files and eliminates any risk of chunk-boundary divergence. `registry_get_tile()` accepts a `const hsize_t *tile_coords` array (n-dimensional). `registry_scan_file()` iterates `H5Dget_chunk_info` over allocated chunks and marks tiles `TILE_STATUS_ON_DISK`.

**`src/engine.c`** — Contraction orchestrator
Reads rank and global dims from the file dataspaces; validates K-dimension compatibility; uses `registry_create_from_dset` for all three tensors. The compute kernel uses `m → k → n` loop order with `restrict`-qualified pointers: the inner `n`-loop streams `B[k*N+n]` and `C[m*N+n]` sequentially. Boundary tile padding is benign (zeros from `read_chunk_fast` pre-zero).

**`src/gen_data.c`** — Test data generator
Generates rank-2 HDF5 files. Passes nominal `chunk_dims` to `write_chunk_fast` and lets it handle boundary clamping internally.

**`src/main.c`** — Entry point
Thin wrapper calling `run_contraction`.

### Data Flow

```
HDF5 files (A, B)
    → registry_scan_file()   [mark which tiles exist]
    → pool_acquire()         [get a RAM page]
    → read_chunk_fast()      [load tile into page]
    → compute kernel         [multiply tile pair]
    → write_chunk_fast()     [flush result to C]
    → pool_release()         [return page]
```

### Development Phases (per plan.txt)

- **Phase 1–3:** Complete — storage layer, registry, memory pool
- **Phase 4:** In progress (`contraction-scheduler` branch) — single-threaded compute core (has known segfault)
- **Phase 5:** Planned — threading, async prefetch, double-buffering, LRU cache
