#!/usr/bin/env python3
"""
tests/verify_mixed_rank.py

Mixed-rank COMPLEX128 contraction verifier for the C11 out-of-core tensor engine.

Expression  : abcde,cfeg->abdfg   (rank-5 × rank-4 → rank-5)

  A  shape  : (DIM, DIM, DIM, DIM, DIM)   [indices: a b c d e]
  B  shape  : (DIM, DIM, DIM, DIM)         [indices: c f e g]
  C  shape  : (DIM, DIM, DIM, DIM, DIM)   [indices: a b d f g]

  Contracted        : c, e
  Free dims from A  : a b d
  Free dims from B  : f g

Chunk layout (NVMe-page aligned, matching C11 pool geometry):
  A chunks  : (CHUNK_A,)*5    8^5 × 16 B = 512 KiB per tile
  B chunks  : (CHUNK_B,)*4    8^4 × 16 B =  64 KiB per tile

Tolerance:
  rtol = atol = 1e-12  — strict IEEE-754 FP64 accuracy with AMX accumulation
  drift tolerance for DIM=32 and K=DIM^2=1024 contracted terms:
    expected rounding error ≈ K × ε ≈ 1024 × 2.2e-16 = 2.3e-13  <<  1e-12  ✓

Usage:
  python tests/verify_mixed_rank.py          # DIM=32 (~1.1 GiB for numpy)
  TEST_DIM=24 python tests/verify_mixed_rank.py

Workflow:
  1. Generates random COMPLEX128 A and B → build/A_mixed.h5, build/B_mixed.h5
  2. Pauses — user runs the C11 engine to produce build/C_mixed.h5
  3. Reads C11 output, computes numpy.einsum reference, verifies with rtol=atol=1e-12
"""

import sys
import os
import textwrap

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPR     = "abcde,cfeg->abdfg"
DIM      = int(os.environ.get("TEST_DIM", "32"))
CHUNK_A  = 8   # rank-5 chunk side:  8^5 × 16 B = 524 288 B = 512 KiB  (32× 16 KiB)
CHUNK_B  = 8   # rank-4 chunk side:  8^4 × 16 B =  65 536 B =  64 KiB   (4× 16 KiB)
SEED     = 42

_HERE     = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(_HERE, "..", "build")
FILE_A    = os.path.join(BUILD_DIR, "A_mixed.h5")
FILE_B    = os.path.join(BUILD_DIR, "B_mixed.h5")
FILE_C    = os.path.join(BUILD_DIR, "C_mixed.h5")
DSET      = "tensor"

RTOL = 1e-12
ATOL = 1e-12

# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _to_struct(arr: np.ndarray) -> np.ndarray:
    """
    Convert np.complex128 array to a structured array {'r': float64, 'i': float64}.

    This exactly matches the compound type created by the C11 engine:
        H5Tinsert(type, "r", 0,              H5T_NATIVE_DOUBLE)
        H5Tinsert(type, "i", sizeof(double), H5T_NATIVE_DOUBLE)
    so both the C reader and h5py speak the same on-disk layout.
    """
    dt  = np.dtype([("r", np.float64), ("i", np.float64)])
    out = np.empty(arr.shape, dtype=dt)
    out["r"] = arr.real
    out["i"] = arr.imag
    return out


def _from_struct(arr: np.ndarray) -> np.ndarray:
    """Inverse of _to_struct: structured {r,i} → complex128."""
    return arr["r"].astype(np.float64) + 1j * arr["i"].astype(np.float64)


def write_tensor(path: str, name: str, data: np.ndarray, chunk_side: int) -> None:
    """
    Write a complex128 tensor to HDF5 with an explicit compound type {r,i}
    and uniform chunking.

    Chunking is specified explicitly — no h5py auto-chunking heuristics,
    no compression, no shuffle filters.  Each chunk is a contiguous raw
    block of doubles, exactly as the C11 engine expects.
    """
    rank         = data.ndim
    chunk_tuple  = (chunk_side,) * rank
    struct_arr   = _to_struct(data)
    tile_counts  = tuple(
        int(np.ceil(s / chunk_side)) for s in data.shape
    )
    n_tiles      = 1
    for t in tile_counts:
        n_tiles *= t
    tile_bytes   = (chunk_side ** rank) * 16      # complex128 = 16 B
    size_mib     = data.nbytes / (1024 ** 2)

    with h5py.File(path, "w") as f:
        # create_dataset with an explicit structured dtype and chunk tuple.
        # The HDF5 file will contain a single chunked dataset with compound
        # type {r: H5T_IEEE_F64LE, i: H5T_IEEE_F64LE} — the canonical format
        # for the C11 engine's read_chunk_typed / write_chunk_typed paths.
        f.create_dataset(name, data=struct_arr, chunks=chunk_tuple)

    print(f"  {path}")
    print(f"    shape  : {data.shape}   dtype: complex128")
    print(f"    chunks : {chunk_tuple}   {tile_bytes // 1024} KiB/tile")
    print(f"    tiles  : {tile_counts} = {n_tiles:,} total")
    print(f"    size   : {size_mib:.1f} MiB")


def read_tensor(path: str, name: str) -> np.ndarray:
    """
    Read a COMPLEX128 tensor written by the C11 engine.
    The on-disk compound type {r, i} is reconstructed into np.complex128.
    """
    with h5py.File(path, "r") as f:
        raw = f[name][()]
    if raw.dtype.names and "r" in raw.dtype.names and "i" in raw.dtype.names:
        return _from_struct(raw)
    # Fall back: dataset was written as native complex128 by some other tool.
    return np.asarray(raw, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(BUILD_DIR, exist_ok=True)

    # ==================================================================
    # Phase 1 — generate random COMPLEX128 inputs
    # ==================================================================

    shape_A = (DIM,) * 5
    shape_B = (DIM,) * 4

    a_mib = np.prod(shape_A) * 16 / (1024 ** 2)
    b_mib = np.prod(shape_B) * 16 / (1024 ** 2)
    c_mib = a_mib  # C has same rank/dims as A for this expression

    print("=" * 66)
    print("  Mixed-Rank COMPLEX128 Contraction Verifier")
    print(f"  Expression : {EXPR}")
    print(f"  DIM        : {DIM}")
    print(f"  A size     : {a_mib:.1f} MiB   B size : {b_mib:.1f} MiB")
    print(f"  C size     : {c_mib:.1f} MiB   (expected output)")
    print(f"  numpy RAM  : ~{(a_mib + b_mib + c_mib) / 1024:.2f} GiB  (A + B + C)")
    print("=" * 66)
    print()

    print("Generating random COMPLEX128 data  (seed=42, reproducible)...")
    rng = np.random.default_rng(SEED)

    A = (
        rng.standard_normal(shape_A) + 1j * rng.standard_normal(shape_A)
    ).astype(np.complex128)

    B = (
        rng.standard_normal(shape_B) + 1j * rng.standard_normal(shape_B)
    ).astype(np.complex128)

    print(f"  A: mean magnitude = {np.abs(A).mean():.4f}")
    print(f"  B: mean magnitude = {np.abs(B).mean():.4f}")
    print()

    print("Writing input tensors to HDF5...")
    write_tensor(FILE_A, DSET, A, CHUNK_A)
    print()
    write_tensor(FILE_B, DSET, B, CHUNK_B)
    print()

    # ==================================================================
    # Phase 2 — hand-off to C11 engine
    # ==================================================================

    print("-" * 66)
    print(
        "Inputs generated.  Run the C11 engine to produce "
        f"{os.path.relpath(FILE_C)}, then press Enter to continue "
        "verification..."
    )
    print()
    print(textwrap.dedent(f"""\
        From the repository root:

          ./build/test_mixed_rank

        Or call run_contraction_einsum directly in a C harness:

          run_contraction_einsum(
              "{EXPR}",
              "{os.path.relpath(FILE_A)}", "{DSET}",
              "{os.path.relpath(FILE_B)}", "{DSET}",
              "{os.path.relpath(FILE_C)}", "{DSET}"
          );
    """))
    print("-" * 66)
    input()

    # ==================================================================
    # Phase 3 — ultra-strict verification
    # ==================================================================

    print("Loading C11 engine output...")
    if not os.path.exists(FILE_C):
        print(
            f"ERROR: {FILE_C} does not exist.  "
            "Did the engine run successfully?",
            file=sys.stderr,
        )
        sys.exit(1)

    C_engine = read_tensor(FILE_C, DSET)
    print(f"  shape  : {C_engine.shape}   dtype: {C_engine.dtype}")

    expected_shape = tuple(
        DIM if idx in "abdg" else DIM  # all indices happen to be DIM
        for idx in "abdfg"
    )
    # The C11 engine produces C with shape derived from the expression:
    # abdfg — all DIM, so expected shape = (DIM,)*5
    if C_engine.shape != (DIM,) * 5:
        print(
            f"ERROR: unexpected C shape {C_engine.shape}, "
            f"expected {(DIM,)*5}",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print(f"Computing reference  np.einsum('{EXPR}', A, B)  ...")
    C_ref = np.einsum(EXPR, A, B)
    print(f"  shape  : {C_ref.shape}   dtype: {C_ref.dtype}")

    # ---- per-element error diagnostics -----------------------------------

    print()
    print("Comparing engine output to numpy reference...")

    abs_diff  = np.abs(C_engine - C_ref)
    max_err   = float(abs_diff.max())
    max_idx   = np.unravel_index(int(abs_diff.argmax()), abs_diff.shape)
    ref_val   = complex(C_ref[max_idx])
    act_val   = complex(C_engine[max_idx])
    ref_mag   = abs(ref_val)
    rel_err   = max_err / ref_mag if ref_mag > 0 else float("inf")

    print(f"  max |error|        : {max_err:.6e}")
    print(f"  relative error     : {rel_err:.6e}")
    print(f"  at index           : {max_idx}")
    print(f"  reference value    : {ref_val.real:.15g} + {ref_val.imag:.15g}j")
    print(f"  engine value       : {act_val.real:.15g} + {act_val.imag:.15g}j")
    print(f"  tolerance (rtol)   : {RTOL}")
    print(f"  tolerance (atol)   : {ATOL}")

    # ---- strict assert ---------------------------------------------------

    try:
        np.testing.assert_allclose(C_engine, C_ref, rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        print()
        print("=" * 66)
        print("  *** VERIFICATION FAILED ***")
        print("=" * 66)
        # Print the first line of numpy's summary (contains mismatch count).
        summary = str(exc).splitlines()
        for line in summary[:8]:
            print(f"  {line}")
        print()
        print(f"  Worst error coordinate : {max_idx}")
        print(f"  Reference              : {ref_val!r}")
        print(f"  Engine output          : {act_val!r}")
        print(f"  |error|                : {max_err:.6e}")
        print("=" * 66)
        sys.exit(1)

    print()
    print("=" * 66)
    print("  *** ALL PASSED ***")
    print()
    print(f"  Expression : {EXPR}")
    print(f"  DIM        : {DIM}   (all indices)")
    print(f"  max |error|: {max_err:.3e}   (tolerance: {RTOL})")
    print()
    print("  C11 engine output matches numpy.einsum reference within")
    print(f"  rtol={RTOL}  atol={ATOL}  — strict IEEE-754 FP64 accuracy.")
    print("=" * 66)


if __name__ == "__main__":
    main()
