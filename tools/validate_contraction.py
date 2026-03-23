#!/usr/bin/env python3
"""
validate_contraction.py

Ground-truth validator for run_contraction_einsum output.

Usage
-----
  python tools/validate_contraction.py A.h5 B.h5 C.h5 [--expr EINSUM] [--tiles N]

For the default expression ijab,akbl->klji:
  C(k,l,j,i) = sum_{a,b} A(i,j,a,b) * B(a,k,b,l)

The script reads a sample of C tiles from the engine output, recomputes
each one with NumPy, and checks that the results agree to within a
relative tolerance (default 1e-9).

HDF5 complex type
-----------------
Datasets are stored as HDF5 compound type {r: float64, i: float64}.
h5py returns a structured ndarray; we reconstruct complex128 as
  data['r'] + 1j * data['i']
"""

import argparse
import sys

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_complex(ds, sel=None):
    """
    Read a dataset that may be:
      - compound {r, i} → reconstruct as complex128
      - native complex128 → return as-is
      - float64           → return as float64
    sel: a numpy index-expression for slicing (applied before reconstruction).
    """
    raw = ds[sel] if sel is not None else ds[()]
    if raw.dtype.names and 'r' in raw.dtype.names and 'i' in raw.dtype.names:
        return raw['r'] + 1j * raw['i']
    return raw


def tile_slice(global_dims, chunk_dims, tile_idx):
    """
    Return a tuple of slice objects for the tile at tile_idx (int array)
    given global_dims and chunk_dims.  Clamps at dataset boundaries.
    """
    slices = []
    for d, (ti, cd, gd) in enumerate(zip(tile_idx, chunk_dims, global_dims)):
        start = ti * cd
        stop  = min(start + cd, gd)
        slices.append(slice(start, stop))
    return tuple(slices)


def sample_tile_indices(grid_dims, n_samples=5):
    """
    Return a list of sample tile coordinates:
      • corner (all zeros)
      • all-boundary (last tile in each dim)
      • centre
      • (optionally) additional random tiles
    """
    rank = len(grid_dims)
    samples = []

    corner = [0] * rank
    samples.append(corner)

    last = [g - 1 for g in grid_dims]
    if last != corner:
        samples.append(last)

    centre = [g // 2 for g in grid_dims]
    if centre not in samples:
        samples.append(centre)

    rng = np.random.default_rng(42)
    while len(samples) < n_samples:
        idx = [int(rng.integers(0, g)) for g in grid_dims]
        if idx not in samples:
            samples.append(idx)

    return samples


# ---------------------------------------------------------------------------
# Expression parser (subset: two-operand einsum only)
# ---------------------------------------------------------------------------

def parse_einsum(expr):
    """
    Parse 'ij...,kl...->mn...' → (lhs_A, lhs_B, rhs)
    Returns (str, str, str).
    """
    lhs, rhs = expr.split('->')
    parts = lhs.split(',')
    if len(parts) != 2:
        raise ValueError(f"Expected two operands, got: {expr}")
    return parts[0].strip(), parts[1].strip(), rhs.strip()


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(file_A, dset_A_name,
             file_B, dset_B_name,
             file_C, dset_C_name,
             expr, n_tiles, rtol, verbose):

    lhs_A, lhs_B, rhs_C = parse_einsum(expr)
    rank_A = len(lhs_A)
    rank_B = len(lhs_B)
    rank_C = len(rhs_C)

    print(f"Expression : {expr}")
    print(f"  A labels : {lhs_A}  (rank {rank_A})")
    print(f"  B labels : {lhs_B}  (rank {rank_B})")
    print(f"  C labels : {rhs_C}  (rank {rank_C})")

    with h5py.File(file_A, 'r') as fa, \
         h5py.File(file_B, 'r') as fb, \
         h5py.File(file_C, 'r') as fc:

        ds_A = fa[dset_A_name]
        ds_B = fb[dset_B_name]
        ds_C = fc[dset_C_name]

        global_A = ds_A.shape          # tuple of ints
        global_B = ds_B.shape
        global_C = ds_C.shape
        chunk_C  = ds_C.chunks         # may be None for contiguous datasets

        print(f"\nShapes:")
        print(f"  A : {global_A}")
        print(f"  B : {global_B}")
        print(f"  C : {global_C}")
        if chunk_C:
            print(f"  C chunks: {chunk_C}")

        if chunk_C is None:
            # Treat the whole tensor as one tile.
            chunk_C = global_C

        grid_dims = [int(np.ceil(g / c))
                     for g, c in zip(global_C, chunk_C)]
        print(f"  C grid  : {grid_dims}  ({np.prod(grid_dims)} tiles total)")

        # Build index maps: which C dim ↔ which A/B dim.
        # c_to_A[d] = index in A if rhs_C[d] is in lhs_A, else None.
        # c_to_B[d] = index in B if rhs_C[d] is in lhs_B, else None.
        c_to_A = {}
        c_to_B = {}
        for d, ch in enumerate(rhs_C):
            if ch in lhs_A:
                c_to_A[d] = lhs_A.index(ch)
            if ch in lhs_B:
                c_to_B[d] = lhs_B.index(ch)

        # Contracted indices (appear in both A and B but not in C).
        contracted = [ch for ch in lhs_A
                      if ch in lhs_B and ch not in rhs_C]

        print(f"\nContracted indices : {''.join(contracted)}")

        # Sample tiles.
        tile_samples = sample_tile_indices(grid_dims, n_tiles)
        print(f"\nChecking {len(tile_samples)} sample tiles ...\n")

        n_fail = 0
        n_pass = 0

        for tile_idx in tile_samples:
            c_sel = tile_slice(global_C, chunk_C, tile_idx)

            # Load actual C tile.
            C_actual = load_complex(ds_C, c_sel)
            if np.iscomplexobj(C_actual):
                C_actual = C_actual.astype(np.complex128)
            else:
                C_actual = C_actual.astype(np.float64)

            # Determine A and B slices:
            # For free dims of A: use the C-tile slice.
            # For contracted dims of A/B: use slice(None) (load all).
            a_slices = [slice(None)] * rank_A
            b_slices = [slice(None)] * rank_B

            for d, s in enumerate(c_sel):
                if d in c_to_A:
                    a_slices[c_to_A[d]] = s
                if d in c_to_B:
                    b_slices[c_to_B[d]] = s

            A_slab = load_complex(ds_A, tuple(a_slices))
            B_slab = load_complex(ds_B, tuple(b_slices))

            if np.iscomplexobj(A_slab) or np.iscomplexobj(B_slab):
                A_slab = A_slab.astype(np.complex128)
                B_slab = B_slab.astype(np.complex128)
            else:
                A_slab = A_slab.astype(np.float64)
                B_slab = B_slab.astype(np.float64)

            # Run NumPy reference.
            C_ref = np.einsum(expr, A_slab, B_slab, optimize=True)

            # Compare.
            abs_diff = np.abs(C_actual - C_ref)
            ref_norm = np.abs(C_ref)
            ref_norm = np.where(ref_norm == 0, 1.0, ref_norm)
            rel_err  = abs_diff / ref_norm
            max_rel  = float(np.max(rel_err))
            max_abs  = float(np.max(abs_diff))

            status = "PASS" if max_rel <= rtol else "FAIL"
            if status == "FAIL":
                n_fail += 1
            else:
                n_pass += 1

            idx_str = ",".join(str(i) for i in tile_idx)
            shp_str = "×".join(str(s) for s in C_actual.shape)
            print(f"  tile ({idx_str})  shape {shp_str}"
                  f"  max_rel={max_rel:.2e}  max_abs={max_abs:.2e}"
                  f"  [{status}]")

            if verbose and status == "FAIL":
                bad = np.unravel_index(np.argmax(rel_err), rel_err.shape)
                print(f"    worst element {bad}:"
                      f"  actual={C_actual[bad]:.6g}"
                      f"  ref={C_ref[bad]:.6g}")

    print()
    if n_fail == 0:
        print(f"=== ALL {n_pass} tiles PASSED (rtol={rtol:.0e}) ===")
        return 0
    else:
        print(f"=== {n_fail}/{n_pass+n_fail} tiles FAILED ===")
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Ground-truth validator for run_contraction_einsum output.")
    p.add_argument("file_A", help="HDF5 input file A")
    p.add_argument("file_B", help="HDF5 input file B")
    p.add_argument("file_C", help="HDF5 output file C (engine result)")
    p.add_argument("--dset",   default="tensor",
                   help="Dataset name in all three files (default: tensor)")
    p.add_argument("--dset-A", default=None, help="Dataset name in A (overrides --dset)")
    p.add_argument("--dset-B", default=None, help="Dataset name in B (overrides --dset)")
    p.add_argument("--dset-C", default=None, help="Dataset name in C (overrides --dset)")
    p.add_argument("--expr",   default="ijab,akbl->klji",
                   help="Einsum expression (default: ijab,akbl->klji)")
    p.add_argument("--tiles",  type=int, default=5,
                   help="Number of tiles to sample (default: 5)")
    p.add_argument("--rtol",   type=float, default=1e-9,
                   help="Relative tolerance for pass/fail (default: 1e-9)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print worst-element details on failure")
    args = p.parse_args()

    dA = args.dset_A or args.dset
    dB = args.dset_B or args.dset
    dC = args.dset_C or args.dset

    sys.exit(validate(
        args.file_A, dA,
        args.file_B, dB,
        args.file_C, dC,
        args.expr, args.tiles, args.rtol, args.verbose))


if __name__ == "__main__":
    main()
