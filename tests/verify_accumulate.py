#!/usr/bin/env python3
"""
tests/verify_accumulate.py

Independent numpy oracle for test_accumulate.

Reads the HDF5 input and output files written by the C test driver
(./build/test_accumulate) and verifies every output element against
numpy.einsum.  No knowledge of constant fill values is required — the
actual on-disk data is used directly as the reference.

Usage (run from the repo root after ./build/test_accumulate):
    python3 tests/verify_accumulate.py

Requirements:
    pip install h5py numpy
"""

import sys
import os
import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required — install with:  pip install h5py")
    sys.exit(2)

TOL = 1e-10


def read_tensor(fname):
    """Read the 'tensor' dataset, converting HDF5 compound complex types."""
    with h5py.File(fname, "r") as f:
        data = f["tensor"][:]
    # COMPLEX128 is stored as compound {r: float64, i: float64}.
    # Older h5py returns a structured array; newer h5py may auto-convert to
    # native complex128.  Handle both.
    if data.dtype.names and "r" in data.dtype.names and "i" in data.dtype.names:
        return (data["r"] + 1j * data["i"]).astype(np.complex128)
    if np.issubdtype(data.dtype, np.complexfloating):
        return data
    return data.astype(np.float64)


def check(label, actual, expected):
    """Compare actual vs expected arrays; print PASS/FAIL and return bool."""
    if actual.shape != expected.shape:
        print(f"  FAIL  {label}: shape mismatch — "
              f"got {actual.shape}, expected {expected.shape}")
        return False
    max_err = float(np.max(np.abs(actual - expected)))
    ok = max_err <= TOL
    print(f"  {'PASS' if ok else 'FAIL'}  {label}  "
          f"(shape={actual.shape}, max_err={max_err:.2e})")
    return ok


def files_exist(*fnames):
    """Return True if all files exist; print SKIP and return False otherwise."""
    missing = [f for f in fnames if not os.path.exists(f)]
    if missing:
        print(f"  SKIP (missing files: {', '.join(missing)})")
        return False
    return True


# -----------------------------------------------------------------------
# T1: rank-2 FP64  "ij,jk->ik"  two terms
# -----------------------------------------------------------------------
def verify_t1():
    print("\n=== T1: ij,jk->ik  FP64  two terms ===")
    fs = ["acc_t1_A1.h5", "acc_t1_B1.h5",
          "acc_t1_A2.h5", "acc_t1_B2.h5", "acc_t1_C.h5"]
    if not files_exist(*fs):
        return None

    A1 = read_tensor("acc_t1_A1.h5")
    B1 = read_tensor("acc_t1_B1.h5")
    A2 = read_tensor("acc_t1_A2.h5")
    B2 = read_tensor("acc_t1_B2.h5")
    C  = read_tensor("acc_t1_C.h5")

    C_exp = np.einsum("ij,jk->ik", A1, B1) + np.einsum("ij,jk->ik", A2, B2)
    return check("C", C, C_exp)


# -----------------------------------------------------------------------
# T2: rank-2 FP64  "ij,jk->ki"  two terms (transposed output)
# -----------------------------------------------------------------------
def verify_t2():
    print("\n=== T2: ij,jk->ki  FP64  two terms (transposed output) ===")
    fs = ["acc_t2_A1.h5", "acc_t2_B1.h5",
          "acc_t2_A2.h5", "acc_t2_B2.h5", "acc_t2_C.h5"]
    if not files_exist(*fs):
        return None

    A1 = read_tensor("acc_t2_A1.h5")
    B1 = read_tensor("acc_t2_B1.h5")
    A2 = read_tensor("acc_t2_A2.h5")
    B2 = read_tensor("acc_t2_B2.h5")
    C  = read_tensor("acc_t2_C.h5")

    C_exp = np.einsum("ij,jk->ki", A1, B1) + np.einsum("ij,jk->ki", A2, B2)
    return check("C", C, C_exp)


# -----------------------------------------------------------------------
# T3: rank-2 FP64  "ij,jk->ik"  three terms
# -----------------------------------------------------------------------
def verify_t3():
    print("\n=== T3: ij,jk->ik  FP64  three terms ===")
    fs = ["acc_t3_A1.h5", "acc_t3_B1.h5",
          "acc_t3_A2.h5", "acc_t3_B2.h5",
          "acc_t3_A3.h5", "acc_t3_B3.h5", "acc_t3_C.h5"]
    if not files_exist(*fs):
        return None

    A1 = read_tensor("acc_t3_A1.h5")
    B1 = read_tensor("acc_t3_B1.h5")
    A2 = read_tensor("acc_t3_A2.h5")
    B2 = read_tensor("acc_t3_B2.h5")
    A3 = read_tensor("acc_t3_A3.h5")
    B3 = read_tensor("acc_t3_B3.h5")
    C  = read_tensor("acc_t3_C.h5")

    C_exp = (np.einsum("ij,jk->ik", A1, B1) +
             np.einsum("ij,jk->ik", A2, B2) +
             np.einsum("ij,jk->ik", A3, B3))
    return check("C", C, C_exp)


# -----------------------------------------------------------------------
# T4: rank-3 FP64  "abc,cbd->ad"  two terms
# -----------------------------------------------------------------------
def verify_t4():
    print("\n=== T4: abc,cbd->ad  FP64  two terms ===")
    fs = ["acc_t4_A1.h5", "acc_t4_B1.h5",
          "acc_t4_A2.h5", "acc_t4_B2.h5", "acc_t4_C.h5"]
    if not files_exist(*fs):
        return None

    A1 = read_tensor("acc_t4_A1.h5")
    B1 = read_tensor("acc_t4_B1.h5")
    A2 = read_tensor("acc_t4_A2.h5")
    B2 = read_tensor("acc_t4_B2.h5")
    C  = read_tensor("acc_t4_C.h5")

    C_exp = (np.einsum("abc,cbd->ad", A1, B1) +
             np.einsum("abc,cbd->ad", A2, B2))
    return check("C", C, C_exp)


# -----------------------------------------------------------------------
# T5: rank-2 COMPLEX128  "ij,jk->ik"  two terms
# -----------------------------------------------------------------------
def verify_t5():
    print("\n=== T5: ij,jk->ik  COMPLEX128  two terms ===")
    fs = ["acc_t5_A1.h5", "acc_t5_B1.h5",
          "acc_t5_A2.h5", "acc_t5_B2.h5", "acc_t5_C.h5"]
    if not files_exist(*fs):
        return None

    A1 = read_tensor("acc_t5_A1.h5")
    B1 = read_tensor("acc_t5_B1.h5")
    A2 = read_tensor("acc_t5_A2.h5")
    B2 = read_tensor("acc_t5_B2.h5")
    C  = read_tensor("acc_t5_C.h5")

    C_exp = (np.einsum("ij,jk->ik", A1, B1) +
             np.einsum("ij,jk->ik", A2, B2))
    return check("C", C, C_exp)


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------
def main():
    print("=== verify_accumulate.py: numpy oracle for test_accumulate ===")
    print(f"Working directory: {os.getcwd()}")

    results = [
        verify_t1(),
        verify_t2(),
        verify_t3(),
        verify_t4(),
        verify_t5(),
    ]

    passed  = sum(1 for r in results if r is True)
    failed  = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)

    print(f"\n--- Results: {passed} passed, {failed} failed, {skipped} skipped ---")
    if failed:
        print("FAILED")
        sys.exit(1)
    else:
        print("ALL PASSED")


if __name__ == "__main__":
    main()
