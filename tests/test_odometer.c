#include "odometer.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ----------------------------------------------------------------------- */
/* Test harness                                                              */
/* ----------------------------------------------------------------------- */

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        printf("  PASS  %s\n", msg); \
        g_pass++; \
    } else { \
        printf("  FAIL  %s\n", msg); \
        g_fail++; \
    } \
} while (0)

#define CHECK_SZ(a, b, msg) do { \
    if ((a) == (b)) { \
        printf("  PASS  %s  (%zu)\n", msg, (size_t)(a)); \
        g_pass++; \
    } else { \
        printf("  FAIL  %s  expected=%zu got=%zu\n", msg, (size_t)(b), (size_t)(a)); \
        g_fail++; \
    } \
} while (0)

#define CHECK_DBL(a, b, tol, msg) do { \
    double _a = (a), _b = (b); \
    if (fabs(_a - _b) <= (tol)) { \
        printf("  PASS  %s  (%.1f)\n", msg, _a); \
        g_pass++; \
    } else { \
        printf("  FAIL  %s  expected=%.6f got=%.6f\n", msg, _b, _a); \
        g_fail++; \
    } \
} while (0)

/* ----------------------------------------------------------------------- */
/* Test 1: odometer_step — full traversal of a 2×3×2 grid (12 elements)   */
/* ----------------------------------------------------------------------- */
static void test_odometer_traversal(void)
{
    printf("\n=== Test 1: odometer_step 2×3×2 traversal ===\n");

    const size_t extents[3] = {2, 3, 2};
    int visited[2][3][2];
    memset(visited, 0, sizeof(visited));

    size_t coords[3] = {0, 0, 0};
    int count = 0;

    do {
        size_t c0 = coords[0], c1 = coords[1], c2 = coords[2];

        /* bounds check */
        if (c0 >= 2 || c1 >= 3 || c2 >= 2) {
            printf("  FAIL  out-of-bounds coordinate (%zu,%zu,%zu)\n",
                   c0, c1, c2);
            g_fail++;
            return;
        }
        if (visited[c0][c1][c2]++) {
            printf("  FAIL  duplicate coordinate (%zu,%zu,%zu)\n",
                   c0, c1, c2);
            g_fail++;
            return;
        }
        count++;
    } while (odometer_step(3, coords, extents));

    CHECK_SZ((size_t)count, 12, "total iterations == 12");

    /* Verify all 12 cells visited exactly once. */
    int all_visited = 1;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 2; k++)
                if (visited[i][j][k] != 1) all_visited = 0;
    CHECK(all_visited, "all 12 cells visited exactly once");

    /* Verify coords is back to all-zeros after exhaustion. */
    CHECK(coords[0] == 0 && coords[1] == 0 && coords[2] == 0,
          "coords reset to zero after exhaustion");
}

/* ----------------------------------------------------------------------- */
/* Test 2: compute_strides + compute_flat_index                            */
/* ----------------------------------------------------------------------- */
static void test_strides_and_flat_index(void)
{
    printf("\n=== Test 2: compute_strides / compute_flat_index ===\n");

    /* rank-3: dims 2×3×4 */
    const size_t dims[3] = {2, 3, 4};
    size_t strides[3];
    compute_strides(3, dims, strides);

    /* Expected row-major strides: [12, 4, 1] */
    CHECK_SZ(strides[0], 12, "strides[0]==12");
    CHECK_SZ(strides[1],  4, "strides[1]==4");
    CHECK_SZ(strides[2],  1, "strides[2]==1");

    /* spot-check flat indices */
    { size_t c[] = {0,0,0}; CHECK_SZ(compute_flat_index(3,c,strides), 0,  "idx(0,0,0)==0"); }
    { size_t c[] = {0,0,1}; CHECK_SZ(compute_flat_index(3,c,strides), 1,  "idx(0,0,1)==1"); }
    { size_t c[] = {0,1,0}; CHECK_SZ(compute_flat_index(3,c,strides), 4,  "idx(0,1,0)==4"); }
    { size_t c[] = {1,0,0}; CHECK_SZ(compute_flat_index(3,c,strides), 12, "idx(1,0,0)==12"); }
    { size_t c[] = {1,2,3}; CHECK_SZ(compute_flat_index(3,c,strides), 23, "idx(1,2,3)==23"); }

    /* rank-1 edge case */
    const size_t dims1[1] = {5};
    size_t strides1[1];
    compute_strides(1, dims1, strides1);
    CHECK_SZ(strides1[0], 1, "rank-1 strides[0]==1");
}

/* ----------------------------------------------------------------------- */
/* Test 3: tensor_permute — rank-3, perm [2,0,1]                          */
/*                                                                          */
/* src: 2×3×4, sequential values src[i*12+j*4+k] = i*12+j*4+k            */
/* perm = {2,0,1}: dst_dim0←src_dim2, dst_dim1←src_dim0, dst_dim2←src_dim1*/
/* dst shape: 4×2×3 (perm_nominal = [4,2,3])                              */
/* dst strides: [6,3,1]                                                    */
/* expected: dst[k*6+i*3+j] = i*12+j*4+k                                  */
/* ----------------------------------------------------------------------- */
static void test_permute_rank3(void)
{
    printf("\n=== Test 3: tensor_permute rank-3, perm [2,0,1] ===\n");

    /* Build sequential source buffer (24 elements). */
    double src[24];
    for (int n = 0; n < 24; n++) src[n] = (double)n;

    double dst[24];
    memset(dst, 0, sizeof(dst));

    const size_t nominal_dims[3] = {2, 3, 4};
    /* physical == nominal (full tile, no boundary clamping here) */
    const size_t phys[3]         = {2, 3, 4};
    const int    perm[3]         = {2, 0, 1};

    tensor_permute(src, dst, 3, phys, nominal_dims, perm, sizeof(double));

    /*
     * Verify: dst[k*6 + i*3 + j] == i*12 + j*4 + k
     * for all i in [0,2), j in [0,3), k in [0,4).
     */
    int ok = 1;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                double expected = (double)(i*12 + j*4 + k);
                double got      = dst[k*6 + i*3 + j];
                if (fabs(got - expected) > 0.5) {
                    printf("  FAIL  dst[k=%d,i=%d,j=%d]: expected %.0f got %.0f\n",
                           k, i, j, expected, got);
                    ok = 0;
                }
            }
        }
    }
    CHECK(ok, "all 24 elements correct after perm [2,0,1]");
}

/* ----------------------------------------------------------------------- */
/* Test 4: tensor_permute — rank-2 transpose                               */
/*                                                                          */
/* src: 3×4, src[i*4+j] = i*4+j                                           */
/* perm = {1,0}: dst[j*3+i] = src[i*4+j]                                  */
/* nominal == physical (full tile)                                         */
/* ----------------------------------------------------------------------- */
static void test_permute_transpose(void)
{
    printf("\n=== Test 4: tensor_permute rank-2 transpose, perm [1,0] ===\n");

    double src[12], dst[12];
    for (int n = 0; n < 12; n++) src[n] = (double)n;
    memset(dst, 0, sizeof(dst));

    const size_t nominal_dims[2] = {3, 4};
    const size_t phys[2]         = {3, 4};
    const int    perm[2]         = {1, 0};

    tensor_permute(src, dst, 2, phys, nominal_dims, perm, sizeof(double));

    /* perm_nominal_dims = {4, 3}, dst_strides = {3, 1}
     * dst[j*3 + i] = src[i*4 + j] */
    int ok = 1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            double expected = (double)(i*4 + j);
            double got      = dst[j*3 + i];
            if (fabs(got - expected) > 0.5) {
                printf("  FAIL  dst[j=%d,i=%d]: expected %.0f got %.0f\n",
                       j, i, expected, got);
                ok = 0;
            }
        }
    }
    CHECK(ok, "all 12 elements correct after rank-2 transpose");
}

/* ----------------------------------------------------------------------- */
/* Test 5: boundary clamp — physical_extents < nominal_chunk_dims          */
/*                                                                          */
/* Simulate an edge HDF5 chunk: nominal 4×4, physical 3×2 (zero-padded).  */
/* perm = {1,0}: transpose.                                                */
/*                                                                          */
/* src layout (row-major stride=4):                                        */
/*   col:  0   1   2   3                                                    */
/* row 0:  0   1   2   3                                                    */
/* row 1:  4   5   6   7                                                    */
/* row 2:  8   9  10  11                                                    */
/* row 3: 12  13  14  15   ← row 3 is padding, NOT in physical_extents     */
/*                                                                          */
/* Valid region is rows 0-2, cols 0-1.  Expected writes to dst:            */
/*   dst[j*4+i] = src[i*4+j]                                               */
/*   (j=0,i=0): dst[0] = src[0]  = 0                                       */
/*   (j=0,i=1): dst[1] = src[4]  = 4                                       */
/*   (j=0,i=2): dst[2] = src[8]  = 8                                       */
/*   (j=1,i=0): dst[4] = src[1]  = 1                                       */
/*   (j=1,i=1): dst[5] = src[5]  = 5                                       */
/*   (j=1,i=2): dst[6] = src[9]  = 9                                       */
/* All other 10 cells must remain 0 (untouched).                           */
/* ----------------------------------------------------------------------- */
static void test_permute_boundary_clamp(void)
{
    printf("\n=== Test 5: tensor_permute boundary clamp (4×4 nominal, 3×2 physical) ===\n");

    /* nominal 4×4 = 16 elements */
    double src[16], dst[16];
    for (int n = 0; n < 16; n++) src[n] = (double)n;
    memset(dst, 0, sizeof(dst));   /* explicit zero-init so untouched cells stay 0 */

    const size_t nominal_dims[2] = {4, 4};
    const size_t phys[2]         = {3, 2};   /* only 3 rows, 2 cols are real */
    const int    perm[2]         = {1, 0};   /* transpose */

    tensor_permute(src, dst, 2, phys, nominal_dims, perm, sizeof(double));

    /*
     * perm_nominal_dims = {4, 4}, dst_strides = {4, 1}
     * For i in [0,3), j in [0,2): dst[j*4+i] = src[i*4+j]
     */
    struct { int j; int i; double expected; } expected_writes[] = {
        {0, 0, 0.0}, {0, 1, 4.0}, {0, 2, 8.0},
        {1, 0, 1.0}, {1, 1, 5.0}, {1, 2, 9.0},
    };
    int n_writes = 6;

    /* Mark which dst cells should have been written. */
    int touched[16] = {0};
    int ok = 1;

    for (int w = 0; w < n_writes; w++) {
        int j   = expected_writes[w].j;
        int i   = expected_writes[w].i;
        double e = expected_writes[w].expected;
        int idx  = j * 4 + i;
        touched[idx] = 1;
        if (fabs(dst[idx] - e) > 0.5) {
            printf("  FAIL  dst[j=%d,i=%d] (idx=%d): expected %.0f got %.0f\n",
                   j, i, idx, e, dst[idx]);
            ok = 0;
        }
    }
    CHECK(ok, "6 physical cells written with correct values");

    /* Verify untouched cells remain zero. */
    int no_spill = 1;
    for (int n = 0; n < 16; n++) {
        if (!touched[n] && dst[n] != 0.0) {
            printf("  FAIL  padding cell dst[%d] was overwritten (got %.0f)\n",
                   n, dst[n]);
            no_spill = 0;
        }
    }
    CHECK(no_spill, "10 padding cells remain zero (no out-of-bounds write)");
    CHECK_SZ(3*2, 6, "only 6 elements in physical footprint (sanity)");
}

/* ----------------------------------------------------------------------- */
/* Test 6: identity permutation — tensor_permute with perm [0,1,2]        */
/* ----------------------------------------------------------------------- */
static void test_permute_identity(void)
{
    printf("\n=== Test 6: identity permutation, perm [0,1,2] ===\n");

    double src[24], dst[24];
    for (int n = 0; n < 24; n++) src[n] = (double)(n * 3 + 7);
    memset(dst, 0, sizeof(dst));

    const size_t nominal_dims[3] = {2, 3, 4};
    const size_t phys[3]         = {2, 3, 4};
    const int    perm[3]         = {0, 1, 2};   /* identity */

    tensor_permute(src, dst, 3, phys, nominal_dims, perm, sizeof(double));

    int ok = 1;
    for (int n = 0; n < 24; n++) {
        if (fabs(dst[n] - src[n]) > 0.5) {
            printf("  FAIL  dst[%d]=%.0f != src[%d]=%.0f\n",
                   n, dst[n], n, src[n]);
            ok = 0;
        }
    }
    CHECK(ok, "identity permutation: dst == src element-wise");
}

/* ----------------------------------------------------------------------- */
/* Test 7: rank-1 edge case                                                */
/* ----------------------------------------------------------------------- */
static void test_rank1(void)
{
    printf("\n=== Test 7: rank-1 odometer and permute ===\n");

    /* odometer: 5 elements */
    const size_t ext[1] = {5};
    size_t coords[1] = {0};
    int count = 0;
    do { count++; } while (odometer_step(1, coords, ext));
    CHECK_SZ((size_t)count, 5, "rank-1 odometer: 5 iterations");
    CHECK(coords[0] == 0, "rank-1 odometer resets to 0");

    /* permute rank-1: identity perm, nominal 8, physical 5 */
    double src[8], dst[8];
    for (int n = 0; n < 8; n++) src[n] = (double)n;
    memset(dst, 0, sizeof(dst));
    const size_t nom[1] = {8};
    const size_t phy[1] = {5};
    const int    p[1]   = {0};
    tensor_permute(src, dst, 1, phy, nom, p, sizeof(double));

    int ok = 1;
    for (int n = 0; n < 5; n++)
        if (fabs(dst[n] - src[n]) > 0.5) { ok = 0; }
    for (int n = 5; n < 8; n++)
        if (dst[n] != 0.0) { ok = 0; }
    CHECK(ok, "rank-1 permute: 5 copied, 3 padding untouched");
}

/* ----------------------------------------------------------------------- */
/* Test 8: rank-0 (scalar)                                                 */
/* ----------------------------------------------------------------------- */
static void test_rank0_scalar(void)
{
    printf("\n=== Test 8: rank-0 scalar tensor_permute ===\n");
    double src = 42.0, dst = 0.0;
    /* rank=0: no perm, no extents needed */
    tensor_permute(&src, &dst, 0, NULL, NULL, NULL, sizeof(double));
    CHECK_DBL(dst, 42.0, 1e-12, "rank-0 copy: dst==42.0");
}

/* ----------------------------------------------------------------------- */
/* main                                                                     */
/* ----------------------------------------------------------------------- */

int main(void)
{
    printf("=== odometer + tensor_permute unit tests ===\n");

    test_odometer_traversal();
    test_strides_and_flat_index();
    test_permute_rank3();
    test_permute_transpose();
    test_permute_boundary_clamp();
    test_permute_identity();
    test_rank1();
    test_rank0_scalar();

    printf("\n--- Results: %d passed, %d failed ---\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
