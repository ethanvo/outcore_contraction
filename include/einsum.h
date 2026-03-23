#ifndef EINSUM_H
#define EINSUM_H

#include <stddef.h>   /* size_t */

#ifndef MAX_RANK
#  define MAX_RANK 8   /* must match registry.h */
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * contraction_plan_t — output of einsum_parse().
 *
 * Given an expression like "ijab,akbl->klji":
 *
 *   rank_A = 4, rank_B = 4, rank_C = 4
 *   n_contracted = 2   (indices a, b)
 *   n_free_A     = 2   (indices i, j — appear in A and C)
 *   n_free_B     = 2   (indices k, l — appear in B and C)
 *
 * perm_A[rank_A]:
 *   Maps original A dimension positions to the matricized A layout.
 *   Layout = [free_A dims in A order | contracted dims in A order].
 *   Example: [0,1,2,3] (free={0,1}, contracted={2,3}).
 *
 * perm_B[rank_B]:
 *   Maps original B dimension positions to the matricized B layout.
 *   Layout = [contracted dims in same order as perm_A | free_B dims in B order].
 *   Example: [0,2,1,3] (contracted={0,2}, free={1,3}).
 *
 * perm_C[rank_C]:
 *   For each dimension d of C, perm_C[d] gives the index into the BLAS
 *   output buffer, which is laid out as [free_A_0,...,free_B_0,...].
 *   free_A occupies positions 0..n_free_A-1 (in A order);
 *   free_B occupies positions n_free_A..n_free_A+n_free_B-1 (in B order).
 *   Example: [2,3,1,0] for C="klji" with blas layout [i,j,k,l].
 *
 * contracted_chars[n_contracted]:
 *   The actual index letters that are contracted, in A order.
 *   Stored for diagnostics / pretty-printing.
 *
 * free_A_chars[n_free_A], free_B_chars[n_free_B]:
 *   Free index letters in the order they appear in A (and B respectively).
 */
typedef struct {
    int rank_A;
    int rank_B;
    int rank_C;

    int n_contracted;
    int n_free_A;
    int n_free_B;

    int perm_A[MAX_RANK];   /* length rank_A */
    int perm_B[MAX_RANK];   /* length rank_B */
    int perm_C[MAX_RANK];   /* length rank_C */

    char contracted_chars[MAX_RANK];
    char free_A_chars[MAX_RANK];
    char free_B_chars[MAX_RANK];
} contraction_plan_t;

/*
 * einsum_parse — parse an einsum expression and populate *plan.
 *
 * expr   : null-terminated string in the form "subscripts_A,subscripts_B->subscripts_C"
 *          e.g. "ijab,akbl->klji" or "mk,kn->mn"
 * plan   : output; all fields populated on success.
 *
 * Returns 0 on success, -1 on error (malformed expression, index appears
 * in A but not in B or C, rank_C != n_free_A + n_free_B, etc.).
 *
 * Constraints:
 *   - Index labels must be lower-case ASCII letters (a-z).
 *   - Each label may appear at most once per tensor operand.
 *   - Contracted indices must appear in both A and B and must NOT appear in C.
 *   - Free-A indices appear in A and C (may optionally appear in B if also in C
 *     — "trace-like" reuse — but this is treated as a free index, not contracted).
 *   - rank_A, rank_B, rank_C <= MAX_RANK.
 */
int einsum_parse(const char *expr, contraction_plan_t *plan);

/*
 * einsum_sprint_plan — write a human-readable description of plan into buf.
 * buf must be at least 512 bytes.  Returns buf.
 */
char *einsum_sprint_plan(const contraction_plan_t *plan, char *buf, size_t bufsz);

#ifdef __cplusplus
}
#endif

#endif /* EINSUM_H */
