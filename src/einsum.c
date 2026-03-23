#include "einsum.h"
#include <string.h>
#include <stdio.h>
#include <stddef.h>

/* ----------------------------------------------------------------------- */
/* einsum_parse                                                             */
/* ----------------------------------------------------------------------- */

int einsum_parse(const char *expr, contraction_plan_t *plan)
{
    if (!expr || !plan) return -1;

    /* Locate ',' and "->" separators */
    const char *comma = strchr(expr, ',');
    const char *arrow = strstr(expr, "->");
    if (!comma || !arrow || comma >= arrow) return -1;

    int la = (int)(comma - expr);
    int lb = (int)(arrow - (comma + 1));
    int lc = (int)strlen(arrow + 2);

    if (la < 1 || lb < 1 || lc < 0) return -1;
    if (la > MAX_RANK || lb > MAX_RANK || lc > MAX_RANK) return -1;

    char labels_A[MAX_RANK + 1];
    char labels_B[MAX_RANK + 1];
    char labels_C[MAX_RANK + 1];

    memcpy(labels_A, expr,        la); labels_A[la] = '\0';
    memcpy(labels_B, comma + 1,   lb); labels_B[lb] = '\0';
    memcpy(labels_C, arrow + 2,   lc); labels_C[lc] = '\0';

    plan->rank_A = la;
    plan->rank_B = lb;
    plan->rank_C = lc;

    /* Validate: only lower-case letters */
    for (int i = 0; i < la; i++)
        if (labels_A[i] < 'a' || labels_A[i] > 'z') return -1;
    for (int i = 0; i < lb; i++)
        if (labels_B[i] < 'a' || labels_B[i] > 'z') return -1;
    for (int i = 0; i < lc; i++)
        if (labels_C[i] < 'a' || labels_C[i] > 'z') return -1;

    /* Build membership lookup tables */
    int in_B[26] = {0};
    int in_C[26] = {0};

    for (int i = 0; i < lb; i++) in_B[(int)(labels_B[i] - 'a')] = 1;
    for (int i = 0; i < lc; i++) in_C[(int)(labels_C[i] - 'a')] = 1;

    /* --- Classify A indices -------------------------------------------- */
    /*
     * For each dim in A:
     *   if in_C → free_A (contributes a row dimension in the BLAS matrix)
     *   else if in_B → contracted (summed over)
     *   else → error (index in A but nowhere else)
     */
    int free_A_dims[MAX_RANK],  n_free_A  = 0;
    int contr_A_dims[MAX_RANK], n_contr   = 0;
    char contr_chars[MAX_RANK];

    for (int d = 0; d < la; d++) {
        int idx = (int)(labels_A[d] - 'a');
        if (in_C[idx]) {
            plan->free_A_chars[n_free_A] = labels_A[d];
            free_A_dims[n_free_A++] = d;
        } else if (in_B[idx]) {
            contr_chars[n_contr]    = labels_A[d];
            contr_A_dims[n_contr++] = d;
        } else {
            return -1;  /* index in A but not in B or C */
        }
    }

    plan->n_contracted = n_contr;
    plan->n_free_A     = n_free_A;
    memcpy(plan->contracted_chars, contr_chars,
           (size_t)n_contr * sizeof(char));

    /* perm_A = [free_A_dims... | contr_A_dims...] */
    for (int i = 0; i < n_free_A; i++)
        plan->perm_A[i] = free_A_dims[i];
    for (int i = 0; i < n_contr; i++)
        plan->perm_A[n_free_A + i] = contr_A_dims[i];

    /* --- Classify B indices -------------------------------------------- */
    /*
     * Mark which chars are contracted (appear in A but not C).
     * Build:
     *   contr_B_dims[i] = dimension of contr_chars[i] inside B
     *                     (preserves contracted-index ordering from A)
     *   free_B_dims     = B dims not in contr_chars, in B order
     */
    int contracted_flag[26] = {0};
    for (int i = 0; i < n_contr; i++)
        contracted_flag[(int)(contr_chars[i] - 'a')] = 1;

    /* Map char -> dim in B */
    int B_dim_of[26];
    memset(B_dim_of, -1, sizeof(B_dim_of));
    for (int d = 0; d < lb; d++)
        B_dim_of[(int)(labels_B[d] - 'a')] = d;

    /* Contracted dims in B, ordered to match A's contracted order */
    int contr_B_dims[MAX_RANK];
    for (int i = 0; i < n_contr; i++) {
        int bd = B_dim_of[(int)(contr_chars[i] - 'a')];
        if (bd < 0) return -1;  /* contracted char missing from B */
        contr_B_dims[i] = bd;
    }

    /* Free dims of B: in B, not contracted, must be in C */
    int free_B_dims[MAX_RANK], n_free_B = 0;
    for (int d = 0; d < lb; d++) {
        int idx = (int)(labels_B[d] - 'a');
        if (!contracted_flag[idx]) {
            if (!in_C[idx]) return -1;  /* free in B but missing from C */
            plan->free_B_chars[n_free_B] = labels_B[d];
            free_B_dims[n_free_B++] = d;
        }
    }

    plan->n_free_B = n_free_B;

    if (n_free_A + n_free_B != lc) return -1;

    /* perm_B = [contr_B_dims... | free_B_dims...] */
    for (int i = 0; i < n_contr; i++)
        plan->perm_B[i] = contr_B_dims[i];
    for (int i = 0; i < n_free_B; i++)
        plan->perm_B[n_contr + i] = free_B_dims[i];

    /* --- Build perm_C --------------------------------------------------- */
    /*
     * The BLAS output is laid out as:
     *   positions 0 .. n_free_A-1           → free-A indices in A order
     *   positions n_free_A .. rank_C-1      → free-B indices in B order
     *
     * For each dim d of C, perm_C[d] = position in BLAS output that
     * corresponds to label C[d].
     */
    int blas_pos[26];
    memset(blas_pos, -1, sizeof(blas_pos));

    for (int i = 0; i < n_free_A; i++)
        blas_pos[(int)(plan->free_A_chars[i] - 'a')] = i;
    for (int i = 0; i < n_free_B; i++)
        blas_pos[(int)(plan->free_B_chars[i] - 'a')] = n_free_A + i;

    for (int d = 0; d < lc; d++) {
        int bp = blas_pos[(int)(labels_C[d] - 'a')];
        if (bp < 0) return -1;
        plan->perm_C[d] = bp;
    }

    return 0;
}

/* ----------------------------------------------------------------------- */
/* einsum_sprint_plan                                                       */
/* ----------------------------------------------------------------------- */

char *einsum_sprint_plan(const contraction_plan_t *plan,
                         char *buf, size_t bufsz)
{
    int pos = 0;

#define APPEND(...) \
    pos += snprintf(buf + pos, (pos < (int)bufsz ? bufsz - (size_t)pos : 0), \
                    __VA_ARGS__)

    APPEND("contraction_plan_t {\n");
    APPEND("  rank_A=%d  rank_B=%d  rank_C=%d\n",
           plan->rank_A, plan->rank_B, plan->rank_C);
    APPEND("  n_contracted=%d  n_free_A=%d  n_free_B=%d\n",
           plan->n_contracted, plan->n_free_A, plan->n_free_B);

    APPEND("  contracted_chars = [");
    for (int i = 0; i < plan->n_contracted; i++)
        APPEND("%c", plan->contracted_chars[i]);
    APPEND("]\n");

    APPEND("  free_A_chars = [");
    for (int i = 0; i < plan->n_free_A; i++)
        APPEND("%c", plan->free_A_chars[i]);
    APPEND("]  free_B_chars = [");
    for (int i = 0; i < plan->n_free_B; i++)
        APPEND("%c", plan->free_B_chars[i]);
    APPEND("]\n");

    APPEND("  perm_A = [");
    for (int i = 0; i < plan->rank_A; i++)
        APPEND("%d%s", plan->perm_A[i], i < plan->rank_A-1 ? "," : "");
    APPEND("]\n");

    APPEND("  perm_B = [");
    for (int i = 0; i < plan->rank_B; i++)
        APPEND("%d%s", plan->perm_B[i], i < plan->rank_B-1 ? "," : "");
    APPEND("]\n");

    APPEND("  perm_C = [");
    for (int i = 0; i < plan->rank_C; i++)
        APPEND("%d%s", plan->perm_C[i], i < plan->rank_C-1 ? "," : "");
    APPEND("]\n");

    APPEND("}\n");

#undef APPEND

    return buf;
}
