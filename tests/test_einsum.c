#include "einsum.h"
#include <stdio.h>
#include <string.h>

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

#define CHECK_INT(a, b, msg) do { \
    if ((a) == (b)) { \
        printf("  PASS  %s  (%d)\n", msg, (a)); \
        g_pass++; \
    } else { \
        printf("  FAIL  %s  expected=%d got=%d\n", msg, (b), (a)); \
        g_fail++; \
    } \
} while (0)

static void check_perm(const int *perm, const int *expected, int n, const char *label)
{
    for (int i = 0; i < n; i++) {
        if (perm[i] != expected[i]) {
            printf("  FAIL  %s  mismatch at [%d]: expected %d got %d\n",
                   label, i, expected[i], perm[i]);
            g_fail++;
            return;
        }
    }
    printf("  PASS  %s\n", label);
    g_pass++;
}

/* ----------------------------------------------------------------------- */
/* Test 1: rank-2 matmul  mk,kn->mn                                        */
/*                                                                          */
/*   A: m(0) k(1)     B: k(0) n(1)     C: m(0) n(1)                       */
/*   contracted = {k}                                                       */
/*   free_A = {m}  (A dim 0)                                               */
/*   free_B = {n}  (B dim 1)                                               */
/*   perm_A = [0, 1]   (free_A={0}, contracted={1})                        */
/*   perm_B = [0, 1]   (contracted={0}, free_B={1})                        */
/*   blas layout = [m(0), n(1)]                                            */
/*   perm_C = [0, 1]                                                       */
/* ----------------------------------------------------------------------- */
static void test_rank2_matmul(void)
{
    printf("\n=== Test 1: mk,kn->mn ===\n");
    contraction_plan_t plan;
    int rc = einsum_parse("mk,kn->mn", &plan);

    CHECK(rc == 0, "parse returns 0");
    CHECK_INT(plan.rank_A, 2, "rank_A");
    CHECK_INT(plan.rank_B, 2, "rank_B");
    CHECK_INT(plan.rank_C, 2, "rank_C");
    CHECK_INT(plan.n_contracted, 1, "n_contracted");
    CHECK_INT(plan.n_free_A,     1, "n_free_A");
    CHECK_INT(plan.n_free_B,     1, "n_free_B");
    CHECK(plan.contracted_chars[0] == 'k', "contracted_chars[0]=='k'");
    CHECK(plan.free_A_chars[0]     == 'm', "free_A_chars[0]=='m'");
    CHECK(plan.free_B_chars[0]     == 'n', "free_B_chars[0]=='n'");

    { int e[] = {0, 1}; check_perm(plan.perm_A, e, 2, "perm_A"); }
    { int e[] = {0, 1}; check_perm(plan.perm_B, e, 2, "perm_B"); }
    { int e[] = {0, 1}; check_perm(plan.perm_C, e, 2, "perm_C"); }
}

/* ----------------------------------------------------------------------- */
/* Test 2: rank-2 with transposed operands  im,jm->ij                     */
/*                                                                          */
/*   A: i(0) m(1)     B: j(0) m(1)     C: i(0) j(1)                       */
/*   contracted = {m}                                                       */
/*   free_A = {i}  (A dim 0)                                               */
/*   free_B = {j}  (B dim 0)                                               */
/*   perm_A = [0, 1]   (free_A={0}, contracted={1})                        */
/*   perm_B = [1, 0]   (contracted={1}, free_B={0})                        */
/*   blas layout = [i(0), j(1)]                                            */
/*   perm_C = [0, 1]                                                       */
/* ----------------------------------------------------------------------- */
static void test_rank2_transposed(void)
{
    printf("\n=== Test 2: im,jm->ij ===\n");
    contraction_plan_t plan;
    int rc = einsum_parse("im,jm->ij", &plan);

    CHECK(rc == 0, "parse returns 0");
    CHECK_INT(plan.n_contracted, 1, "n_contracted");
    CHECK_INT(plan.n_free_A,     1, "n_free_A");
    CHECK_INT(plan.n_free_B,     1, "n_free_B");
    CHECK(plan.contracted_chars[0] == 'm', "contracted_chars[0]=='m'");
    CHECK(plan.free_A_chars[0]     == 'i', "free_A_chars[0]=='i'");
    CHECK(plan.free_B_chars[0]     == 'j', "free_B_chars[0]=='j'");

    { int e[] = {0, 1}; check_perm(plan.perm_A, e, 2, "perm_A"); }
    { int e[] = {1, 0}; check_perm(plan.perm_B, e, 2, "perm_B"); }
    { int e[] = {0, 1}; check_perm(plan.perm_C, e, 2, "perm_C"); }
}

/* ----------------------------------------------------------------------- */
/* Test 3: rank-4 contraction  ijab,akbl->klji                            */
/*                                                                          */
/*   A: i(0) j(1) a(2) b(3)                                                */
/*   B: a(0) k(1) b(2) l(3)                                                */
/*   C: k(0) l(1) j(2) i(3)                                                */
/*   contracted = {a, b}  (in A: dims 2,3; in B: dims 0,2)                */
/*   free_A = {i,j}  (A dims 0,1)                                          */
/*   free_B = {k,l}  (B dims 1,3)                                          */
/*   perm_A = [0, 1, 2, 3]                                                 */
/*   perm_B = [0, 2, 1, 3]  (contracted a→B[0], b→B[2]; free k→B[1], l→B[3]) */
/*   blas layout = [i(0), j(1), k(2), l(3)]                               */
/*   perm_C: C[0]=k=blas[2], C[1]=l=blas[3], C[2]=j=blas[1], C[3]=i=blas[0] */
/*         = [2, 3, 1, 0]                                                  */
/* ----------------------------------------------------------------------- */
static void test_rank4_ijab_akbl(void)
{
    printf("\n=== Test 3: ijab,akbl->klji ===\n");
    contraction_plan_t plan;
    int rc = einsum_parse("ijab,akbl->klji", &plan);

    CHECK(rc == 0, "parse returns 0");
    CHECK_INT(plan.rank_A, 4, "rank_A");
    CHECK_INT(plan.rank_B, 4, "rank_B");
    CHECK_INT(plan.rank_C, 4, "rank_C");
    CHECK_INT(plan.n_contracted, 2, "n_contracted");
    CHECK_INT(plan.n_free_A,     2, "n_free_A");
    CHECK_INT(plan.n_free_B,     2, "n_free_B");
    CHECK(plan.contracted_chars[0] == 'a', "contracted_chars[0]=='a'");
    CHECK(plan.contracted_chars[1] == 'b', "contracted_chars[1]=='b'");
    CHECK(plan.free_A_chars[0]     == 'i', "free_A_chars[0]=='i'");
    CHECK(plan.free_A_chars[1]     == 'j', "free_A_chars[1]=='j'");
    CHECK(plan.free_B_chars[0]     == 'k', "free_B_chars[0]=='k'");
    CHECK(plan.free_B_chars[1]     == 'l', "free_B_chars[1]=='l'");

    { int e[] = {0, 1, 2, 3}; check_perm(plan.perm_A, e, 4, "perm_A"); }
    { int e[] = {0, 2, 1, 3}; check_perm(plan.perm_B, e, 4, "perm_B"); }
    { int e[] = {2, 3, 1, 0}; check_perm(plan.perm_C, e, 4, "perm_C"); }
}

/* ----------------------------------------------------------------------- */
/* Test 4: rank-3 with mixed ordering  abc,cbd->ad                        */
/*                                                                          */
/*   A: a(0) b(1) c(2)                                                     */
/*   B: c(0) b(1) d(2)                                                     */
/*   C: a(0) d(1)                                                          */
/*   contracted = {b, c}  (in A dims 1,2; in B dims 1,0)                  */
/*   free_A = {a}  (A dim 0)                                               */
/*   free_B = {d}  (B dim 2)                                               */
/*   perm_A = [0, 1, 2]   (free={0}, contracted b→1 c→2)                  */
/*   perm_B = [1, 0, 2]   (contracted b→B[1], c→B[0]; free d→B[2])        */
/*   blas layout = [a(0), d(1)]                                            */
/*   perm_C = [0, 1]                                                       */
/* ----------------------------------------------------------------------- */
static void test_rank3_mixed(void)
{
    printf("\n=== Test 4: abc,cbd->ad ===\n");
    contraction_plan_t plan;
    int rc = einsum_parse("abc,cbd->ad", &plan);

    CHECK(rc == 0, "parse returns 0");
    CHECK_INT(plan.rank_A, 3, "rank_A");
    CHECK_INT(plan.rank_B, 3, "rank_B");
    CHECK_INT(plan.rank_C, 2, "rank_C");
    CHECK_INT(plan.n_contracted, 2, "n_contracted");
    CHECK_INT(plan.n_free_A,     1, "n_free_A");
    CHECK_INT(plan.n_free_B,     1, "n_free_B");
    CHECK(plan.contracted_chars[0] == 'b', "contracted_chars[0]=='b'");
    CHECK(plan.contracted_chars[1] == 'c', "contracted_chars[1]=='c'");
    CHECK(plan.free_A_chars[0]     == 'a', "free_A_chars[0]=='a'");
    CHECK(plan.free_B_chars[0]     == 'd', "free_B_chars[0]=='d'");

    { int e[] = {0, 1, 2}; check_perm(plan.perm_A, e, 3, "perm_A"); }
    { int e[] = {1, 0, 2}; check_perm(plan.perm_B, e, 3, "perm_B"); }
    { int e[] = {0, 1};    check_perm(plan.perm_C, e, 2, "perm_C"); }
}

/* ----------------------------------------------------------------------- */
/* Test 5: C output is a permutation of BLAS layout  ij,jk->ki            */
/*                                                                          */
/*   A: i(0) j(1)     B: j(0) k(1)     C: k(0) i(1)                       */
/*   contracted = {j}  (A dim 1, B dim 0)                                  */
/*   free_A = {i}  (A dim 0)                                               */
/*   free_B = {k}  (B dim 1)                                               */
/*   perm_A = [0, 1]                                                       */
/*   perm_B = [0, 1]                                                       */
/*   blas layout = [i(0), k(1)]                                            */
/*   perm_C: C[0]=k=blas[1], C[1]=i=blas[0]  → [1, 0]                    */
/* ----------------------------------------------------------------------- */
static void test_rank2_permuted_output(void)
{
    printf("\n=== Test 5: ij,jk->ki ===\n");
    contraction_plan_t plan;
    int rc = einsum_parse("ij,jk->ki", &plan);

    CHECK(rc == 0, "parse returns 0");
    CHECK_INT(plan.n_contracted, 1, "n_contracted");
    CHECK(plan.contracted_chars[0] == 'j', "contracted=='j'");

    { int e[] = {0, 1}; check_perm(plan.perm_A, e, 2, "perm_A"); }
    { int e[] = {0, 1}; check_perm(plan.perm_B, e, 2, "perm_B"); }
    { int e[] = {1, 0}; check_perm(plan.perm_C, e, 2, "perm_C"); }
}

/* ----------------------------------------------------------------------- */
/* Test 6: scalar output (inner product)  ij,ij->                         */
/*                                                                          */
/*   A: i(0) j(1)     B: i(0) j(1)     C: (rank 0)                        */
/*   contracted = {i, j}                                                   */
/*   free_A = {}  free_B = {}                                              */
/*   perm_A = [0, 1]  perm_B = [0, 1]  perm_C = []                        */
/* ----------------------------------------------------------------------- */
static void test_scalar_output(void)
{
    printf("\n=== Test 6: ij,ij-> (scalar inner product) ===\n");
    contraction_plan_t plan;
    int rc = einsum_parse("ij,ij->", &plan);

    CHECK(rc == 0, "parse returns 0");
    CHECK_INT(plan.rank_C, 0, "rank_C==0");
    CHECK_INT(plan.n_contracted, 2, "n_contracted");
    CHECK_INT(plan.n_free_A,     0, "n_free_A");
    CHECK_INT(plan.n_free_B,     0, "n_free_B");
    CHECK(plan.contracted_chars[0] == 'i', "contracted[0]=='i'");
    CHECK(plan.contracted_chars[1] == 'j', "contracted[1]=='j'");

    { int e[] = {0, 1}; check_perm(plan.perm_A, e, 2, "perm_A"); }
    { int e[] = {0, 1}; check_perm(plan.perm_B, e, 2, "perm_B"); }
    /* perm_C is length 0 — nothing to check */
    printf("  PASS  perm_C (empty, rank_C=0)\n");
    g_pass++;
}

/* ----------------------------------------------------------------------- */
/* Test 7: error cases                                                      */
/* ----------------------------------------------------------------------- */
static void test_error_cases(void)
{
    printf("\n=== Test 7: error cases ===\n");
    contraction_plan_t plan;

    /* Missing arrow */
    CHECK(einsum_parse("ij,jk", &plan) == -1, "missing arrow → -1");

    /* Missing comma */
    CHECK(einsum_parse("ij->ij", &plan) == -1, "missing comma → -1");

    /* Index in A but not in B or C */
    CHECK(einsum_parse("ixj,jk->ik", &plan) == -1, "orphan index in A → -1");

    /* Index in C but in neither A nor B */
    CHECK(einsum_parse("ij,jk->ikz", &plan) == -1, "unknown index in C → -1");

    /* rank_C mismatch (2 free but 3 in C) */
    CHECK(einsum_parse("ij,jk->ikk", &plan) == -1, "rank_C mismatch → -1");

    /* Upper-case not allowed */
    CHECK(einsum_parse("Ij,jk->ik", &plan) == -1, "upper-case → -1");

    /* NULL arguments */
    CHECK(einsum_parse(NULL, &plan) == -1, "NULL expr → -1");
    CHECK(einsum_parse("ij,jk->ik", NULL) == -1, "NULL plan → -1");
}

/* ----------------------------------------------------------------------- */
/* Test 8: sprint round-trip smoke test                                    */
/* ----------------------------------------------------------------------- */
static void test_sprint(void)
{
    printf("\n=== Test 8: einsum_sprint_plan smoke test ===\n");
    contraction_plan_t plan;
    einsum_parse("ijab,akbl->klji", &plan);

    char buf[1024];
    char *result = einsum_sprint_plan(&plan, buf, sizeof(buf));
    CHECK(result == buf, "sprint returns buf pointer");
    CHECK(strlen(buf) > 0, "buf non-empty");
    CHECK(strstr(buf, "rank_A=4") != NULL, "contains rank_A=4");
    CHECK(strstr(buf, "perm_B") != NULL, "contains perm_B");
    printf("  Sprint output:\n");
    /* indent each line */
    const char *p = buf;
    while (*p) {
        printf("    ");
        while (*p && *p != '\n') { putchar(*p++); }
        putchar('\n');
        if (*p == '\n') p++;
    }
}

/* ----------------------------------------------------------------------- */
/* main                                                                     */
/* ----------------------------------------------------------------------- */

int main(void)
{
    printf("=== einsum parser unit tests ===\n");

    test_rank2_matmul();
    test_rank2_transposed();
    test_rank4_ijab_akbl();
    test_rank3_mixed();
    test_rank2_permuted_output();
    test_scalar_output();
    test_error_cases();
    test_sprint();

    printf("\n--- Results: %d passed, %d failed ---\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
