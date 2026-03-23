#include "engine.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    /*
     * Contract A(M×K) * B(K×N) → C(M×N).
     *
     * Pool size and chunk granularity are derived automatically from
     * physical RAM by run_contraction / query_physical_ram.
     *
     * To exercise a multi-tile out-of-core run, increase the matrix
     * dimensions in gen_data.c.
     */
    int status = run_contraction("A.h5", "MatrixA",
                                 "B.h5", "MatrixB",
                                 "C.h5", "MatrixC");
    if (status != 0) {
        fprintf(stderr, "Contraction failed.\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
