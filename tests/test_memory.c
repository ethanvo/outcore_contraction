#include "memory.h"
#include <assert.h>
#include <stdio.h>

int main() {
  printf("--- Testing Memory Manager ---\n");

  // Create a tiny pool: 3 pages, 10 doubles each
  BufferPool *pool = pool_create(3, 10);

  // 1. Acquire all pages
  int id1, id2, id3;
  double *p1 = pool_acquire(pool, &id1);
  double *p2 = pool_acquire(pool, &id2);
  double *p3 = pool_acquire(pool, &id3);

  printf("Acquired IDs: %d, %d, %d\n", id1, id2,
         id3); // Should be 2, 1, 0 (Stack order)

  // Check pointer arithmetic distance
  // p1 should be exactly 10 doubles away from p2 (or p2 from p1 depending on
  // stack order)
  long diff = p1 - p2;
  printf("Distance between p1 and p2: %ld doubles\n", diff);

  // Write data to ensure separation
  p1[0] = 1.1;
  p2[0] = 2.2;
  p3[0] = 3.3;

  // 2. Try to acquire when empty
  int id_fail;
  double *p_fail = pool_acquire(pool, &id_fail);
  if (p_fail == NULL)
    printf("Pool correctly refused 4th request.\n");

  // 3. Release a page
  printf("Releasing ID %d...\n", id2);
  pool_release(pool, id2);

  // 4. Acquire again (should get the one we just released)
  int id_new;
  double *p_new = pool_acquire(pool, &id_new);
  printf("Acquired new ID: %d (Expected %d)\n", id_new, id2);

  // Verify data persistence (Memory is not cleared on release/acquire by
  // default)
  printf("Data in reused page: %f (Expected 2.200)\n", p_new[0]);

  pool_destroy(pool);
  return 0;
}
