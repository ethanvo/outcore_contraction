#include "outcore/engine.hpp"

#include <cassert>
int main() {
  using namespace outcore;

  OutcoreEngine engine(1024);
  BlockDescriptor descriptor = OutcoreEngine::AlignChunkToTile({4, 4}, {2, 2}, sizeof(float));
  BlockMetadata metadata{false, "/tensor/block0", descriptor};
  engine.RegisterBlock("block0", metadata);
  engine.QueuePrefetch("block0");

  bool consumed = engine.WaitConsume(std::chrono::milliseconds(250));
  assert(consumed && "Expected IO thread to provide data");
  auto cached = engine.LookupCache("block0");
  assert(cached.has_value());
  std::size_t expected_elements = descriptor.chunk_shape[0] * descriptor.chunk_shape[1];
  assert(cached->data.size() == expected_elements);
  assert(engine.CacheBytes() > 0 && "Cache should have data after consume");

  BlockDescriptor aligned = OutcoreEngine::AlignChunkToTile({7, 3}, {4, 2}, sizeof(float));
  assert(aligned.chunk_shape[0] == 8);
  assert(aligned.chunk_shape[1] == 4);

  return 0;
}
