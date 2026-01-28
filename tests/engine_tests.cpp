#include "outcore/engine.hpp"

#include <cassert>
#include <chrono>
#include <thread>

int main() {
  using namespace outcore;

  OutcoreEngine engine(1024);
  BlockDescriptor descriptor = OutcoreEngine::AlignChunkToTile({4, 4}, {2, 2}, sizeof(float));
  BlockMetadata metadata{false, "/tensor/block0", descriptor};
  engine.RegisterBlock("block0", metadata);
  engine.QueuePrefetch("block0");

  bool consumed = false;
  for (int i = 0; i < 50; ++i) {
    if (engine.TryConsume()) {
      consumed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  assert(consumed && "Expected IO thread to provide data");
  assert(engine.CacheBytes() > 0 && "Cache should have data after consume");

  BlockDescriptor aligned = OutcoreEngine::AlignChunkToTile({7, 3}, {4, 2}, sizeof(float));
  assert(aligned.chunk_shape[0] == 8);
  assert(aligned.chunk_shape[1] == 4);

  return 0;
}
