#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <list>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace outcore {

struct BlockDescriptor {
  std::vector<std::size_t> tile_shape;
  std::vector<std::size_t> chunk_shape;
  std::size_t bytes = 0;
};

struct BlockMetadata {
  bool is_zero = false;
  std::string hdf5_path;
  BlockDescriptor descriptor;
};

class MetadataRegistry {
 public:
  void Register(const std::string &key, BlockMetadata metadata);
  std::optional<BlockMetadata> Lookup(const std::string &key) const;
  std::vector<std::string> Keys() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, BlockMetadata> entries_;
};

struct CacheEntry {
  std::string key;
  std::vector<float> data;
};

class LruCache {
 public:
  explicit LruCache(std::size_t max_bytes);
  std::optional<CacheEntry> Get(const std::string &key);
  void Put(const std::string &key, std::vector<float> data);
  std::size_t CurrentBytes() const;

 private:
  void EvictIfNeeded();
  void Touch(const std::string &key);

  std::size_t max_bytes_;
  std::size_t current_bytes_ = 0;
  std::list<std::string> lru_;
  std::unordered_map<std::string, std::list<std::string>::iterator> lru_lookup_;
  std::unordered_map<std::string, CacheEntry> entries_;
  mutable std::mutex mutex_;
};

class DoubleBuffer {
 public:
  explicit DoubleBuffer(std::size_t buffer_bytes = 0);
  void Resize(std::size_t buffer_bytes);
  std::vector<float> &WriteBuffer();
  const std::vector<float> &ReadBuffer() const;
  void Swap();

 private:
  std::vector<float> buffers_[2];
  std::size_t write_index_ = 0;
};

struct PrefetchRequest {
  std::string key;
  BlockDescriptor descriptor;
};

class IOThread {
 public:
  using FetchCallback = std::function<std::vector<float>(const PrefetchRequest &)>;

  explicit IOThread(FetchCallback fetch_cb);
  ~IOThread();
  void Start();

  void Enqueue(const PrefetchRequest &request);
  std::optional<CacheEntry> PopReady();
  std::optional<CacheEntry> WaitReady(std::chrono::milliseconds timeout);
  void Stop();
  std::size_t Pending() const;

 private:
  void WorkerLoop();

  FetchCallback fetch_cb_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable ready_cv_;
  std::deque<PrefetchRequest> queue_;
  std::deque<CacheEntry> ready_;
  std::thread worker_;
  std::atomic<bool> stop_{false};
};

class OutcoreEngine {
 public:
  explicit OutcoreEngine(std::size_t cache_bytes);

  void RegisterBlock(const std::string &key, BlockMetadata metadata);
  void QueuePrefetch(const std::string &key);
  bool TryConsume();
  bool WaitConsume(std::chrono::milliseconds timeout);
  std::size_t CacheBytes() const;
  std::optional<CacheEntry> LookupCache(const std::string &key);

  static BlockDescriptor AlignChunkToTile(const std::vector<std::size_t> &tile_shape,
                                         const std::vector<std::size_t> &chunk_alignment,
                                         std::size_t element_bytes);

 private:
  MetadataRegistry metadata_;
  LruCache cache_;
  DoubleBuffer double_buffer_;
  IOThread io_thread_;
  mutable std::mutex buffer_mutex_;
};

}  // namespace outcore
