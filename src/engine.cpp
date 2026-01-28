#include "outcore/engine.hpp"

#include <algorithm>
#include <stdexcept>

namespace outcore {

void MetadataRegistry::Register(const std::string &key, BlockMetadata metadata) {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_[key] = std::move(metadata);
}

std::optional<BlockMetadata> MetadataRegistry::Lookup(const std::string &key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::vector<std::string> MetadataRegistry::Keys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> keys;
  keys.reserve(entries_.size());
  for (const auto &entry : entries_) {
    keys.push_back(entry.first);
  }
  return keys;
}

LruCache::LruCache(std::size_t max_bytes) : max_bytes_(max_bytes) {}

std::optional<CacheEntry> LruCache::Get(const std::string &key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  Touch(key);
  return it->second;
}

void LruCache::Put(const std::string &key, std::vector<float> data) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(key);
  std::size_t bytes = data.size() * sizeof(float);
  if (it != entries_.end()) {
    current_bytes_ -= it->second.data.size() * sizeof(float);
    it->second.data = std::move(data);
    current_bytes_ += bytes;
    Touch(key);
  } else {
    entries_.emplace(key, CacheEntry{key, std::move(data)});
    lru_.push_front(key);
    current_bytes_ += bytes;
  }
  EvictIfNeeded();
}

void LruCache::Touch(const std::string &key) {
  auto it = std::find(lru_.begin(), lru_.end(), key);
  if (it != lru_.end()) {
    lru_.erase(it);
  }
  lru_.push_front(key);
}

std::size_t LruCache::CurrentBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return current_bytes_;
}

void LruCache::EvictIfNeeded() {
  while (current_bytes_ > max_bytes_ && !lru_.empty()) {
    auto key = lru_.back();
    lru_.pop_back();
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      current_bytes_ -= it->second.data.size() * sizeof(float);
      entries_.erase(it);
    }
  }
}

DoubleBuffer::DoubleBuffer(std::size_t buffer_bytes) {
  Resize(buffer_bytes);
}

void DoubleBuffer::Resize(std::size_t buffer_bytes) {
  auto element_count = buffer_bytes / sizeof(float);
  buffers_[0].assign(element_count, 0.0f);
  buffers_[1].assign(element_count, 0.0f);
}

std::vector<float> &DoubleBuffer::WriteBuffer() { return buffers_[write_index_]; }

const std::vector<float> &DoubleBuffer::ReadBuffer() const {
  return buffers_[1 - write_index_];
}

void DoubleBuffer::Swap() { write_index_ = 1 - write_index_; }

IOThread::IOThread(FetchCallback fetch_cb) : fetch_cb_(std::move(fetch_cb)) {
  worker_ = std::thread(&IOThread::WorkerLoop, this);
}

IOThread::~IOThread() { Stop(); }

void IOThread::Enqueue(const PrefetchRequest &request) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(request);
  }
  cv_.notify_one();
}

std::optional<CacheEntry> IOThread::PopReady() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (ready_.empty()) {
    return std::nullopt;
  }
  CacheEntry entry = std::move(ready_.front());
  ready_.pop_front();
  return entry;
}

void IOThread::Stop() {
  bool expected = false;
  if (!stop_.compare_exchange_strong(expected, true)) {
    return;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

std::size_t IOThread::Pending() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

void IOThread::WorkerLoop() {
  while (!stop_.load()) {
    PrefetchRequest request;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stop_.load() || !queue_.empty(); });
      if (stop_.load()) {
        break;
      }
      request = queue_.front();
      queue_.pop_front();
    }
    auto data = fetch_cb_(request);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ready_.push_back(CacheEntry{request.key, std::move(data)});
    }
  }
}

OutcoreEngine::OutcoreEngine(std::size_t cache_bytes)
    : cache_(cache_bytes),
      double_buffer_(cache_bytes / 2),
      io_thread_([](const PrefetchRequest &request) {
        std::size_t elements = 1;
        for (auto dim : request.descriptor.tile_shape) {
          elements *= dim;
        }
        return std::vector<float>(elements, 0.0f);
      }) {}

void OutcoreEngine::RegisterBlock(const std::string &key, BlockMetadata metadata) {
  metadata_.Register(key, std::move(metadata));
}

void OutcoreEngine::QueuePrefetch(const std::string &key) {
  auto meta = metadata_.Lookup(key);
  if (!meta || meta->is_zero) {
    return;
  }
  io_thread_.Enqueue(PrefetchRequest{key, meta->descriptor});
}

bool OutcoreEngine::TryConsume() {
  if (auto ready = io_thread_.PopReady()) {
    cache_.Put(ready->key, std::move(ready->data));
    double_buffer_.Swap();
    return true;
  }
  return false;
}

std::size_t OutcoreEngine::CacheBytes() const { return cache_.CurrentBytes(); }

BlockDescriptor OutcoreEngine::AlignChunkToTile(const std::vector<std::size_t> &tile_shape,
                                               const std::vector<std::size_t> &chunk_alignment,
                                               std::size_t element_bytes) {
  if (tile_shape.size() != chunk_alignment.size()) {
    throw std::invalid_argument("tile shape and alignment rank must match");
  }
  BlockDescriptor descriptor;
  descriptor.tile_shape = tile_shape;
  descriptor.chunk_shape.resize(tile_shape.size());
  std::size_t elements = 1;
  for (std::size_t i = 0; i < tile_shape.size(); ++i) {
    std::size_t tile = tile_shape[i];
    std::size_t align = chunk_alignment[i] ? chunk_alignment[i] : 1;
    std::size_t aligned = ((tile + align - 1) / align) * align;
    descriptor.chunk_shape[i] = aligned;
    elements *= tile;
  }
  descriptor.bytes = elements * element_bytes;
  return descriptor;
}

}  // namespace outcore
