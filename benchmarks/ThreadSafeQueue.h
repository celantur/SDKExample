#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>

template <typename T>
class ThreadSafeQueue
{
 public:
  explicit ThreadSafeQueue(size_t max_size = 0) : max_size_(max_size) {}

  T pop()
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty())
    {
      cond_.wait(mlock);
    }
    auto item = queue_.front();
    queue_.pop();
    if (max_size_ > 0)
      cond_not_full_.notify_one();
    return item;
  }

  T pop_timeout(std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    if (cond_.wait_for(mlock, timeout, [this] { return !queue_.empty(); }))
    {
      auto item = queue_.front();
      queue_.pop();
      if (max_size_ > 0)
        cond_not_full_.notify_one();
      return item;
    }
    throw std::runtime_error("Timeout while waiting for item");
  }

  void pop(T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty())
    {
      cond_.wait(mlock);
    }
    item = queue_.front();
    queue_.pop();
    if (max_size_ > 0)
      cond_not_full_.notify_one();
  }

  void push(const T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    if (max_size_ > 0)
      cond_not_full_.wait(mlock, [this] { return queue_.size() < max_size_; });
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }

  void push(T&& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    if (max_size_ > 0)
      cond_not_full_.wait(mlock, [this] { return queue_.size() < max_size_; });
    queue_.push(std::move(item));
    mlock.unlock();
    cond_.notify_one();
  }

 private:
  size_t max_size_;
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::condition_variable cond_not_full_;
};
