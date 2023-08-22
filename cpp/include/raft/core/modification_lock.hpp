/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <set>
#include <thread>
#include <type_traits>

namespace raft {

template <typename T, typename L>
struct locked_proxy {
  locked_proxy(T* wrapped, L&& lock) : wrapped_{wrapped}, lock_{std::move(lock)} {}
  auto* operator->() { return wrapped_; }

 private:
  T* wrapped_;
  L lock_;
};

/* A mutex which yields to threads in the order in which they attempt to
 * acquire a lock.
 */
struct ordered_mutex {
  void lock()
  {
    auto scoped_lock = std::unique_lock<std::mutex>{raw_mtx_};
    auto ticket      = next_ticket_++;
    queue_control_.wait(scoped_lock, [ticket, this]() { return ticket == current_ticket_; });
  }

  void unlock()
  {
    auto scoped_lock = std::unique_lock<std::mutex>{raw_mtx_};
    ++current_ticket_;
    queue_control_.notify_all();
  }

 private:
  std::condition_variable queue_control_{};
  std::mutex raw_mtx_{};
  std::size_t next_ticket_{};
  std::size_t current_ticket_{};
};

/* A scoped lock based on ordered_mutex, which will be acquired in the order in which
 * threads attempt to acquire the underlying mutex */
struct ordered_lock {
  explicit ordered_lock(ordered_mutex& mtx)
    : mtx_{[&mtx]() {
        mtx.lock();
        return &mtx;
      }()}
  {
  }

  ~ordered_lock() { mtx_->unlock(); }

 private:
  ordered_mutex* mtx_;
};

/* This struct wraps an object which may be modified from some host threads
 * but used without modification from others. Because multiple users can safely
 * access the object simultaneously so long as it is not being modified, any
 * const access to a threadsafe_wrapper<T> will acquire a lock solely to
 * increment an atomic counter indicating that it is currently accessing the
 * underlying object. It will then decrement that counter once the const call
 * to the underlying object has been completed. Non-const access will
 * acquire a lock on the same underlying mutex but not proceed with the
 * non-const call until the counter reaches 0.
 *
 * A special lock (ordered_lock) ensures that the mutex is acquired in the
 * order that threads attempt to acquire it. This ensures that
 * modifying threads are not indefinitely delayed.
 *
 * Example usage:
 *
 * struct foo() {
 *   foo(int data) : data_{data} {}
 *   auto get_data() const { return data_; }
 *   void set_data(int new_data) { data_ = new_data; }
 *  private:
 *   int data_;
 * };
 *
 * auto f = threadsafe_wrapper<foo>{5};
 * f->set_data(6);
 * f->get_data();  // Safe but inefficient. Returns 6.
 * std::as_const(f)->get_data();  // Safe and efficient. Returns 6.
 * std::as_const(f)->set_data(7);  // Fails to compile.
 */
template <typename T>
struct threadsafe_wrapper {
  template <typename... Args>
  threadsafe_wrapper(Args&&... args) : wrapped{std::make_unique<T>(std::forward<Args>(args)...)}
  {
  }
  auto operator->() { return locked_proxy<T*, modifier_lock>{wrapped.get(), modifier_lock{mtx_}}; }
  auto operator->() const
  {
    return locked_proxy<T const*, user_lock>{wrapped.get(), user_lock{mtx_}};
  }

 private:
  // A class for coordinating access to a resource that may be modified by some
  // threads and used without modification by others.
  class modification_mutex {
    void acquire_for_modifier()
    {
      // Prevent any new users from incrementing work counter
      lock_ = std::make_unique<ordered_lock>(mtx_);
      // Wait until all work in progress is done
      while (currently_using_.load() != 0)
        ;
      std::atomic_thread_fence(std::memory_order_acquire);
    }
    void release_from_modifier() { lock_.reset(); }
    void acquire_for_user() const
    {
      auto tmp_lock = ordered_lock{mtx_};
      ++currently_using_;
    }
    void release_from_user() const
    {
      std::atomic_thread_fence(std::memory_order_release);
      --currently_using_;
    }
    mutable ordered_mutex mtx_{};
    mutable std::atomic<int> currently_using_{};
    mutable std::unique_ptr<ordered_lock> lock_{nullptr};
    friend struct modifier_lock;
    friend struct user_lock;
  };

  // A lock acquired to modify the wrapped object.
  struct modifier_lock {
    modifier_lock(modification_mutex& mtx)
      : mtx_{[&mtx]() {
          mtx.acquire_for_modifier();
          return &mtx;
        }()}
    {
    }
    ~modifier_lock() { mtx_->release_from_modifier(); }

   private:
    modification_mutex* mtx_;
  };

  // A lock acquired to use but not modify the wrapped object. We ensure that
  // only const methods can be accessed while protected by this lock.
  struct user_lock {
    user_lock(modification_mutex const& mtx)
      : mtx_{[&mtx]() {
          mtx.acquire_for_user();
          return &mtx;
        }()}
    {
    }
    ~user_lock() { mtx_->release_from_user(); }

   private:
    modification_mutex const* mtx_;
  };
  modification_mutex mtx_;
  std::unique_ptr<T> wrapped;
};

/* A wrapper used to ensure that an object is not being used while it is being
 * modified on another host thread or device stream
 *
 * Example usage:
 *
 * struct foo() {
 *   foo(raft::resources const& res) : data_{res, 3, 2} {}
 *   auto get_row(raft::resources const& res, int row_idx) const {
 *     auto result = raft::make_host_vector<int>{res, data_.extent(1)};
 *     raft::copy(
 *       result.data_handle(),
 *       data_.data_handle() + row_idx * data_.extent(1),
 *       data_.extent(1),
 *       raft::resource::get_cuda_stream(res)
 *     );
 *     return result;
 *   }
 *   void set_row(raft::resources const& res, int row_idx, raft::host_vector<int> row) {
 *     raft::copy(
 *       data_.data_handle() + row_idx * data_.extent(1),
 *       row.data_handle(),
 *       data_.extent(1),
 *       raft::resource::get_cuda_stream(res)
 *     );
 *   }
 *  private:
 *   raft::device_matrix<int> data_;
 * };
 *
 * auto stream0 = rmm::cuda_stream{};
 * auto stream1 = rmm::cuda_stream{};
 * auto stream2 = rmm::cuda_stream{};
 *
 * auto res0 = raft::device_resources{stream0.view()};
 * auto res1 = raft::device_resources{stream1.view()};
 * auto res2 = raft::device_resources{stream2.view()};
 *
 * auto f_safe = streamsafe_wrapper<foo>{res0};
 *
 * auto data0 = raft::host_vector<int>{res0, 2};
 * data0(0) = 1;
 * data0(1) = 2;
 * auto data1 = raft::host_vector<int>{res1, 2};
 * data1(0) = 3;
 * data1(1) = 4;
 * auto data2 = raft::host_vector<int>{res2, 2};
 * data2(0) = 5;
 * data2(1) = 6;
 * TODO(wphicks)
 */
template <typename T>
struct streamsafe_wrapper {
  using wrapped_type = T;
  template <typename... Args>
  explicit streamsafe_wrapper(resources const& res, Args&&... args)
    : mtx_{}, wrapped_{[this, &res, args = std::make_tuple(std::forward<Args>(args)...)]() {
        auto lock = modifier_lock{mtx_, res};
        return std::apply(
          [res](auto&&... args) { return std::make_unique<T>(res, std::forward<Args>(args)...); },
          std::move(args));
      }()}
  {
  }

  template <typename... LambdaTs>
  auto apply(raft::resources const& res, LambdaTs&&... lambdas)
  {
    return apply_<modifier_lock>(res, std::move(lambdas)...);
  }

  template <typename... LambdaTs>
  auto apply(raft::resources const& res, LambdaTs&&... lambdas) const
  {
    return apply_<user_lock>(res, std::move(lambdas)...);
  }

  // Synchronize all device-side work that has occurred on the underlying
  // object, including both modification and use
  auto synchronize() { mtx_->synchronize(); }
  // Synchronize on any stream owned by res and remove those streams from
  // the sets requiring synchronization
  auto synchronize(resources const& res) { mtx_->synchronize(res); }
  // Synchronize on the given stream and remove that stream from
  // the sets requiring synchronization
  auto synchronize(resources const& res, rmm::cuda_stream_view stream)
  {
    mtx_->synchronize(res, stream);
  }

  // Synchronize on any stream owned by res if and only if it is among those
  // requiring synchronization and then remove it from the corresponding set.
  auto synchronize_if_required(resources const& res) { mtx_->synchronize_if_required(res); }
  // Synchronize on the given stream if and only if it is among those
  // requiring synchronization and then remove it from the corresponding set.
  auto synchronize_if_required(resources const& res, rmm::cuda_stream_view stream)
  {
    mtx_->synchronize_if_required(res, stream);
  }

 private:
  // A class for coordinating access to a resource that may be modified by some
  // threads/streams and used without modification by others.
  struct modification_mutex {
    void synchronize()
    {
      // Grab a lock to prevent new users from adding work or new
      // modifiers from modifying
      auto tmp_lock = ordered_lock{mtx_};
      // Synchronize all streams that might be using or modifying the locked
      // object
      synchronize_modifiers();
      synchronize_users();
    }
    void synchronize(resources const& res)
    {
      auto tmp_lock = ordered_lock{mtx_};
      auto stream   = resource::get_cuda_stream(res).value();
      resource::sync_stream(res);
      user_streams_.erase(stream);
      modifier_streams_.erase(stream);
      if (resource::is_stream_pool_initialized(res)) {
        resource::sync_stream_pool(res);
        for (auto stream_idx = std::size_t{}; stream_idx < resource::get_stream_pool_size(res);
             ++stream_idx) {
          stream = resource::get_stream_from_stream_pool(res, stream_idx).value();
          user_streams_.erase(stream);
          modifier_streams_.erase(stream);
        }
      }
    }

    void synchronize(resources const& res, rmm::cuda_stream_view stream)
    {
      auto tmp_lock = ordered_lock{mtx_};
      resource::sync_stream(res, stream);
      user_streams_.erase(stream.value());
      modifier_streams_.erase(stream.value());
    };

    // Synchronize the indicated stream only if it was used to modify or access
    // the locked object and has not yet been synchronized
    void synchronize_if_required(resources const& res, rmm::cuda_stream_view stream)
    {
      auto tmp_lock = ordered_lock{mtx_};
      _synchronize_if_required(res, stream);
    }

    void synchronize_if_required(resources const& res)
    {
      auto tmp_lock = ordered_lock{mtx_};
      _synchronize_if_required(res, resource::get_cuda_stream(res));
      if (resource::is_stream_pool_initialized(res)) {
        for (auto stream_idx = std::size_t{}; stream_idx < resource::get_stream_pool_size(res);
             ++stream_idx) {
          _synchronize_if_required(res, resource::get_stream_from_stream_pool(res, stream_idx));
        }
      }
    }

   private:
    void _synchronize_if_required(resources const& res, rmm::cuda_stream_view stream)
    {
      auto users_iter = user_streams_.find(stream.value());
      if (users_iter != std::end(user_streams_)) {
        user_streams_.erase(users_iter);
        resource::sync_stream(res, stream);
      }
    }
    static void synchronize(std::set<cudaStream_t>& stream_set)
    {
      while (stream_set.size() != std::size_t{}) {
        for (auto stream : stream_set) {
          auto status = cudaStreamQuery(stream);
          if (status != cudaErrorNotReady) {
            // Whether this stream has succeeded or errored out, it is no
            // longer a stream we need to keep track of
            if (stream_set.erase(stream) != std::size_t{}) {
              if (status != cudaErrorInvalidResourceHandle) { RAFT_CUDA_TRY(status); }
              break;  // Do not continue to iterate on modified set
            }
          }
        }
      }
    }

    // Synchronize on any stream that might have modified the locked object
    void synchronize_modifiers() { synchronize(modifier_streams_); }

    // Synchronize on any stream that accessed this object but did not modify
    // it
    void synchronize_users() { synchronize(user_streams_); }

    void acquire_for_modifier(resources const& mod_res, bool allow_modifier_overlap = false)
    {
      // Prevent any new users from incrementing work counter
      lock_ = std::make_unique<ordered_lock>(mtx_);
      // Wait until all work in progress is done
      while (currently_using_.load() != 0 && user_streams_.size() != 0) {
        if (currently_using_.load() == 0) {
          synchronize_users();
        } else {
          // Yield to user threads so they have a chance to decrement
          // currently_using_, since that occurs outside of a lock
          std::this_thread::yield();
        }
      };
      if (!allow_modifier_overlap) { synchronize_modifiers(); }
      // Keep track of any streams which might be modifying the locked object
      modifier_streams_.insert(resource::get_cuda_stream(mod_res).value());
      if (resource::is_stream_pool_initialized(mod_res)) {
        for (auto stream_idx = std::size_t{}; stream_idx < resource::get_stream_pool_size(mod_res);
             ++stream_idx) {
          modifier_streams_.insert(
            resource::get_stream_from_stream_pool(mod_res, stream_idx).value());
        }
      }
      std::atomic_thread_fence(std::memory_order_acquire);
    }

    void release_from_modifier() { lock_.reset(); }

    void acquire_for_user(resources const& user_res) const
    {
      auto tmp_lock = ordered_lock{mtx_};
      // Ensure that all modifying threads have completed their work on device
      synchronize_modifiers();
      // Any stream on this resource is a potential user of this object. Track
      // them all in order to synchronize before modification.
      user_streams_.insert(resource::get_cuda_stream(user_res).value());
      if (resource::is_stream_pool_initialized(user_res)) {
        for (auto stream_idx = std::size_t{}; stream_idx < resource::get_stream_pool_size(user_res);
             ++stream_idx) {
          user_streams_.insert(resource::get_stream_from_stream_pool(user_res, stream_idx).value());
        }
      }
      ++currently_using_;
    }
    void release_from_user() const
    {
      std::atomic_thread_fence(std::memory_order_release);
      --currently_using_;
    }
    mutable ordered_mutex mtx_{};
    mutable std::atomic<int> currently_using_{};
    mutable std::unique_ptr<ordered_lock> lock_{nullptr};
    // The streams which have been used or may have been used to access but
    // not modify the locked object.
    // Q: Why not simply store the raft::resources objects themselves?
    // A: Because those objects may go out of scope before we need to
    // query them in order to guarantee that user streams have completed work before modification
    // streams. The streams may have been destroyed, but we can at least query
    // them and ignore the cudaErrorInvalidResourceHandle that gets
    // returned.
    mutable std::set<cudaStream_t> user_streams_{};
    mutable std::set<cudaStream_t> modifier_streams_{};
    friend struct modifier_lock;
    friend struct user_lock;
  };

  // A lock acquired to modify the wrapped object.
  struct modifier_lock {
    modifier_lock(modification_mutex& mtx,
                  resources const& res,
                  bool allow_modifier_overlap = false)
      : mtx_{[&mtx, &res, allow_modifier_overlap]() {
          mtx.acquire_for_modifier(res, allow_modifier_overlap);
          return &mtx;
        }()}
    {
    }
    ~modifier_lock() { mtx_->release_from_modifier(); }

   private:
    modification_mutex* mtx_;
  };

  // A lock acquired to use but not modify the wrapped object. We ensure that
  // only const methods can be accessed while protected by this lock.
  struct user_lock {
    user_lock(modification_mutex const& mtx, resources const& res)
      : mtx_{[&mtx, &res]() {
          mtx.acquire_for_user(res);
          return &mtx;
        }()}
    {
    }
    ~user_lock() { mtx_->release_from_user(); }

   private:
    modification_mutex const* mtx_;
  };

  template <typename LockT, typename... LambdaTs>
  auto apply_(raft::resources const& res, LambdaTs&&... lambdas)
  {
    auto lock = LockT{mtx_, res, true};
    if constexpr (sizeof...(lambdas) == 1) {
      return std::get<0>(std::make_tuple(std::move(lambdas)...))(res, get_wrapped());
    } else {
      return std::make_tuple([&res, this](auto&& f) {
        auto res_from_pool = resources(res);
        resource::set_cuda_stream(res_from_pool, resource::get_next_usable_stream(res));
        return f(res_from_pool, get_wrapped());
      }(lambdas)...);
    }
  }
  auto& get_wrapped() { return *wrapped_; }
  auto const& get_wrapped() const { return *wrapped_; }
  modification_mutex mtx_;
  std::unique_ptr<T> wrapped_;
};

}  // namespace raft
