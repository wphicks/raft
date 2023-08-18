#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>


template<typename T, typename L>
struct locked_proxy {
  locked_proxy(T* wrapped, L&& lock) : wrapped_{wrapped}, lock_{std::move(lock)} {}
  auto* operator->() { return wrapped_; }
 private:
  T* wrapped_;
  L lock_;
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
template<typename T>
struct threadsafe_wrapper {
  template<typename... Args>
  threadsafe_wrapper(Args&&... args): wrapped{std::make_unique<T>(std::forward<Args>(args)...)} {}
  auto operator->() {
    return locked_proxy<T*, modifier_lock>{wrapped.get(), modifier_lock{mtx_}};
  }
  auto operator->() const {
    return locked_proxy<T const*, user_lock>{wrapped.get(), user_lock{mtx_}};
  }
 private:
  // A class for coordinating access to a resource that may be modified by some
  // threads and used without modification by others.
  class modification_mutex {
    void acquire_for_modifier() {
      // Prevent any new users from incrementing work counter
      // TODO(wphicks): Use ordered locks for this
      lock_ = std::make_unique<std::unique_lock<std::mutex>>(mtx_);
      // Wait until all work in progress is done
      while(currently_using_.load() != 0);
      std::atomic_thread_fence(std::memory_order_acquire);
    }
    void release_from_modifier() {
      lock_.reset();
    }
    void acquire_for_user() const {
      auto tmp_lock = std::unique_lock<std::mutex>{mtx_};
      ++currently_using_;
    }
    void release_from_user() const {
      std::atomic_thread_fence(std::memory_order_release);
      --currently_using_;
    }
    mutable std::mutex mtx_{};
    mutable std::atomic<int> currently_using_{};
    mutable std::unique_ptr<std::unique_lock<std::mutex>> lock_{nullptr};
    friend struct modifier_lock;
    friend struct user_lock;
  };

  // A lock acquired to modify the wrapped object.
  struct modifier_lock {
    modifier_lock(modification_mutex& mtx) : mtx_{
      [&mtx]() {
        mtx.acquire_for_modifier();
        return &mtx;
      }()
    } {}
    ~modifier_lock() {
      mtx_->release_from_modifier();
    }
   private:
    modification_mutex* mtx_;
  };

  // A lock acquired to use but not modify the wrapped object. We ensure that
  // only const methods can be accessed while protected by this lock.
  struct user_lock {
    user_lock(modification_mutex const& mtx) : mtx_{
      [&mtx]() {
        mtx.acquire_for_user();
        return &mtx;
      }()
    } {}
    ~user_lock() {
      mtx_->release_from_user();
    }
   private:
    modification_mutex const* mtx_;
  };
  modification_mutex mtx_;
  std::unique_ptr<T> wrapped;
};

#define RAFT_STREAMSAFE_CALL(wrapper, methodname, ...) wrapper.call(typename decltype(wrapper)::wrapped_type::methodname, __VA_ARGS__)

#define RAFT_CONST_STREAMSAFE_CALL(wrapper, methodname, ...) std::as_const(wrapper).call(typename decltype(wrapper)::wrapped_type::methodname, __VA_ARGS__)

/* Example usage:
 *
 * struct foo() {
 *   foo(int data) : data_{data} {}
 *   auto get_data() const { return data_; }
 *   void set_data(int new_data) { data_ = new_data; }
 *  private:
 *   int data_;
 * };
 *
 * auto f = streamsafe_wrapper<foo>{5};
 * auto& res = raft::device_resources_manager::get_device_resources();
 * f.call(foo::wrapped_type::set_data, res, 6);
 * RAFT_STREAMSAFE_CALL(f, set_data, res, 6);
 * f.call(foo::wrapped_type::get_data, res);
 * RAFT_STREAMSAFE_CALL(f, get_data);
 * std::as_const(f).call(foo::wrapped_type::get_data, res);
 * RAFT_CONST_STREAMSAFE_CALL(f, get_data);
 */
template<typename T>
struct streamsafe_wrapper {
  using wrapped_type = T;
  template<typename... Args>
  streamsafe_wrapper(Args&&... args): wrapped_{std::make_unique<T>(std::forward<Args>(args)...)} {}

  template <typename Ret, typename... Args>
  auto call(Ret (T::*func)(Args...), raft::device_resources const& res, Args... args) const {
    auto lock = user_lock{mtx_, res};
    return wrapped_->*func(res, std::forward<Args>(args)...);
  }
  template <typename Ret, typename... Args>
  auto call(Ret (T::*func)(Args...), raft::device_resources const& res, Args... args) {
    auto lock = modifier_lock{mtx_, res};
    return wrapped_->*func(res, std::forward<Args>(args)...);
  }
 private:
  // A class for coordinating access to a resource that may be modified by some
  // threads and used without modification by others.
  class modification_mutex {
    void acquire_for_modifier(raft::device_resources& mod_res) {
      // Prevent any new users from incrementing work counter
      // TODO(wphicks): Use ordered locks for this
      lock_ = std::make_unique<std::unique_lock<std::mutex>>(mtx_);
      modifier_resources_ = &mod_res;
      // Wait until all work in progress is done
      while(currently_using_.load() != 0);
      std::atomic_thread_fence(std::memory_order_acquire);
      // Ensure that all user threads have completed their work on device
      for (auto res_ : user_resources_) {
        res_->sync_stream();
        if (res_->is_stream_pool_initialized()) {
          res_->sync_stream_pool();
        }
      }
      user_resources_.clear();
    }
    void release_from_modifier() {
      lock_.reset();
    }
    void acquire_for_user(raft::device_resources const& user_res) const {
      auto tmp_lock = std::unique_lock<std::mutex>{mtx_};
      // Ensure that all modifying threads have completed their work on device
      if (modifier_resources_ != nullptr) {
        modifier_resources_->sync_stream();
        if (modifier_resources_->is_stream_pool_initialized()) {
          modifier_resources_->sync_stream_pool();
        }
        modifier_resources_ = nullptr;
      }
      if (std::find(std::begin(user_resources_), std::end(user_resources_), &user_res) == std::end(user_resources_)) {
        user_resources_.push_back(&user_res);
      }
      ++currently_using_;
    }
    void release_from_user() const {
      std::atomic_thread_fence(std::memory_order_release);
      --currently_using_;
    }
    mutable std::mutex mtx_{};
    mutable std::atomic<int> currently_using_{};
    mutable std::unique_ptr<std::unique_lock<std::mutex>> lock_{nullptr};
    // TODO(wphicks): Just store the streams and sync on them
    mutable std::vector<raft::device_resources*> user_resources_{};
    mutable raft::device_resources const* modifier_resources_{nullptr};
    std::unique_ptr<T> wrapped_;
    friend struct modifier_lock;
    friend struct user_lock;
  };

  // A lock acquired to modify the wrapped object.
  struct modifier_lock {
    modifier_lock(modification_mutex& mtx, raft::device_resources const& res) : mtx_{
      [&mtx, &res]() {
        mtx.acquire_for_modifier(res);
        return &mtx;
      }()
    } {}
    ~modifier_lock() {
      mtx_->release_from_modifier();
    }
   private:
    modification_mutex* mtx_;
  };

  // A lock acquired to use but not modify the wrapped object. We ensure that
  // only const methods can be accessed while protected by this lock.
  struct user_lock {
    user_lock(modification_mutex const& mtx, raft::device_resources const& res) : mtx_{
      [&mtx, &res]() {
        mtx.acquire_for_user(res);
        return &mtx;
      }()
    } {}
    ~user_lock() {
      mtx_->release_from_user();
    }
   private:
    modification_mutex const* mtx_;
  };
  modification_mutex mtx_;
};
