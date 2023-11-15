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
#include <optional>
#include <raft/core/logger.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/resources.hpp>
#include <type_traits>
#ifdef RAFT_DISABLE_CUDA
#include <raft/core/mdbuffer.hpp>
#else
#include <raft/core/mdbuffer.cuh>
#endif

namespace raft {
/**
 * \defgroup copyback_buffer `raft::copyback_buffer`
 * @{
 */

/**
 * @brief A wrapper for an mdspan which provides a temporary "view" of the
 * mdspan's data in a possibly different format (e.g. on device instead of on host or in
 * a different memory layout). The new data configuration can be accessed via
 * the `view()` method.` On construction, the data will be copied to a
 * temporary location if and only if necessary. When the `sync()` method is called, the data will be
 * copied back to the original mdspan if and only if necessary.
 *
 * If a `copyback_buffer` is destroyed without `sync()` having been called
 * during its lifetime, a warning will be emitted.
 *
 * @tparam DesiredMemType the desired memory type (host/device/managed...) for
 * the temporary data
 * @tparam SrcMdspanType the original mdspan
 * @tparam ElementType the desired element type for the temporary data. Note that
 * this must be compatible with the original mdspan's element type
 * @tparam Extents the desired extents type for the temporary data.
 * @tparam LayoutPolicy the desired layout for the temporary data
 */
template <raft::memory_type DesiredMemType,
          typename SrcMdspanType,
          typename ElementType                             = typename SrcMdspanType::element_type,
          typename Extents                                 = typename SrcMdspanType::extents_type,
          typename LayoutPolicy                            = typename SrcMdspanType::layout_type,
          std::enable_if_t<is_output_mdspan_v<SrcMdspan>>* = nullptr>
struct copyback_buffer {
  auto static constexpr mem_type = DesiredMemType;
  using src_data_type            = SrcMdspanType;

  copyback_buffer(raft::resources const& res, SrcMdspanType src_data)
    : src_data_{src_data},
      tmp_data_{res, mdbuffer{src_data_}, std::make_optional<raft::memory_type>(mem_type)}
  {
  }

  /**
   * Returns an mdspan of the data in the desired format and location
   *
   * If a copy was required on construction, the returned mdspan will be a
   * view of the temporary data. Otherwise, the returned mdspan will be a view
   * of the original data.
   */
  auto view() { return tmp_data_.view<mem_type>(); }

  /**
   * Returns an mdspan of the data in the desired format and location
   *
   * If a copy was required on construction, the returned mdspan will be a
   * view of the temporary data. Otherwise, the returned mdspan will be a view
   * of the original data.
   */
  auto view() const { return tmp_data_.view<mem_type>(); }

  /**
   * Copies data back to the original location if necessary
   */
  void sync(raft::resources const& res)
  {
    if (tmp_data_.is_owning()) { raft::copy(res, src_data_, view()); }
    synced_ = true;
  }

  ~copyback_buffer()
  {
    if (!synced_) { RAFT_LOG_WARN("copyback_buffer was not synced before destruction"); }
  }

 private:
  src_data_type src_data_;
  mdbuffer<ElementType, Extents, LayoutPolicy> tmp_data_;
  bool synced_ = false;
};

/**
 * @brief A wrapper for an mdspan which provides a temporary "view" of the
 * mdspan's data in a possibly different format (e.g. on device instead of on host or in
 * a different memory layout). The new data configuration can be accessed via
 * the `view()` method.` On construction, the data will be copied to a
 * temporary location if and only if necessary. When the `auto_copyback_buffer`
 * is destroyed, the data will automatically be copied back to the original
 * location if and only if necessary using the raft::resources object provided
 * at construction.
 *
 * Note that `auto_copyback_buffer` _can_ throw an exception in its destructor,
 * so take care to use it only in a context where this is acceptable.
 * Furthermore, it is the responsibility of the caller to manage the lifetime
 * of the raft::resources object provided at construction and to synchronize
 * that object after the `auto_copyback_buffer` is destroyed. For contexts
 * where either or both of these caveats are undesirable, consider using
 * copyback_buffer instead. Unlike `auto_copyback_buffer`, `copyback_buffer`
 * requires an explicit call to `sync()` to copy the data back to the
 * original location (if necessary) instead of doing so automatically in the
 * destructor, but it offers a non-throwing destructor and makes resource
 * handling more explicit.
 *
 * @tparam DesiredMemType the desired memory type (host/device/managed...) for
 * the temporary data
 * @tparam SrcMdspanType the original mdspan
 * @tparam ElementType the desired element type for the temporary data. Note that
 * this must be compatible with the original mdspan's element type
 * @tparam Extents the desired extents type for the temporary data.
 * @tparam LayoutPolicy the desired layout for the temporary data
 */
template <raft::memory_type DesiredMemType,
          typename SrcMdspanType,
          typename ElementType                             = typename SrcMdspanType::element_type,
          typename Extents                                 = typename SrcMdspanType::extents_type,
          typename LayoutPolicy                            = typename SrcMdspanType::layout_type,
          std::enable_if_t<is_output_mdspan_v<SrcMdspan>>* = nullptr>
struct auto_copyback_buffer {
  auto static constexpr mem_type = DesiredMemType;
  using src_data_type            = SrcMdspanType;

  auto_copyback_buffer(raft::resources const& res, SrcMdspanType src_data)
    : res_{res},
      src_data_{src_data},
      tmp_data_{res, mdbuffer{src_data_}, std::make_optional<raft::memory_type>(mem_type)}
  {
  }

  /**
   * Returns an mdspan of the data in the desired format and location
   *
   * If a copy was required on construction, the returned mdspan will be a
   * view of the temporary data. Otherwise, the returned mdspan will be a view
   * of the original data.
   */
  auto view() { return tmp_data_.view<mem_type>(); }

  /**
   * Returns an mdspan of the data in the desired format and location
   *
   * If a copy was required on construction, the returned mdspan will be a
   * view of the temporary data. Otherwise, the returned mdspan will be a view
   * of the original data.
   */
  auto view() const { return tmp_data_.view<mem_type>(); }

  ~auto_copyback_buffer() noexcept(false)
  {
    if (tmp_data_.is_owning()) { raft::copy(res_, src_data_, view()); }
  }

 private:
  raft::resources const& res_;
  src_data_type src_data_;
  mdbuffer<ElementType, Extents, LayoutPolicy> tmp_data_;
};

/**@}*/

}  // namespace raft
