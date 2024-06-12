/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"

#include <raft/core/copyback_buffer.cuh>

#include <gtest/gtest.h>

namespace raft {

template <raft::memory_type from_mem_type, raft::memory_type tmp_mem_type>
void test_copyback_buffer()
{
  auto res            = device_resources{};
  auto constexpr rows = std::uint32_t{3};
  auto constexpr cols = std::uint32_t{2};
  auto data           = [rows, cols]() {
    // constexpr based on from_mem_type
    return make_host_mdarray<int, std::uint32_t, layout_c_contiguous, rows, cols>(
      res, extents<std::uint32_t, rows, cols>{});
  }();

  auto gen_unique_entry  = [](auto&& x, auto&& y) { return x * 7 + y * 11; };
  auto alter_entry       = [](auto&& x) { return x + 13; };
  auto gen_altered_entry = [](auto&& x, auto&& y, auto&& z) {
    return alter_entry(gen_unique_entry(x, y, z));
  };
  res.sync_stream();

  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      data(i, j) = gen_unique_entry(i, j);
    }
  }
  res.sync_stream();
  auto new_data      = copyback_buffer<tmp_mem_type>(res, data.view());
  auto new_data_view = new_data.view();
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      new_data_view(i, j) = alter_entry(new_data_view(i, j));
    }
  }
  new_data.sync();
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_EQ(new_data_view(i, j), gen_altered_entry(i, j));
    }
  }
}

TEST(CopybackBuffer, FromHost)
{
  auto res            = device_resources{};
  auto constexpr rows = std::uint32_t{3};
  auto constexpr cols = std::uint32_t{2};
  auto data           = make_host_mdarray<int, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});

  auto gen_unique_entry  = [](auto&& x, auto&& y) { return x * 7 + y * 11; };
  auto alter_entry       = [](auto&& x) { return x + 13; };
  auto gen_altered_entry = [](auto&& x, auto&& y, auto&& z) {
    return alter_entry(gen_unique_entry(x, y, z));
  };
  res.sync_stream();

  // Temporary on host
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      data(i, j) = gen_unique_entry(i, j);
    }
  }
  res.sync_stream();
  auto new_data      = copyback_buffer<raft::memory_type::host>(res, data.view());
  auto new_data_view = new_data.view();
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      new_data_view(i, j) = alter_entry(new_data_view(i, j));
    }
  }
  new_data.sync();
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_EQ(new_data_view(i, j), gen_altered_entry(i, j));
    }
  }

  // Temporary on device
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      data(i, j) = gen_unique_entry(i, j);
    }
  }
  res.sync_stream();
  auto new_data      = copyback_buffer<raft::memory_type::device>(res, data.view());
  auto new_data_view = new_data.view();
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      new_data_view(i, j) = alter_entry(new_data_view(i, j));
    }
  }
  new_data.sync();
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_EQ(new_data_view(i, j), gen_altered_entry(i, j));
    }
  }
}

TEST(AutoCopybackBuffer, FromHost) {}

}  // namespace raft
