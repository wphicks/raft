/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nccl.h>

#include <comms.hpp>

#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
#include "ucp_helper.hpp"

#include <nccl.h>

constexpr bool UCX_ENABLED = true;

#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <raft/handle.hpp>
#include <cstdio>
#include <exception>
#include <memory>

#include <thread>

#include <cuda_runtime.h>

#include <raft/cudart_utils.h>


#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t status = call;                                                \
    ASSERT(ncclSuccess == status, "ERROR: NCCL call='%s'. Reason:%s\n", #call, \
           ncclGetErrorString(status));                                        \
  } while (0)

#define NCCL_CHECK_NO_THROW(call)                                 \
  do {                                                            \
    ncclResult_t status = call;                                   \
    if (status != ncclSuccess) {                                  \
      CUML_LOG_ERROR("NCCL call='%s' failed. Reason:%s\n", #call, \
                     ncclGetErrorString(status));                 \
    }                                                             \
  } while (0)



namespace raft {

namespace {

size_t getDatatypeSize(const comms::datatype_t datatype) {
  switch (datatype) {
    case comms::CHAR:
      return sizeof(char);
    case comms::UINT8:
      return sizeof(uint8_t);
    case comms::INT:
      return sizeof(int);
    case comms::UINT:
      return sizeof(unsigned int);
    case comms::INT64:
      return sizeof(int64_t);
    case comms::UINT64:
      return sizeof(uint64_t);
    case comms::FLOAT:
      return sizeof(float);
    case comms::DOUBLE:
      return sizeof(double);
  }
}

ncclDataType_t getNCCLDatatype(
  const comms::datatype_t datatype) {
  switch (datatype) {
    case comms::CHAR:
      return ncclChar;
    case comms::UINT8:
      return ncclUint8;
    case comms::INT:
      return ncclInt;
    case comms::UINT:
      return ncclUint32;
    case comms::INT64:
      return ncclInt64;
    case comms::UINT64:
      return ncclUint64;
    case comms::FLOAT:
      return ncclFloat;
    case comms::DOUBLE:
      return ncclDouble;
  }
}

ncclRedOp_t getNCCLOp(const comms::op_t op) {
  switch (op) {
    case comms::SUM:
      return ncclSum;
    case comms::PROD:
      return ncclProd;
    case comms::MIN:
      return ncclMin;
    case comms::MAX:
      return ncclMax;
  }
}
}  // namespace

bool ucx_enabled() { return UCX_ENABLED; }

/**
 * @brief Underlying comms, like NCCL and UCX, should be initialized and ready for use,
 * and maintained, outside of the cuML Comms lifecycle. This allows us to decouple the
 * ownership of the actual comms from cuml so that they can also be used outside of cuml.
 *
 * For instance, nccl-py can be used to bootstrap a ncclComm_t before it is
 * used to construct a cuml comms instance. UCX endpoints can be bootstrapped
 * in Python using ucx-py, before being used to construct a cuML comms instance.
 */
void inject_comms(cumlHandle &handle, ncclComm_t comm, ucp_worker_h ucp_worker,
                  std::shared_ptr<ucp_ep_h *> eps, int size, int rank) {
  auto communicator = std::make_shared<comms>(
    std::unique_ptr<MLCommon::comms>(
      new std_comms(comm, ucp_worker, eps, size, rank)));
  handle.getImpl().setCommunicator(communicator);
}

void inject_comms(cumlHandle &handle, ncclComm_t comm, int size, int rank) {
  auto communicator = std::make_shared<comms>(
    std::unique_ptr<MLCommon::comms>(
      new std_comms(comm, size, rank)));
  handle.getImpl().setCommunicator(communicator);
}

void inject_comms_py_coll(cumlHandle *handle, ncclComm_t comm, int size,
                          int rank) {
  inject_comms(*handle, comm, size, rank);
}

void inject_comms_py(ML::cumlHandle *handle, ncclComm_t comm, void *ucp_worker,
                     void *eps, int size, int rank) {
  std::shared_ptr<ucp_ep_h *> eps_sp =
    std::make_shared<ucp_ep_h *>(new ucp_ep_h[size]);

  size_t *size_t_ep_arr = (size_t *)eps;

  for (int i = 0; i < size; i++) {
    size_t ptr = size_t_ep_arr[i];
    ucp_ep_h *ucp_ep_v = (ucp_ep_h *)*eps_sp;

    if (ptr != 0) {
      ucp_ep_h eps_ptr = (ucp_ep_h)size_t_ep_arr[i];
      ucp_ep_v[i] = eps_ptr;
    } else {
      ucp_ep_v[i] = nullptr;
    }
  }

  inject_comms(*handle, comm, (ucp_worker_h)ucp_worker, eps_sp, size, rank);
}


/**
 * @brief A comms implementation capable of running collective communications
 * with NCCL and point-to-point-communications with UCX. Note that the latter is optional.
 *
 * Underlying comms, like NCCL and UCX, should be initialized and ready for use,
 * and maintained, outside of the cuML Comms lifecycle. This allows us to decouple the
 * ownership of the actual comms from cuml so that they can also be used outside of cuml.
 *
 * For instance, nccl-py can be used to bootstrap a ncclComm_t before it is
 * used to construct a cuml comms instance. UCX endpoints can be bootstrapped
 * in Python using ucx-py, before being used to construct a cuML comms instance.
 */
class std_comms : public raft::comms {
 public:
  std_comms() = delete;

  /**
   * @brief Constructor for collective + point-to-point operation.
   * @param comm initialized nccl comm
   * @param ucp_worker initialized ucp_worker instance
   * @param eps shared pointer to array of ucp endpoints
   * @param size size of the cluster
   * @param rank rank of the current worker
   */
  std_comms(ncclComm_t comm, ucp_worker_h ucp_worker,
                           std::shared_ptr<ucp_ep_h*> eps, int size, int rank)  : _nccl_comm(comm),
                        		    _ucp_worker(ucp_worker),
                        		    _ucp_eps(eps),
                        		    _size(size),
                        		    _rank(rank),
                        		    _next_request_id(0) {
                        		  initialize();
                        		  p2p_enabled = true;
                        		};


  /**
   * @brief constructor for collective-only operation
   * @param comm initilized nccl communicator
   * @param size size of the cluster
   * @param rank rank of the current worker
   */
  std_comms(ncclComm_t comm, int size, int rank)
  : _nccl_comm(comm), _size(size), _rank(rank) {
  initialize();
  };

  virtual ~std_comms(){
	  CUDA_CHECK_NO_THROW(cudaStreamDestroy(_stream));

	  CUDA_CHECK_NO_THROW(cudaFree(_sendbuff));
	  CUDA_CHECK_NO_THROW(cudaFree(_recvbuff));
	}

  size_t getDatatypeSize(const comms::datatype_t datatype) {
    switch (datatype) {
      case comms::CHAR:
        return sizeof(char);
      case comms::UINT8:
        return sizeof(uint8_t);
      case comms::INT:
        return sizeof(int);
      case comms::UINT:
        return sizeof(unsigned int);
      case comms::INT64:
        return sizeof(int64_t);
      case comms::UINT64:
        return sizeof(uint64_t);
      case comms::FLOAT:
        return sizeof(float);
      case comms::DOUBLE:
        return sizeof(double);
    }
  }


  template <>
  comms::datatype_t getDataType<char>() const {
    return comms::CHAR;
  }

  template <>
  comms::datatype_t getDataType<uint8_t>() const {
    return comms::UINT8;
  }

  template <>
  comms::datatype_t getDataType<int>() const {
    return comms::INT;
  }

  template <>
  comms::datatype_t getDataType<uint32_t>() const {
    return comms::UINT;
  }

  template <>
  comms::datatype_t getDataType<int64_t>() const {
    return comms::INT64;
  }

  template <>
  comms::datatype_t getDataType<uint64_t>() const {
    return comms::UINT64;
  }

  template <>
  comms::datatype_t getDataType<float>() const {
    return comms::FLOAT;
  }

  template <>
  comms::datatype_t getDataType<double>() const {
    return comms::DOUBLE;
  }

  void initialize() {
    CUDA_CHECK(cudaStreamCreate(&_stream));

    CUDA_CHECK(cudaMalloc(&_sendbuff, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&_recvbuff, sizeof(int)));
  }


  int getSize() const { return _size; }

  int getRank() const { return _rank; }

  std::unique_ptr<comms>
  commSplit(int color, int key) const {
    // Not supported by NCCL
    ASSERT(false,
           "ERROR: commSplit called but not yet supported in this comms "
           "implementation.");
  }

  void barrier() const {
    CUDA_CHECK(cudaMemsetAsync(_sendbuff, 1, sizeof(int), _stream));
    CUDA_CHECK(cudaMemsetAsync(_recvbuff, 1, sizeof(int), _stream));

    allreduce(_sendbuff, _recvbuff, 1, comms::INT,
              comms::SUM, _stream);

    ASSERT(syncStream(_stream) == status_t::commStatusSuccess,
           "ERROR: syncStream failed. This can be caused by a failed rank.");
  }

  void get_request_id(request_t *req) const {

    request_t req_id;

    if (this->_free_requests.empty())
      req_id = this->_next_request_id++;
    else {
      auto it = this->_free_requests.begin();
      req_id = *it;
      this->_free_requests.erase(it);
    }
    *req = req_id;
  }

  void isend(const void *buf, int size, int dest,
                                       int tag, request_t *request) const {
    ASSERT(UCX_ENABLED, "cuML Comms not built with UCX support");
    ASSERT(p2p_enabled,
           "cuML Comms instance was not initialized for point-to-point");

    ASSERT(_ucp_worker != nullptr,
           "ERROR: UCX comms not initialized on communicator.");

    get_request_id(request);
    ucp_ep_h ep_ptr = (*_ucp_eps)[dest];

    ucp_request *ucp_req = (ucp_request *)malloc(sizeof(ucp_request));

    this->_ucp_handler.ucp_isend(ucp_req, ep_ptr, buf, size, tag,
                                 default_tag_mask, getRank());

    CUML_LOG_DEBUG(
      "%d: Created send request [id=%llu], ptr=%llu, to=%llu, ep=%llu", getRank(),
      (unsigned long long)*request, (unsigned long long)ucp_req->req,
      (unsigned long long)dest, (unsigned long long)ep_ptr);

    _requests_in_flight.insert(std::make_pair(*request, ucp_req));
  }

  void irecv(void *buf, int size, int source, int tag,
                                       request_t *request) const {
    ASSERT(UCX_ENABLED, "cuML Comms not built with UCX support");
    ASSERT(p2p_enabled,
           "cuML Comms instance was not initialized for point-to-point");

    ASSERT(_ucp_worker != nullptr,
           "ERROR: UCX comms not initialized on communicator.");

    get_request_id(request);

    ucp_ep_h ep_ptr = (*_ucp_eps)[source];

    ucp_tag_t tag_mask = default_tag_mask;

    ucp_request *ucp_req = (ucp_request *)malloc(sizeof(ucp_request));
    _ucp_handler.ucp_irecv(ucp_req, _ucp_worker, ep_ptr, buf, size, tag, tag_mask,
                           source);

    CUML_LOG_DEBUG(
      "%d: Created receive request [id=%llu], ptr=%llu, from=%llu, ep=%llu",
      getRank(), (unsigned long long)*request, (unsigned long long)ucp_req->req,
      (unsigned long long)source, (unsigned long long)ep_ptr);

    _requests_in_flight.insert(std::make_pair(*request, ucp_req));
  }

  void waitall(int count,
                                         request_t array_of_requests[]) const {
    ASSERT(UCX_ENABLED, "cuML Comms not built with UCX support");
    ASSERT(p2p_enabled,
           "cuML Comms instance was not initialized for point-to-point");

    ASSERT(_ucp_worker != nullptr,
           "ERROR: UCX comms not initialized on communicator.");

    std::vector<ucp_request *> requests;
    requests.reserve(count);

    time_t start = time(NULL);

    for (int i = 0; i < count; ++i) {
      auto req_it = _requests_in_flight.find(array_of_requests[i]);
      ASSERT(_requests_in_flight.end() != req_it,
             "ERROR: waitall on invalid request: %d", array_of_requests[i]);
      requests.push_back(req_it->second);
      _free_requests.insert(req_it->first);
      _requests_in_flight.erase(req_it);
    }

    while (requests.size() > 0) {
      time_t now = time(NULL);

      // Timeout if we have not gotten progress or completed any requests
      // in 10 or more seconds.
      ASSERT(now - start < 10, "Timed out waiting for requests.");

      for (std::vector<ucp_request *>::iterator it = requests.begin();
           it != requests.end();) {
        bool restart = false;  // resets the timeout when any progress was made

        // Causes UCP to progress through the send/recv message queue
        while (_ucp_handler.ucp_progress(_ucp_worker) != 0) {
          restart = true;
        }

        auto req = *it;

        // If the message needs release, we know it will be sent/received
        // asynchronously, so we will need to track and verify its state
        if (req->needs_release) {
          ASSERT(UCS_PTR_IS_PTR(req->req),
                 "UCX Request Error. Request is not valid UCX pointer");
          ASSERT(!UCS_PTR_IS_ERR(req->req), "UCX Request Error: %d\n",
                 UCS_PTR_STATUS(req->req));
          ASSERT(req->req->completed == 1 || req->req->completed == 0,
                 "request->completed not a valid value: %d\n",
                 req->req->completed);
        }

        // If a message was sent synchronously (eg. completed before
        // `isend`/`irecv` completed) or an asynchronous message
        // is complete, we can go ahead and clean it up.
        if (!req->needs_release || req->req->completed == 1) {
          restart = true;
          CUML_LOG_DEBUG(
            "%d: request completed. [ptr=%llu, num_left=%lu,"
            " other_rank=%d, is_send=%d, completed_immediately=%d]",
            getRank(), (unsigned long long)req->req, requests.size() - 1,
            req->other_rank, req->is_send_request, !req->needs_release);

          // perform cleanup
          _ucp_handler.free_ucp_request(req);

          // remove from pending requests
          it = requests.erase(it);
        } else {
          ++it;
        }
        // if any progress was made, reset the timeout start time
        if (restart) {
          start = time(NULL);
        }
      }
    }
  }

  void allreduce(const void *sendbuff, void *recvbuff,
                                           int count, datatype_t datatype,
                                           op_t op, cudaStream_t stream) const {
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, getNCCLDatatype(datatype),
                             getNCCLOp(op), _nccl_comm, stream));
  }

  void bcast(void *buff, int count, datatype_t datatype,
                                       int root, cudaStream_t stream) const {
    NCCL_CHECK(ncclBroadcast(buff, buff, count, getNCCLDatatype(datatype), root,
                             _nccl_comm, stream));
  }

  void reduce(const void *sendbuff, void *recvbuff,
                                        int count, datatype_t datatype, op_t op,
                                        int root, cudaStream_t stream) const {
    NCCL_CHECK(ncclReduce(sendbuff, recvbuff, count, getNCCLDatatype(datatype),
                          getNCCLOp(op), root, _nccl_comm, stream));
  }

  void allgather(const void *sendbuff, void *recvbuff,
                                           int sendcount, datatype_t datatype,
                                           cudaStream_t stream) const {
    NCCL_CHECK(ncclAllGather(sendbuff, recvbuff, sendcount,
                             getNCCLDatatype(datatype), _nccl_comm, stream));
  }

  void allgatherv(const void *sendbuf, void *recvbuf,
                                            const int recvcounts[],
                                            const int displs[],
                                            datatype_t datatype,
                                            cudaStream_t stream) const {
    //From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" - https://arxiv.org/pdf/1812.05964.pdf
    //Listing 1 on page 4.
    for (int root = 0; root < _size; ++root)
      NCCL_CHECK(ncclBroadcast(
        sendbuf,
        static_cast<char *>(recvbuf) + displs[root] * getDatatypeSize(datatype),
        recvcounts[root], getNCCLDatatype(datatype), root, _nccl_comm, stream));
  }

  void reducescatter(const void *sendbuff,
                                               void *recvbuff, int recvcount,
                                               datatype_t datatype, op_t op,
                                               cudaStream_t stream) const {
    NCCL_CHECK(ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                 getNCCLDatatype(datatype), getNCCLOp(op),
                                 _nccl_comm, stream));
  }

  status_t std_comms::syncStream(
    cudaStream_t stream) const {
    cudaError_t cudaErr;
    ncclResult_t ncclErr, ncclAsyncErr;
    while (1) {
      cudaErr = cudaStreamQuery(stream);
      if (cudaErr == cudaSuccess) return status_t::commStatusSuccess;

      if (cudaErr != cudaErrorNotReady) {
        // An error occurred querying the status of the stream
        return status_t::commStatusError;
      }

      ncclErr = ncclCommGetAsyncError(_nccl_comm, &ncclAsyncErr);
      if (ncclErr != ncclSuccess) {
        // An error occurred retrieving the asynchronous error
        return status_t::commStatusError;
      }

      if (ncclAsyncErr != ncclSuccess) {
        // An asynchronous error happened. Stop the operation and destroy
        // the communicator
        ncclErr = ncclCommAbort(_nccl_comm);
        if (ncclErr != ncclSuccess)
          // Caller may abort with an exception or try to re-create a new communicator.
          return status_t::commStatusAbort;
      }

      // Let other threads (including NCCL threads) use the CPU.
      pthread_yield();
    }
  }

 private:
  ncclComm_t _nccl_comm;
  cudaStream_t _stream;

  int *_sendbuff, *_recvbuff;

  int _size;
  int _rank;

  bool p2p_enabled = false;
  comms_ucp_handler _ucp_handler;
  ucp_worker_h _ucp_worker;
  std::shared_ptr<ucp_ep_h*> _ucp_eps;
  mutable request_t _next_request_id;
  mutable std::unordered_map<request_t, struct ucp_request*>
    _requests_in_flight;
  mutable std::unordered_set<request_t> _free_requests;
};

}  // end namespace ML
