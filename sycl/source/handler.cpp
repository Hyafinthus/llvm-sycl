//==-------- handler.cpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/handler_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/sycl_mem_obj_i.hpp>
#include <detail/sycl_mem_obj_t.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/usm/usm_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/pi.h>
#include <sycl/detail/pi.hpp>
#include <sycl/access/access.hpp>
#include <sycl/context.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/event.hpp>
#include <sycl/handler.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/stream.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <mqueue.h>
#include <unistd.h>

#include <detail/daemon/daemon.hpp>
#include <detail/daemon/define.hpp>
#include <detail/program_manager/program_manager.hpp>

using sycl::detail::SyclKernelCg;
// #define PRINT_TRACE 1

extern mqd_t mq_id_daemon, mq_id_program;

struct OfflineProfileEvent {
  int KernelCount = 0;
  int DeviceIndex = 0;
  int NumParts = 1;
  uint64_t HostStartNs = 0;
  uint64_t HostEndNs = 0;
  sycl::event Event;
  // Capture the key before ProgramManager::kernel_reqs is cleared.  This also
  // makes profile collection independent of the deferred command-group
  // lifetime and is required by a future asynchronous completion path.
  std::string KernelKey;
  uint64_t MaterializationNs = 0;
};

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

handler::handler(std::shared_ptr<detail::queue_impl> Queue, bool IsHost)
    : handler(Queue, Queue, nullptr, IsHost) {}

handler::handler(std::shared_ptr<detail::queue_impl> Queue,
                 std::shared_ptr<detail::queue_impl> PrimaryQueue,
                 std::shared_ptr<detail::queue_impl> SecondaryQueue,
                 bool IsHost)
    : MImpl(std::make_shared<detail::handler_impl>(std::move(PrimaryQueue),
                                                   std::move(SecondaryQueue))),
      MQueue(std::move(Queue)), MIsHost(IsHost) {}

#if defined(SCHEDULE) || defined(SCHEDULE_OFFLINE) || defined(SNMD_OFFLINE)
static constexpr const char *OfflineMqMultipartTag =
    "SYCL_OFFLINE_MQ_MULTIPART_V1";

static void sendOfflineMqPayload(mqd_t Queue, const std::string &Payload,
                                 size_t MaxMessageSize,
                                 const char *Description) {
  if (Payload.size() <= MaxMessageSize) {
    if (mq_send(Queue, Payload.c_str(), Payload.size(), 0) == -1) {
      std::string ErrorMsg =
          std::string("Error: ") + Description + " mq_send failed";
      perror(ErrorMsg.c_str());
      exit(1);
    }
    return;
  }

  const size_t ChunkPayloadSize = MaxMessageSize;
  const size_t ChunkCount =
      (Payload.size() + ChunkPayloadSize - 1) / ChunkPayloadSize;
  std::ostringstream Header;
  Header << OfflineMqMultipartTag << "\n" << Payload.size() << "\n"
         << ChunkCount << "\n";
  std::string HeaderPayload = Header.str();
  if (HeaderPayload.size() > MaxMessageSize) {
    std::cerr << "Error: " << Description
              << " multipart header exceeds mq message size" << std::endl;
    exit(1);
  }

  if (mq_send(Queue, HeaderPayload.c_str(), HeaderPayload.size(), 0) == -1) {
    std::string ErrorMsg =
        std::string("Error: ") + Description + " multipart header mq_send failed";
    perror(ErrorMsg.c_str());
    exit(1);
  }

  for (size_t Offset = 0; Offset < Payload.size();
       Offset += ChunkPayloadSize) {
    const size_t ChunkSize =
        std::min(ChunkPayloadSize, Payload.size() - Offset);
    if (mq_send(Queue, Payload.data() + Offset, ChunkSize, 0) == -1) {
      std::string ErrorMsg =
          std::string("Error: ") + Description + " multipart chunk mq_send failed";
      perror(ErrorMsg.c_str());
      exit(1);
    }
  }

  HANDLER_TRACE_STREAM << "=== handler === " << Description
                       << " multipart mq_send payload_size: "
                       << Payload.size() << " chunks: " << ChunkCount
                       << std::endl;
}

static bool parseOfflineMqMultipartHeader(const std::string &Message,
                                          size_t &PayloadSize,
                                          size_t &ChunkCount) {
  std::istringstream Stream(Message);
  std::string Tag;
  std::getline(Stream, Tag);
  if (Tag != OfflineMqMultipartTag) {
    return false;
  }
  Stream >> PayloadSize;
  Stream >> ChunkCount;
  return Stream.good() || Stream.eof();
}

static std::string receiveOfflineMqPayload(mqd_t Queue, size_t MaxMessageSize,
                                           const char *Description) {
  std::vector<char> Buffer(MaxMessageSize);
  ssize_t BytesReceived =
      mq_receive(Queue, Buffer.data(), MaxMessageSize, nullptr);
  if (BytesReceived <= 0) {
    std::string ErrorMsg =
        std::string("Error: ") + Description + " mq_receive failed";
    perror(ErrorMsg.c_str());
    exit(1);
  }

  std::string Message(Buffer.data(), static_cast<size_t>(BytesReceived));
  size_t PayloadSize = 0;
  size_t ChunkCount = 0;
  if (!parseOfflineMqMultipartHeader(Message, PayloadSize, ChunkCount)) {
    return Message;
  }

  std::string Payload;
  Payload.reserve(PayloadSize);
  for (size_t I = 0; I < ChunkCount; ++I) {
    BytesReceived = mq_receive(Queue, Buffer.data(), MaxMessageSize, nullptr);
    if (BytesReceived <= 0) {
      std::string ErrorMsg = std::string("Error: ") + Description +
                             " multipart chunk mq_receive failed";
      perror(ErrorMsg.c_str());
      exit(1);
    }
    Payload.append(Buffer.data(), static_cast<size_t>(BytesReceived));
  }

  if (Payload.size() != PayloadSize) {
    std::cerr << "Error: " << Description
              << " multipart payload size mismatch, expected " << PayloadSize
              << " got " << Payload.size() << std::endl;
    exit(1);
  }

  HANDLER_TRACE_STREAM << "=== handler === " << Description
                       << " multipart mq_receive payload_size: "
                       << Payload.size() << " chunks: " << ChunkCount
                       << std::endl;
  return Payload;
}

static std::vector<D2SKernelExecInfo>
parseOfflineKernelExecInfos(const std::string &ReceivedData) {
  std::vector<D2SKernelExecInfo> KernelExecInfos;
  std::istringstream Stream(ReceivedData);
  std::string Line;

  while (std::getline(Stream, Line)) {
    if (Line.empty()) {
      continue;
    }

    std::string ObjData = Line + "\n"; // kernel_count
    std::getline(Stream, Line);
    ObjData += Line + "\n"; // exec
    std::getline(Stream, Line);
    ObjData += Line + "\n"; // device_index
    std::getline(Stream, Line);
    ObjData += Line + "\n"; // num_parts
    std::getline(Stream, Line);
    ObjData += Line + "\n"; // split_devices.size()
    int SplitDeviceCount = std::stoi(Line);
    for (int I = 0; I < SplitDeviceCount; ++I) {
      std::getline(Stream, Line);
      ObjData += Line + "\n"; // split device index
    }
    std::getline(Stream, Line);
    ObjData += Line + "\n"; // scale_count
    std::getline(Stream, Line);
    ObjData += Line + "\n"; // req_counts.size()
    int ReqCount = std::stoi(Line);
    for (int I = 0; I < ReqCount; ++I) {
      std::getline(Stream, Line);
      ObjData += Line + "\n";
    }

    KernelExecInfos.push_back(D2SKernelExecInfo::deserialize(ObjData));
  }

  return KernelExecInfos;
}

struct ParsedOfflineDispatchBatch {
  bool CompletionDriven = false;
  bool WindowComplete = false;
  bool WindowFailed = false;
  std::vector<D2SKernelExecInfo> KernelExecInfos;
};

static ParsedOfflineDispatchBatch
parseOfflineDispatchBatch(const std::string &ReceivedData) {
  ParsedOfflineDispatchBatch Parsed;
  if (D2SDispatchBatchData::isDispatchBatch(ReceivedData)) {
    D2SDispatchBatchData Batch =
        D2SDispatchBatchData::deserialize(ReceivedData);
    Parsed.CompletionDriven = Batch.completion_driven;
    Parsed.WindowComplete = Batch.window_complete;
    Parsed.WindowFailed = Batch.window_failed;
    Parsed.KernelExecInfos = std::move(Batch.kernel_exec_infos);
    return Parsed;
  }
  Parsed.KernelExecInfos = parseOfflineKernelExecInfos(ReceivedData);
  return Parsed;
}

static property_list
getOfflineProfilingPropertyList(const detail::QueueImplPtr &Queue) {
  (void)Queue;
  return property_list(property::queue::in_order{},
                       property::queue::enable_profiling{});
}

// A platform default context can contain more than one device.  That is fine
// for the normal SYCL scheduler, but the offline split implementation uses a
// MemObjRecord's current context as the identity of the device that owns the
// complete, merged value.  Reusing one default context for two CUDA devices
// makes that state ambiguous and can select an allocation created for the
// other device during a split merge.  Keep one stable private context per
// offline device so context identity remains a valid ownership key.
static detail::ContextImplPtr
getOfflineDeviceContext(const detail::DeviceImplPtr &Device) {
  static std::mutex ContextMutex;
  static std::vector<
      std::pair<detail::DeviceImplPtr, detail::ContextImplPtr>>
      DeviceContexts;

  std::lock_guard<std::mutex> Lock(ContextMutex);
  for (const auto &Entry : DeviceContexts) {
    if (Entry.first == Device) {
      return Entry.second;
    }
  }

  detail::ContextImplPtr NewContext = detail::getSyclObjImpl(
      context{detail::createSyclObjFromImpl<device>(Device), {}, {}});
  DeviceContexts.push_back({Device, NewContext});
  HANDLER_TRACE_STREAM
      << "=== handler === Offline private device context created: device "
      << Device.get() << " context " << NewContext << std::endl;
  return NewContext;
}

static detail::QueueImplPtr
makeOfflineProfilingQueue(const detail::DeviceImplPtr &Device,
                          const detail::QueueImplPtr &OldQueue) {
  // The daemon models every device as one HEFT processor timeline. Creating a
  // new out-of-order queue for every rebind violates that model: kernels
  // assigned sequentially to one GPU are launched on unrelated CUDA streams
  // and can overlap pending split kernels and their transfers. Keep one
  // in-order profiling queue per offline device. Different devices still run
  // concurrently, including the different devices used by one split kernel.
  static std::mutex QueueMutex;
  static std::vector<std::pair<detail::DeviceImplPtr, detail::QueueImplPtr>>
      DeviceQueues;

  std::lock_guard<std::mutex> Lock(QueueMutex);
  for (const auto &Entry : DeviceQueues) {
    if (Entry.first == Device) {
      HANDLER_TRACE_STREAM
          << "=== handler === Offline profiling queue reused, profiling: "
          << Entry.second
                 ->has_property<property::queue::enable_profiling>()
          << " in_order: "
          << Entry.second->has_property<property::queue::in_order>()
          << std::endl;
      return Entry.second;
    }
  }

  try {
    property_list ProfilingProps = getOfflineProfilingPropertyList(OldQueue);
    detail::QueueImplPtr NewQueue = std::make_shared<detail::queue_impl>(
        Device, getOfflineDeviceContext(Device),
        OldQueue->getAsyncHandler(), ProfilingProps);
    HANDLER_TRACE_STREAM << "=== handler === Offline profiling queue created, profiling: "
              << NewQueue->has_property<property::queue::enable_profiling>()
              << " in_order: "
              << NewQueue->has_property<property::queue::in_order>()
              << std::endl;
    DeviceQueues.push_back({Device, NewQueue});
    return NewQueue;
  } catch (const std::exception &e) {
    HANDLER_TRACE_STREAM << "=== handler === Offline profiling queue creation failed: "
              << e.what() << ". Fallback to original queue properties."
              << std::endl;
  } catch (...) {
    HANDLER_TRACE_STREAM << "=== handler === Offline profiling queue creation failed. "
              << "Fallback to original queue properties." << std::endl;
  }

  // Profiling may be unsupported by a backend, but ordering is part of the
  // offline scheduler contract and must not be dropped in the fallback.
  detail::QueueImplPtr NewQueue = std::make_shared<detail::queue_impl>(
      Device, getOfflineDeviceContext(Device), OldQueue->getAsyncHandler(),
      property_list(property::queue::in_order{}));
  DeviceQueues.push_back({Device, NewQueue});
  return NewQueue;
}

static detail::SyclKernelCg *
findOfflineKernelCg(std::vector<detail::SyclKernelCg *> &KernelCgs,
                    int KernelCount) {
  auto It = std::find_if(KernelCgs.begin(), KernelCgs.end(),
                         [KernelCount](detail::SyclKernelCg *KernelCg) {
                           return KernelCg &&
                                  KernelCg->kernel_count == KernelCount;
                         });
  if (It == KernelCgs.end()) {
    throw sycl::runtime_error(
        "Internal Error. Offline kernel_count not found in this wait batch.",
        PI_ERROR_INVALID_OPERATION);
  }
  return *It;
}

static int clampOfflineDeviceIndex(int RequestedDeviceIndex) {
  const auto &Devices = detail::ProgramManager::getInstance().globalDevices;
  if (Devices.empty()) {
    throw sycl::runtime_error(
        "Internal Error. Offline scheduler has no available SYCL devices.",
        PI_ERROR_INVALID_OPERATION);
  }

  if (RequestedDeviceIndex >= 0 &&
      RequestedDeviceIndex < static_cast<int>(Devices.size())) {
    return RequestedDeviceIndex;
  }

  const int ClampedDeviceIndex =
      std::min(std::max(RequestedDeviceIndex, 0),
               static_cast<int>(Devices.size()) - 1);
  HANDLER_TRACE_STREAM << "=== handler === Offline device_index out of range: requested "
            << RequestedDeviceIndex << " available " << Devices.size()
            << ", use " << ClampedDeviceIndex << std::endl;
  return ClampedDeviceIndex;
}

#ifdef SNMD_OFFLINE
static std::vector<int>
normalizeOfflineSplitDevices(const D2SKernelExecInfo &KernelExecInfo,
                             int ActualDeviceIndex) {
  const auto &Devices = detail::ProgramManager::getInstance().globalDevices;
  std::vector<int> SplitDevices;
  const int RequestedParts = std::max(1, KernelExecInfo.num_parts);

  auto addDevice = [&](int DeviceIndex) {
    if (DeviceIndex <= 0 ||
        DeviceIndex >= static_cast<int>(Devices.size())) {
      return;
    }
    if (std::find(SplitDevices.begin(), SplitDevices.end(), DeviceIndex) !=
        SplitDevices.end()) {
      return;
    }
    SplitDevices.push_back(DeviceIndex);
  };

  if (RequestedParts > 1) {
    for (int DeviceIndex : KernelExecInfo.split_devices) {
      addDevice(DeviceIndex);
    }
    addDevice(ActualDeviceIndex);
    for (int DeviceIndex = 1;
         DeviceIndex < static_cast<int>(Devices.size()) &&
         static_cast<int>(SplitDevices.size()) < RequestedParts;
         ++DeviceIndex) {
      addDevice(DeviceIndex);
    }
  }

  if (static_cast<int>(SplitDevices.size()) > RequestedParts) {
    SplitDevices.resize(RequestedParts);
  }
  return SplitDevices;
}

static bool offlineSplitWriteAccess(access::mode Mode) {
  return Mode == access::mode::write || Mode == access::mode::read_write ||
         Mode == access::mode::discard_write ||
         Mode == access::mode::discard_read_write ||
         Mode == access::mode::atomic;
}

static bool offlineSplitCanUseDim0ContiguousWrites(
    const detail::CGExecKernel *ExecCG, size_t NumParts) {
  if (ExecCG == nullptr || NumParts <= 1) {
    return false;
  }

  const detail::NDRDescT &NDR = ExecCG->MNDRDesc;
  if (NDR.GlobalSize[0] < NumParts || NDR.GlobalSize[0] % NumParts != 0) {
    return false;
  }

  const bool KernelSplitsOnlyDim0 =
      NDR.GlobalSize[0] > 1 && NDR.GlobalSize[1] <= 1 &&
      NDR.GlobalSize[2] <= 1;

  for (detail::Requirement *Req : ExecCG->MRequirements) {
    if (Req == nullptr) {
      continue;
    }

    const range<3> AccessRange = Req->MAccessRange;
    const range<3> MemoryRange = Req->MMemoryRange;

    if (offlineSplitWriteAccess(Req->MAccessMode)) {
      // The merge implementation owns complete dim-0 row blocks.  Require the
      // write accessor to cover the kernel's full dim-0 range and every trailing
      // dimension so each part is one contiguous row-major byte interval.
      if (Req->MIsSubBuffer || AccessRange[0] != NDR.GlobalSize[0] ||
          AccessRange[0] < NumParts || AccessRange[0] % NumParts != 0 ||
          Req->MOffset[1] != 0 || Req->MOffset[2] != 0 ||
          AccessRange[1] != MemoryRange[1] ||
          AccessRange[2] != MemoryRange[2]) {
        return false;
      }

      if (KernelSplitsOnlyDim0 &&
          (AccessRange[1] > 1 || AccessRange[2] > 1)) {
        HANDLER_TRACE_STREAM
            << "=== handler === Split disabled: dim0-only kernel writes "
            << "non-contiguous access range " << AccessRange[0] << ","
            << AccessRange[1] << "," << AccessRange[2] << std::endl;
        return false;
      }
    }
  }

  return true;
}

static std::unique_ptr<detail::Requirement>
makeOfflineLinearRowBlockReq(const detail::Requirement *Req,
                             size_t RelativeBegin0, size_t Rows) {
  if (Req == nullptr || Req->MIsSubBuffer || Req->MDims <= 1 || Rows == 0 ||
      Req->MSYCLMemObj == nullptr ||
      Req->MSYCLMemObj->getType() !=
          detail::SYCLMemObjI::MemObjType::Buffer) {
    return nullptr;
  }

  const range<3> AccessRange = Req->MAccessRange;
  const range<3> MemoryRange = Req->MMemoryRange;
  if (Req->MOffset[1] != 0 || Req->MOffset[2] != 0 ||
      AccessRange[1] != MemoryRange[1] ||
      AccessRange[2] != MemoryRange[2] ||
      RelativeBegin0 > AccessRange[0] ||
      Rows > AccessRange[0] - RelativeBegin0) {
    return nullptr;
  }

  const size_t Max = std::numeric_limits<size_t>::max();
  if ((MemoryRange[1] != 0 && MemoryRange[2] > Max / MemoryRange[1]) ||
      RelativeBegin0 > Max - Req->MOffset[0]) {
    return nullptr;
  }

  const size_t RowElements = MemoryRange[1] * MemoryRange[2];
  const size_t LinearRow = Req->MOffset[0] + RelativeBegin0;
  if (RowElements == 0 || LinearRow > MemoryRange[0] ||
      Rows > MemoryRange[0] - LinearRow) {
    return nullptr;
  }

  const size_t TotalElements = MemoryRange.size();
  if (LinearRow > TotalElements / RowElements ||
      Rows > (TotalElements - LinearRow * RowElements) / RowElements) {
    return nullptr;
  }
  const size_t LinearOffset = LinearRow * RowElements;
  const size_t CopyElements = Rows * RowElements;
  if (LinearOffset > TotalElements ||
      CopyElements > TotalElements - LinearOffset ||
      (Req->MElemSize != 0 && LinearOffset > Max / Req->MElemSize)) {
    return nullptr;
  }

  auto LinearReq = std::make_unique<detail::Requirement>(*Req);
  LinearReq->MDims = 1;
  LinearReq->MOffset = id<3>(LinearOffset, 0, 0);
  LinearReq->MAccessRange = range<3>(CopyElements, 1, 1);
  LinearReq->MMemoryRange = range<3>(TotalElements, 1, 1);
  LinearReq->MOffsetInBytes = LinearOffset * Req->MElemSize;
  return LinearReq;
}

static void applyOfflineSplitDecision(const D2SKernelExecInfo &KernelExecInfo,
                                      int ActualDeviceIndex) {
  auto &PM = detail::ProgramManager::getInstance();
  PM.NumParts = 1;
  PM.SplitDevices.clear();
  PM.SplitEvents.clear();

  if (KernelExecInfo.num_parts <= 1) {
    return;
  }

  PM.SplitDevices =
      normalizeOfflineSplitDevices(KernelExecInfo, ActualDeviceIndex);
  if (PM.SplitDevices.size() <= 1) {
    PM.SplitDevices.clear();
    return;
  }

  PM.NumParts = PM.SplitDevices.size();
  HANDLER_TRACE_STREAM << "=== handler === Offline split devices:";
  for (int DeviceIndex : PM.SplitDevices) {
    HANDLER_TRACE_STREAM << " " << DeviceIndex;
  }
  HANDLER_TRACE_STREAM << " num_parts: " << PM.NumParts << std::endl;
}

static void finalizeAllPendingOfflineSplits();
static void clearPendingOfflineSplitState();
#ifdef SNMD_OFFLINE_SPLIT_STATS
static void printAndResetOfflineSplitStats();
#endif
#endif

static void clearOfflineBatch() {
  auto &PM = detail::ProgramManager::getInstance();
  for (detail::SyclKernelCg *KernelCg : PM.kernel_cgs) {
    delete KernelCg;
  }
  PM.kernel_cgs.clear();
#ifdef SCHEDULE_OFFLINE
  PM.kernel_reqs.clear();
#endif
#ifdef SNMD_OFFLINE
  finalizeAllPendingOfflineSplits();
#ifdef SNMD_OFFLINE_SPLIT_STATS
  printAndResetOfflineSplitStats();
#endif
  PM.NumParts = 1;
  PM.SplitDevices.clear();
  PM.SplitQueues_Write.clear();
  PM.SplitEvents.clear();
  clearPendingOfflineSplitState();
#endif
}

static std::string findOfflineKernelProfileKey(int KernelCount) {
#ifdef SCHEDULE_OFFLINE
  const auto &KernelReqs = detail::ProgramManager::getInstance().kernel_reqs;
  auto It = std::find_if(KernelReqs.begin(), KernelReqs.end(),
                         [KernelCount](const S2DKernelReqData &KernelReqData) {
                           return KernelReqData.kernel_count == KernelCount;
                         });
  if (It != KernelReqs.end()) {
    return It->profileKey();
  }
#endif
  return "kernel_count=" + std::to_string(KernelCount);
}

static void fillOfflineKernelReqData(
    S2DKernelReqData &KernelReqData, pid_t Pid, int KernelCount,
    const std::string &KernelName, const detail::NDRDescT &NDRDesc,
    const std::vector<detail::Requirement *> &Requirements) {
  KernelReqData.pid = Pid;
  KernelReqData.kernel_count = KernelCount;
  KernelReqData.kernel_identity = stableKernelIdentity(KernelName);
  KernelReqData.req_size = Requirements.size();
  KernelReqData.work_dim = NDRDesc.Dims;
  KernelReqData.global_size0 = NDRDesc.GlobalSize[0];
  KernelReqData.global_size1 = NDRDesc.GlobalSize[1];
  KernelReqData.global_size2 = NDRDesc.GlobalSize[2];
  KernelReqData.reqs.clear();

  for (int I = 0; I < static_cast<int>(Requirements.size()); I++) {
    detail::Requirement *Req = Requirements[I];

    SyclReqData ReqData;
    ReqData.mem_pointer = Req->MSYCLMemObj;
    ReqData.kernel_count = KernelReqData.kernel_count;
    ReqData.req_count = I + 1;
    ReqData.req_accmode = static_cast<acc_mode>(Req->MAccessMode);
    // Daemon HEFT uses elem_size to infer fp32/fp64 precision.
    ReqData.elem_size = static_cast<int>(Req->MElemSize);
    ReqData.buff_size = static_cast<int>(Req->MMemoryRange.size());
    ReqData.range0 = Req->MMemoryRange[0];
    ReqData.range1 = Req->MMemoryRange[1];
    ReqData.range2 = Req->MMemoryRange[2];

    KernelReqData.reqs.push_back(ReqData);
  }
}

static uint64_t offlineNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

#ifdef SNMD_OFFLINE
#ifdef SNMD_OFFLINE_SPLIT_STATS
struct OfflineSplitStats {
  uint64_t SingleKernelCount = 0;
  uint64_t SplitKernelCount = 0;
  uint64_t InputDirectD2DBytes = 0;
  uint64_t InputD2HBytes = 0;
  uint64_t InputH2DBytes = 0;
  uint64_t ReusedReadReplicaBytes = 0;
  uint64_t MergeDirectD2DBytes = 0;
  uint64_t MergeD2HBytes = 0;
  uint64_t MergeH2DBytes = 0;
  uint64_t PrepareWaitNs = 0;
  uint64_t PartWaitNs = 0;
  uint64_t MergeWaitNs = 0;
};

static OfflineSplitStats &offlineSplitStats() {
  static OfflineSplitStats Stats;
  return Stats;
}

static uint64_t offlineRequirementBytes(const detail::Requirement *Req) {
  if (Req == nullptr) {
    return 0;
  }
  const uint64_t Elements = static_cast<uint64_t>(Req->MAccessRange.size());
  const uint64_t ElemSize = static_cast<uint64_t>(Req->MElemSize);
  if (ElemSize != 0 &&
      Elements > std::numeric_limits<uint64_t>::max() / ElemSize) {
    return std::numeric_limits<uint64_t>::max();
  }
  return Elements * ElemSize;
}

static void printAndResetOfflineSplitStats() {
  OfflineSplitStats &Stats = offlineSplitStats();
  if (Stats.SingleKernelCount == 0 && Stats.SplitKernelCount == 0) {
    return;
  }
  std::cout << "SNMD_SPLIT_STATS pid=" << getpid()
            << " single_kernels=" << Stats.SingleKernelCount
            << " split_kernels=" << Stats.SplitKernelCount
            << " input_direct_d2d_bytes=" << Stats.InputDirectD2DBytes
            << " input_d2h_bytes=" << Stats.InputD2HBytes
            << " input_h2d_bytes=" << Stats.InputH2DBytes
            << " reused_read_replica_bytes="
            << Stats.ReusedReadReplicaBytes
            << " merge_direct_d2d_bytes=" << Stats.MergeDirectD2DBytes
            << " merge_d2h_bytes=" << Stats.MergeD2HBytes
            << " merge_h2d_bytes=" << Stats.MergeH2DBytes
            << " prepare_wait_ns=" << Stats.PrepareWaitNs
            << " part_wait_ns=" << Stats.PartWaitNs
            << " merge_wait_ns=" << Stats.MergeWaitNs << std::endl;
  Stats = OfflineSplitStats{};
}
#endif

enum class OfflineSplitWaitKind { Prepare, Part, Merge };

static inline void waitOfflineSplitEvent(const detail::EventImplPtr &Event,
                                         OfflineSplitWaitKind Kind) {
  if (!Event) {
    return;
  }
#ifdef SNMD_OFFLINE_SPLIT_STATS
  const uint64_t StartNs = offlineNowNs();
#else
  (void)Kind;
#endif
  Event->wait(Event);
#ifdef SNMD_OFFLINE_SPLIT_STATS
  const uint64_t DurationNs = offlineNowNs() - StartNs;
  OfflineSplitStats &Stats = offlineSplitStats();
  switch (Kind) {
  case OfflineSplitWaitKind::Prepare:
    Stats.PrepareWaitNs += DurationNs;
    break;
  case OfflineSplitWaitKind::Part:
    Stats.PartWaitNs += DurationNs;
    break;
  case OfflineSplitWaitKind::Merge:
    Stats.MergeWaitNs += DurationNs;
    break;
  }
#endif
}

struct PendingOfflineSplitMerge {
  int KernelCount = 0;
  detail::EventImplPtr Event;
  std::vector<detail::EventImplPtr> Events;
  bool EventsWaited = false;
  uint64_t PartsCompleteNs = 0;
  detail::QueueImplPtr HostQueue;
  detail::ContextImplPtr HostContext;
  std::vector<detail::QueueImplPtr> SplitQueues;
  std::vector<std::vector<detail::Requirement *>> SplitReqsCopy;
  std::vector<std::unique_ptr<detail::Requirement>> SplitReqOwners;
  std::vector<detail::SYCLMemObjI *> ReadMemObjs;
  std::vector<detail::SYCLMemObjI *> WrittenMemObjs;
};

struct OfflineSplitFinalizeTiming {
  int KernelCount = 0;
  uint64_t PartsCompleteNs = 0;
  uint64_t MaterializationNs = 0;
};

struct OfflineReadReplicaCacheEntry {
  detail::SYCLMemObjI *MemObj = nullptr;
  std::vector<detail::QueueImplPtr> Queues;
};

static std::vector<PendingOfflineSplitMerge> &pendingOfflineSplitMerges() {
  static std::vector<PendingOfflineSplitMerge> PendingSplits;
  return PendingSplits;
}

static std::vector<OfflineReadReplicaCacheEntry> &offlineReadReplicaCache() {
  static std::vector<OfflineReadReplicaCacheEntry> Cache;
  return Cache;
}

static int offlineSubmittedNumParts(int KernelCount) {
  for (const PendingOfflineSplitMerge &Pending :
       pendingOfflineSplitMerges()) {
    if (Pending.KernelCount == KernelCount) {
      return std::max<int>(1, static_cast<int>(Pending.SplitQueues.size()));
    }
  }
  return 1;
}

static std::vector<OfflineSplitFinalizeTiming> &offlineSplitFinalizeTimes() {
  static std::vector<OfflineSplitFinalizeTiming> FinalizeTimes;
  return FinalizeTimes;
}

static void clearPendingOfflineSplitState() {
  pendingOfflineSplitMerges().clear();
  offlineSplitFinalizeTimes().clear();
  offlineReadReplicaCache().clear();
}

static void rememberOfflineSplitWrite(PendingOfflineSplitMerge &Pending,
                                      detail::SYCLMemObjI *MemObj) {
  if (MemObj == nullptr) {
    return;
  }
  if (std::find(Pending.WrittenMemObjs.begin(), Pending.WrittenMemObjs.end(),
                MemObj) == Pending.WrittenMemObjs.end()) {
    Pending.WrittenMemObjs.push_back(MemObj);
  }
}

static void rememberOfflineSplitRead(PendingOfflineSplitMerge &Pending,
                                     detail::SYCLMemObjI *MemObj) {
  if (MemObj == nullptr) {
    return;
  }
  if (std::find(Pending.ReadMemObjs.begin(), Pending.ReadMemObjs.end(),
                MemObj) == Pending.ReadMemObjs.end()) {
    Pending.ReadMemObjs.push_back(MemObj);
  }
}

static void cacheOfflineSplitReadReplicas(
    const PendingOfflineSplitMerge &Pending) {
  for (detail::SYCLMemObjI *MemObj : Pending.ReadMemObjs) {
    auto cache_it = std::find_if(
        offlineReadReplicaCache().begin(), offlineReadReplicaCache().end(),
        [MemObj](const OfflineReadReplicaCacheEntry &Entry) {
          return Entry.MemObj == MemObj;
        });
    if (cache_it == offlineReadReplicaCache().end()) {
      offlineReadReplicaCache().push_back({MemObj, Pending.SplitQueues});
      continue;
    }
    for (const detail::QueueImplPtr &Queue : Pending.SplitQueues) {
      if (std::find(cache_it->Queues.begin(), cache_it->Queues.end(), Queue) ==
          cache_it->Queues.end()) {
        cache_it->Queues.push_back(Queue);
      }
    }
  }
}

static void invalidateOfflineReadReplica(detail::SYCLMemObjI *MemObj) {
  std::vector<OfflineReadReplicaCacheEntry> &Cache =
      offlineReadReplicaCache();
  Cache.erase(std::remove_if(Cache.begin(), Cache.end(),
                             [MemObj](const OfflineReadReplicaCacheEntry &Entry) {
                               return Entry.MemObj == MemObj;
                             }),
              Cache.end());
}

static bool pendingOfflineSplitHasReadReplica(
    detail::SYCLMemObjI *MemObj, const detail::QueueImplPtr &TargetQueue) {
  if (MemObj == nullptr || TargetQueue == nullptr) {
    return false;
  }

  for (const OfflineReadReplicaCacheEntry &Entry :
       offlineReadReplicaCache()) {
    if (Entry.MemObj != MemObj) {
      continue;
    }
    for (const detail::QueueImplPtr &ReplicaQueue : Entry.Queues) {
      if (ReplicaQueue != nullptr &&
          detail::sameCtx(ReplicaQueue->getContextImplPtr(),
                          TargetQueue->getContextImplPtr()) &&
          ReplicaQueue->getDeviceImplPtr() ==
              TargetQueue->getDeviceImplPtr()) {
        return true;
      }
    }
  }

  for (const PendingOfflineSplitMerge &Pending :
       pendingOfflineSplitMerges()) {
    if (std::find(Pending.ReadMemObjs.begin(), Pending.ReadMemObjs.end(),
                  MemObj) == Pending.ReadMemObjs.end() ||
        std::find(Pending.WrittenMemObjs.begin(),
                  Pending.WrittenMemObjs.end(), MemObj) !=
            Pending.WrittenMemObjs.end()) {
      continue;
    }

    for (const detail::QueueImplPtr &ReplicaQueue : Pending.SplitQueues) {
      if (ReplicaQueue != nullptr &&
          detail::sameCtx(ReplicaQueue->getContextImplPtr(),
                          TargetQueue->getContextImplPtr()) &&
          ReplicaQueue->getDeviceImplPtr() ==
              TargetQueue->getDeviceImplPtr()) {
        return true;
      }
    }
  }
  return false;
}

static bool kernelTouchesPendingOfflineSplit(
    detail::SyclKernelCg *KernelCg, const PendingOfflineSplitMerge &Pending) {
  if (KernelCg == nullptr || !KernelCg->kernel_cg) {
    return false;
  }

  auto *ExecCG =
      dynamic_cast<detail::CGExecKernel *>(KernelCg->kernel_cg.get());
  if (ExecCG == nullptr) {
    return false;
  }

  for (detail::Requirement *Req : ExecCG->MRequirements) {
    if (Req == nullptr) {
      continue;
    }
    const bool TouchesPendingWrite =
        std::find(Pending.WrittenMemObjs.begin(),
                  Pending.WrittenMemObjs.end(), Req->MSYCLMemObj) !=
        Pending.WrittenMemObjs.end();
    const bool WritesPendingRead =
        offlineSplitWriteAccess(Req->MAccessMode) &&
        std::find(Pending.ReadMemObjs.begin(), Pending.ReadMemObjs.end(),
                  Req->MSYCLMemObj) != Pending.ReadMemObjs.end();
    if (TouchesPendingWrite || WritesPendingRead) {
      return true;
    }
  }
  return false;
}

static void markOfflineSplitPartsComplete(int KernelCount,
                                          uint64_t CompleteNs) {
  for (PendingOfflineSplitMerge &Pending : pendingOfflineSplitMerges()) {
    if (Pending.KernelCount == KernelCount && Pending.PartsCompleteNs == 0) {
      Pending.PartsCompleteNs = CompleteNs;
    }
  }
}

static void waitPendingOfflineSplit(PendingOfflineSplitMerge &Pending) {
  if (Pending.EventsWaited) {
    return;
  }

  if (!Pending.Event && Pending.Events.empty()) {
    Pending.EventsWaited = true;
    return;
  }

  HANDLER_TRACE_STREAM << "=== handler === Split finalize kernel_count: "
            << Pending.KernelCount << " before wait" << std::endl;
  if (!Pending.Events.empty()) {
    for (size_t Part = 0; Part < Pending.Events.size(); ++Part) {
      const detail::EventImplPtr &Event = Pending.Events[Part];
      if (Event) {
        HANDLER_TRACE_STREAM
            << "=== handler === Split finalize kernel_count: "
            << Pending.KernelCount << " part: " << Part << " before wait"
            << std::endl;
        waitOfflineSplitEvent(Event, OfflineSplitWaitKind::Part);
        HANDLER_TRACE_STREAM
            << "=== handler === Split finalize kernel_count: "
            << Pending.KernelCount << " part: " << Part << " after wait"
            << std::endl;
      }
    }
  } else {
    waitOfflineSplitEvent(Pending.Event, OfflineSplitWaitKind::Part);
  }
  HANDLER_TRACE_STREAM << "=== handler === Split finalize kernel_count: "
            << Pending.KernelCount << " after wait" << std::endl;

  Pending.EventsWaited = true;
  if (Pending.PartsCompleteNs == 0) {
    Pending.PartsCompleteNs = offlineNowNs();
  }
}

static void mergePendingOfflineSplit(PendingOfflineSplitMerge &Pending) {
  // Keep this helper safe if it is called independently, while the
  // multi-pending finalizers below deliberately wait every relevant kernel
  // before they enqueue the first merge copy.
  waitPendingOfflineSplit(Pending);
  const uint64_t MaterializationStartNs = offlineNowNs();

  if (Pending.SplitQueues.empty()) {
    offlineSplitFinalizeTimes().push_back(
        {Pending.KernelCount, Pending.PartsCompleteNs, 0});
    return;
  }

  for (size_t p = 0; p < Pending.SplitReqsCopy.size(); ++p) {
    for (detail::Requirement *CopyReq : Pending.SplitReqsCopy[p]) {
      detail::MemObjRecord *Rec =
          detail::Scheduler::getInstance().getMemObjRecord(CopyReq);
      if (Rec == nullptr) {
        continue;
      }
      detail::ContextImplPtr SrcCtx = Rec->MCurContext;
      detail::QueueImplPtr SrcQueue = Pending.HostQueue;
      if (SrcCtx != Pending.HostContext) {
        SrcQueue = nullptr;
        for (detail::AllocaCommandBase *AllocaCmd : Rec->MAllocaCommands) {
          if (AllocaCmd->getQueue() != nullptr &&
              AllocaCmd->getQueue()->getContextImplPtr() == SrcCtx) {
            SrcQueue = AllocaCmd->getQueue();
            break;
          }
        }
      }
      HANDLER_TRACE_STREAM << "=== handler === Split finalize PartReq " << CopyReq
                << " Record: " << Rec << " SrcCtx: " << SrcCtx
                << " SrcQueue: " << SrcQueue << " is host: "
                << (SrcCtx == Pending.HostContext ? "true" : "false")
                << std::endl;

      // normalizeOfflineSplitDevices preserves the daemon order, whose first
      // entry is D2SKernelExecInfo::device_index. Keep this canonical source in
      // sync with estimateCommCostForDevices when the fix is enabled.
      detail::QueueImplPtr MergeQueue = Pending.SplitQueues.front();
#if !defined(SNMD_OFFLINE_CANONICAL_MERGE)
      if (SrcCtx != Pending.HostContext && SrcQueue != nullptr) {
        for (const detail::QueueImplPtr &SplitQueue : Pending.SplitQueues) {
          if (detail::sameCtx(SplitQueue->getContextImplPtr(), SrcCtx)) {
            MergeQueue = SplitQueue;
            break;
          }
        }
      }
#endif
      detail::ContextImplPtr MergeCtx = MergeQueue->getContextImplPtr();

      if (p < Pending.SplitQueues.size() &&
          detail::sameCtx(Pending.SplitQueues[p]->getContextImplPtr(),
                          MergeCtx)) {
        Rec->MCurContext = MergeCtx;
        HANDLER_TRACE_STREAM
            << "=== handler === Split finalize keep partition on merge device"
            << std::endl;
        continue;
      }

      bool moved_by_p2p = false;
      if (p < Pending.SplitQueues.size()) {
        try {
          detail::EventImplPtr ev_p2p =
              detail::Scheduler::getInstance().addMemoryMove(
                  CopyReq, MergeQueue, Pending.SplitQueues[p]);
          waitOfflineSplitEvent(ev_p2p, OfflineSplitWaitKind::Merge);
#ifdef SNMD_OFFLINE_SPLIT_STATS
          offlineSplitStats().MergeDirectD2DBytes +=
              offlineRequirementBytes(CopyReq);
#endif
          moved_by_p2p = true;
          HANDLER_TRACE_STREAM
              << "=== handler === Split finalize merge direct D2D success"
              << std::endl;
        } catch (const std::exception &e) {
          HANDLER_TRACE_STREAM
              << "=== handler === Split finalize merge direct D2D failed, "
              << "fallback D2H->H2D, reason: " << e.what() << std::endl;
        } catch (...) {
          HANDLER_TRACE_STREAM
              << "=== handler === Split finalize merge direct D2D failed, "
              << "fallback D2H->H2D" << std::endl;
        }
      }

      if (!moved_by_p2p && p < Pending.SplitQueues.size()) {
        detail::EventImplPtr ev_host =
            detail::Scheduler::getInstance().addMemoryMove(
                CopyReq, Pending.HostQueue, Pending.SplitQueues[p]);
        waitOfflineSplitEvent(ev_host, OfflineSplitWaitKind::Merge);
#ifdef SNMD_OFFLINE_SPLIT_STATS
        offlineSplitStats().MergeD2HBytes +=
            offlineRequirementBytes(CopyReq);
#endif
        HANDLER_TRACE_STREAM << "=== handler === Split finalize copy partition to host"
                  << std::endl;

        detail::EventImplPtr ev_merge =
            detail::Scheduler::getInstance().addMemoryMove(
                CopyReq, MergeQueue, Pending.HostQueue);
        waitOfflineSplitEvent(ev_merge, OfflineSplitWaitKind::Merge);
#ifdef SNMD_OFFLINE_SPLIT_STATS
        offlineSplitStats().MergeH2DBytes +=
            offlineRequirementBytes(CopyReq);
#endif
        HANDLER_TRACE_STREAM
            << "=== handler === Split finalize copy partition to merge device"
            << std::endl;
      }

      Rec->MCurContext = MergeCtx;
      HANDLER_TRACE_STREAM
          << "=== handler === Split finalize current context moved to merge "
          << "device" << std::endl;
    }
  }

  const uint64_t MaterializationEndNs = offlineNowNs();
  offlineSplitFinalizeTimes().push_back(
      {Pending.KernelCount, Pending.PartsCompleteNs,
       MaterializationEndNs >= MaterializationStartNs
           ? MaterializationEndNs - MaterializationStartNs
           : 0});
}

static void finalizePendingOfflineSplitsForKernel(
    detail::SyclKernelCg *KernelCg) {
  std::vector<PendingOfflineSplitMerge> &PendingSplits =
      pendingOfflineSplitMerges();

  if (KernelCg != nullptr && KernelCg->kernel_cg) {
    if (auto *ExecCG =
            dynamic_cast<detail::CGExecKernel *>(KernelCg->kernel_cg.get())) {
      for (detail::Requirement *Req : ExecCG->MRequirements) {
        if (Req != nullptr && offlineSplitWriteAccess(Req->MAccessMode)) {
          invalidateOfflineReadReplica(Req->MSYCLMemObj);
        }
      }
    }
  }

  // Split kernels share stable in-order queues.  A later pending kernel can
  // therefore already be queued behind an earlier one.  Waiting and merging
  // one pending split at a time would enqueue the earlier merge *after* that
  // later kernel, then only discover a failure from the later kernel while
  // executing the earlier merge.  First wait every dependency touched by the
  // current kernel, so errors remain attributed to the actual kernel/part and
  // no merge is inserted ahead of an unchecked pending dependency.
  for (PendingOfflineSplitMerge &Pending : PendingSplits) {
    if (kernelTouchesPendingOfflineSplit(KernelCg, Pending)) {
      waitPendingOfflineSplit(Pending);
    }
  }

  for (size_t I = 0; I < PendingSplits.size();) {
    if (kernelTouchesPendingOfflineSplit(KernelCg, PendingSplits[I])) {
      mergePendingOfflineSplit(PendingSplits[I]);
      PendingSplits.erase(PendingSplits.begin() + I);
    } else {
      ++I;
    }
  }
}

static void finalizeAllPendingOfflineSplits() {
  std::vector<PendingOfflineSplitMerge> &PendingSplits =
      pendingOfflineSplitMerges();
  for (PendingOfflineSplitMerge &Pending : PendingSplits) {
    waitPendingOfflineSplit(Pending);
  }
  for (PendingOfflineSplitMerge &Pending : PendingSplits) {
    mergePendingOfflineSplit(Pending);
  }
  PendingSplits.clear();
}

static void finalizeCompletedOfflineSplit(int KernelCount) {
  std::vector<PendingOfflineSplitMerge> &PendingSplits =
      pendingOfflineSplitMerges();
  for (size_t I = 0; I < PendingSplits.size(); ++I) {
    if (PendingSplits[I].KernelCount != KernelCount) {
      continue;
    }
    // Completion-driven gang scheduling guarantees no later command has been
    // queued on these devices before this acknowledgement. It is therefore
    // safe to materialize this one completed Split without the static path's
    // all-pending pre-wait.
    mergePendingOfflineSplit(PendingSplits[I]);
    PendingSplits.erase(PendingSplits.begin() + I);
    return;
  }
}

static void applyOfflineSplitFinalizeTimes(
    std::vector<OfflineProfileEvent> &ProfileEvents) {
  for (const OfflineSplitFinalizeTiming &FinalizeTime :
       offlineSplitFinalizeTimes()) {
    for (OfflineProfileEvent &ProfileEvent : ProfileEvents) {
      if (ProfileEvent.KernelCount == FinalizeTime.KernelCount) {
        if (FinalizeTime.PartsCompleteNs > ProfileEvent.HostEndNs) {
          ProfileEvent.HostEndNs = FinalizeTime.PartsCompleteNs;
        }
        ProfileEvent.MaterializationNs += FinalizeTime.MaterializationNs;
      }
    }
  }
  offlineSplitFinalizeTimes().clear();
}

static bool offlineSplitPartsComplete(int KernelCount) {
  for (const PendingOfflineSplitMerge &Pending :
       pendingOfflineSplitMerges()) {
    if (Pending.KernelCount != KernelCount) {
      continue;
    }
    if (!Pending.Events.empty()) {
      for (const detail::EventImplPtr &Event : Pending.Events) {
        if (Event) {
          try {
            const info::event_command_status Status =
                Event->get_info<info::event::command_execution_status>();
            // Backend execution failures are terminal negative statuses. Let
            // the subsequent wait propagate the asynchronous exception rather
            // than polling forever for a value equal to COMPLETE.
            if (static_cast<pi_int32>(Status) > PI_EVENT_COMPLETE) {
              return false;
            }
          } catch (...) {
            // A failed status query is also terminal for this polling loop;
            // waitPendingOfflineSplit() remains the source of the real error.
          }
        }
      }
      return true;
    }
    if (!Pending.Event) {
      return true;
    }
    try {
      const info::event_command_status Status =
          Pending.Event->get_info<info::event::command_execution_status>();
      return static_cast<pi_int32>(Status) <= PI_EVENT_COMPLETE;
    } catch (...) {
      return true;
    }
  }
  return true;
}
#endif

static bool offlineProfileEventComplete(
    const OfflineProfileEvent &ProfileEvent) {
  try {
#ifdef SNMD_OFFLINE
    if (ProfileEvent.NumParts > 1 &&
        !offlineSplitPartsComplete(ProfileEvent.KernelCount)) {
      return false;
    }
#endif
    sycl::event EventCopy = ProfileEvent.Event;
    const info::event_command_status Status =
        EventCopy.get_info<info::event::command_execution_status>();
    return static_cast<pi_int32>(Status) <= PI_EVENT_COMPLETE;
  } catch (...) {
    // Do not turn an event-query failure into an infinite scheduler stall.
    // The mandatory wait immediately after polling propagates the real error.
    return true;
  }
}

static void fillOfflineProfileData(int WaitCount,
                                   const OfflineProfileEvent &ProfileEvent,
                                   uint64_t Duration,
                                   S2DKernelProfileData &ProfileData) {
  ProfileData.pid = getpid();
  ProfileData.wait_count = WaitCount;
  ProfileData.kernel_count = ProfileEvent.KernelCount;
  ProfileData.device_index = ProfileEvent.DeviceIndex;
  ProfileData.num_parts = std::max(1, ProfileEvent.NumParts);
  ProfileData.duration_ns = Duration;
  ProfileData.kernel_key =
      ProfileEvent.KernelKey.empty()
          ? findOfflineKernelProfileKey(ProfileEvent.KernelCount)
          : ProfileEvent.KernelKey;
}

static void waitOfflineBatchForUserFence(
    const std::vector<OfflineProfileEvent> &ProfileEvents) {
  // scheduleOffline is entered from queue::wait().  Waiting for every event is
  // therefore required by the user-visible fence, not by profiling.  Keep the
  // synchronization explicit and separate so querying profiling timestamps
  // never creates an additional serialization dependency.
  for (const OfflineProfileEvent &ProfileEvent : ProfileEvents) {
    if (ProfileEvent.NumParts <= 1) {
      sycl::event EventCopy = ProfileEvent.Event;
      EventCopy.wait();
    }
  }
}

static bool collectOfflineProfilingInfo(
    int WaitCount, const OfflineProfileEvent &ProfileEvent,
    S2DKernelProfileData &ProfileData) {
  const uint64_t HostDuration =
      ProfileEvent.HostEndNs > ProfileEvent.HostStartNs
          ? ProfileEvent.HostEndNs - ProfileEvent.HostStartNs
          : 0;
  const uint64_t SplitDuration =
      ProfileEvent.MaterializationNs >
              std::numeric_limits<uint64_t>::max() - HostDuration
          ? std::numeric_limits<uint64_t>::max()
          : HostDuration + ProfileEvent.MaterializationNs;
  if (ProfileEvent.NumParts > 1 && SplitDuration > 0) {
    HANDLER_TRACE_STREAM << "=== handler === Offline profiling split wall "
                         << "kernel_count: " << ProfileEvent.KernelCount
                         << " device_index: " << ProfileEvent.DeviceIndex
                         << " num_parts: " << ProfileEvent.NumParts
                         << " compute_wall_ns: " << HostDuration
                         << " materialization_ns: "
                         << ProfileEvent.MaterializationNs
                         << " duration_ns: " << SplitDuration << std::endl;
    fillOfflineProfileData(WaitCount, ProfileEvent, SplitDuration,
                           ProfileData);
    return true;
  }

  try {
    sycl::event ProfileEventCopy = ProfileEvent.Event;
    uint64_t Start =
        ProfileEventCopy.get_profiling_info<
            info::event_profiling::command_start>();
    uint64_t End =
        ProfileEventCopy.get_profiling_info<
            info::event_profiling::command_end>();
    uint64_t Duration = End >= Start ? End - Start : 0;
    HANDLER_TRACE_STREAM << "=== handler === Offline profiling kernel_count: "
              << ProfileEvent.KernelCount << " device_index: "
              << ProfileEvent.DeviceIndex << " num_parts: "
              << ProfileEvent.NumParts << " start_ns: " << Start
              << " end_ns: " << End << " duration_ns: " << Duration
              << std::endl;

    fillOfflineProfileData(WaitCount, ProfileEvent, Duration, ProfileData);
    return Duration > 0;
  } catch (const std::exception &e) {
    HANDLER_TRACE_STREAM << "=== handler === Offline profiling unavailable for kernel_count: "
              << ProfileEvent.KernelCount << " reason: " << e.what()
              << std::endl;
  } catch (...) {
    HANDLER_TRACE_STREAM << "=== handler === Offline profiling unavailable for kernel_count: "
              << ProfileEvent.KernelCount << std::endl;
  }

  if (HostDuration > 0) {
    HANDLER_TRACE_STREAM << "=== handler === Offline profiling host fallback kernel_count: "
              << ProfileEvent.KernelCount << " device_index: "
              << ProfileEvent.DeviceIndex << " num_parts: "
              << ProfileEvent.NumParts << " duration_ns: " << HostDuration
              << std::endl;
    fillOfflineProfileData(WaitCount, ProfileEvent, HostDuration, ProfileData);
    return true;
  }

  return false;
}

static void processOfflineProfilingBatch(
    int WaitCount, const std::vector<OfflineProfileEvent> &ProfileEvents) {
  waitOfflineBatchForUserFence(ProfileEvents);

  S2DProfileBatchData Batch;
  for (const OfflineProfileEvent &ProfileEvent : ProfileEvents) {
    S2DKernelProfileData ProfileData;
    if (collectOfflineProfilingInfo(WaitCount, ProfileEvent, ProfileData)) {
      Batch.profiles.push_back(ProfileData);
    }
  }

#ifdef SCHEDULE_OFFLINE
  if (!Batch.profiles.empty()) {
    std::string SerializedData = Batch.serialize();
    sendOfflineMqPayload(mq_id_daemon, SerializedData, MAX_MSG_DAEMON_SIZE,
                         "Offline profiling");
    HANDLER_TRACE_STREAM << "=== handler === Offline profiling sent samples: "
                         << Batch.profiles.size() << std::endl;
  }
#endif
}
#endif

// Sets the submission state to indicate that an explicit kernel bundle has been
// set. Throws a sycl::exception with errc::invalid if the current state
// indicates that a specialization constant has been set.
void handler::setStateExplicitKernelBundle() {
  MImpl->setStateExplicitKernelBundle();
}

// Sets the submission state to indicate that a specialization constant has been
// set. Throws a sycl::exception with errc::invalid if the current state
// indicates that an explicit kernel bundle has been set.
void handler::setStateSpecConstSet() { MImpl->setStateSpecConstSet(); }

// Returns true if the submission state is EXPLICIT_KERNEL_BUNDLE_STATE and
// false otherwise.
bool handler::isStateExplicitKernelBundle() const {
  return MImpl->isStateExplicitKernelBundle();
}

// Returns a shared_ptr to the kernel_bundle.
// If there is no kernel_bundle created:
// returns newly created kernel_bundle if Insert is true
// returns shared_ptr(nullptr) if Insert is false
std::shared_ptr<detail::kernel_bundle_impl>
handler::getOrInsertHandlerKernelBundle(bool Insert) const {
  if (!MImpl->MKernelBundle && Insert) {
    MImpl->MKernelBundle =
        detail::getSyclObjImpl(get_kernel_bundle<bundle_state::input>(
            MQueue->get_context(), {MQueue->get_device()}, {}));
  }
  return MImpl->MKernelBundle;
}

// Sets kernel bundle to the provided one.
void handler::setHandlerKernelBundle(
    const std::shared_ptr<detail::kernel_bundle_impl> &NewKernelBundleImpPtr) {
  MImpl->MKernelBundle = NewKernelBundleImpPtr;
}

void handler::setHandlerKernelBundle(kernel Kernel) {
  // Kernel may not have an associated kernel bundle if it is created from a
  // program. As such, apply getSyclObjImpl directly on the kernel, i.e. not
  //  the other way around: getSyclObjImp(Kernel->get_kernel_bundle()).
  std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpl =
      detail::getSyclObjImpl(Kernel)->get_kernel_bundle();
  setHandlerKernelBundle(KernelBundleImpl);
}

#ifdef REBIND
namespace detail {
  extern device select_device(DSelectorInvocableType DeviceSelectorInvocable, bool rebind);
}
#endif

event handler::finalize() {
  // This block of code is needed only for reduction implementation.
  // It is harmless (does nothing) for everything else.
  if (MIsFinalized)
    return MLastEvent;
  MIsFinalized = true;

  // According to 4.7.6.9 of SYCL2020 spec, if a placeholder accessor is passed
  // to a command without being bound to a command group, an exception should
  // be thrown. There should be as many requirements as unique accessors,
  // otherwise some of the accessors are unbound, and thus we throw.
  {
    // A counter is not good enough since we can have the same accessor several
    // times as arg
    std::unordered_set<void *> accessors;
    for (const auto &arg : MArgs) {
      if (arg.MType != detail::kernel_param_kind_t::kind_accessor)
        continue;

      accessors.insert(arg.MPtr);
    }
    if (accessors.size() > MRequirements.size())
      throw sycl::exception(make_error_code(errc::kernel_argument),
                            "placeholder accessor must be bound by calling "
                            "handler::require() before it can be used.");
  }



// 【START】=======================================================
#ifdef SCHEDULE
  using namespace sycl::detail;
  const auto &cmdType = getType();
  if (cmdType == detail::CG::Kernel) {
    int &daemon_kernel_count = detail::ProgramManager::getInstance().kernel_count;
    daemon_kernel_count++;
    int &daemon_scale_count = detail::ProgramManager::getInstance().scale_count;
    // 快速跳过直到scale_count
    if (daemon_kernel_count < daemon_scale_count) {
      return MLastEvent;
    }
    //【scale】传输所需数据 OPTI 或执行前置kernel
    else if (daemon_kernel_count == daemon_scale_count) {
      // ====【scale 处理依赖】
      {
        S2DKernelReqData kernel_req_data;
        {
          detail::combineAccessModesOfReqs(MRequirements);

          fillOfflineKernelReqData(kernel_req_data, getpid(),
                                   daemon_kernel_count, MKernelName, MNDRDesc,
                                   MRequirements);

          std::string serialized_data = kernel_req_data.serialize();
          size_t message_size = serialized_data.size();

          mq_send(mq_id_daemon, serialized_data.c_str(), message_size, 0);
          HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === Scale mq_send kernel_req_data" << std::endl;
        }
      
        {
          for (int i = 0; i < MRequirements.size(); i++) {
            int daemon_req_count = i + 1;
            Requirement *Req = MRequirements[i];
            if (Req->MAccessMode == access::mode::read || Req->MAccessMode == access::mode::read_write || Req->MAccessMode == access::mode::atomic) {
              Requirement *hostReq = new Requirement(*Req);
              EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
              hostEvent->wait(hostEvent);
              delete hostReq;
              HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== Scale receiver add host acc" << std::endl;

              using DATA_TYPE = std::byte;
              size_t elem_size = Req->MElemSize;
              size_t buff_size = Req->MMemoryRange.size();
              HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== Scale receiver elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;

              SYCLMemObjI *MemObj = Req->MSYCLMemObj;
              SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
              void *UserPtr = BufferObj->getUserPtr();
              DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);

              std::vector<DATA_TYPE> host_data(elem_size * buff_size);
              SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, daemon_kernel_count, daemon_req_count, elem_size * buff_size);
              readFromSharedMemory(handle, host_data.data(), elem_size * buff_size);
              HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== Scale data read successfully." << std::endl;
              cleanupSharedMemory(handle, elem_size * buff_size);
              std::memcpy(DataPtr, host_data.data(), elem_size * buff_size);
              HANDLER_TRACE_STREAM << getpid() << " === handler === Scale mem copy" << std::endl;
            }
          }
        }
      }

      // ====【scale rebind】
      {
        device exec_device = detail::ProgramManager::getInstance().globalDevices.at(detail::ProgramManager::getInstance().scale_device);
        detail::DeviceImplPtr dp = detail::getSyclObjImpl(exec_device);
        HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === rebind_device is_gpu: " << exec_device.is_gpu() << std::endl;
        MQueue.reset(new detail::queue_impl(dp, detail::queue_impl::getDefaultOrNew(dp), MQueue->getAsyncHandler(), MQueue->getPropertyList()));
      }
    }
    //【通用情况】
    else {
      // ====【打包kernel内req发送给daemon】
      // 因为scheduler维护了kernel历史执行 所以不需要数据移动SameCtx判断
      // 即使SameCtx判断 有的Req可能并不在同一节点上 没有必要
      // OPTI Req对应内存的具体信息 如大小
      S2DKernelReqData kernel_req_data;
      {
        detail::combineAccessModesOfReqs(MRequirements);

        fillOfflineKernelReqData(kernel_req_data, getpid(),
                                 daemon_kernel_count, MKernelName, MNDRDesc,
                                 MRequirements);

        std::string serialized_data = kernel_req_data.serialize();
        size_t message_size = serialized_data.size();

        mq_send(mq_id_daemon, serialized_data.c_str(), message_size, 0);
        HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === mq_send kernel_req_data" << std::endl;
      }

      // ====【接收daemon执行决策】
      // 包括kernel是否执行以及哪个device执行
      D2SKernelExecInfo kernel_exec_info;
      {
        char buffer[MAX_MSG_DAEMON_SIZE];
        ssize_t bytes_received = mq_receive(mq_id_program, buffer, MAX_MSG_PROGRAM_SIZE, nullptr);
        if (bytes_received > 0) {
          std::string received_data(buffer, bytes_received);
          kernel_exec_info = D2SKernelExecInfo::deserialize(received_data);
        } else {
          std::string errorMsg = "Error: Process " + std::to_string(getpid()) + " PROGRAM mq_receive failed";
          perror(errorMsg.c_str());
          exit(1);
        }
        if (kernel_exec_info.scale_count > 1) {
          daemon_scale_count = kernel_exec_info.scale_count;
          detail::ProgramManager::getInstance().scale_device =
              clampOfflineDeviceIndex(kernel_exec_info.device_index);
          HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === scale_count: " << daemon_scale_count << " device_index: " << kernel_exec_info.device_index << std::endl;
          return MLastEvent;
        } else {
          HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === mq_receive kernel_exec_info === kernel_count: " << kernel_exec_info.kernel_count << " exec: " << kernel_exec_info.exec << " device_index: " << kernel_exec_info.device_index << " req_size: " << kernel_exec_info.req_counts.size() << std::endl;
        }
      }

      // ====【scale 提供依赖】
      if (kernel_exec_info.scale_count == -1) {
        for (int i = 0; i < MRequirements.size(); i++) {
          int daemon_req_count = i + 1;
          Requirement *Req = MRequirements[i];
          if (Req->MAccessMode == access::mode::read || Req->MAccessMode == access::mode::read_write || Req->MAccessMode == access::mode::atomic) {
            Requirement *hostReq = new Requirement(*Req);
            EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
            hostEvent->wait(hostEvent);
            delete hostReq;
            HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== Scale sender add host acc" << std::endl;

            using DATA_TYPE = std::byte;
            size_t elem_size = Req->MElemSize;
            size_t buff_size = Req->MMemoryRange.size();
            HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== Scale sender elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;

            SYCLMemObjI *MemObj = Req->MSYCLMemObj;
            SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
            void *UserPtr = BufferObj->getUserPtr();
            DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);

            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, daemon_kernel_count, daemon_req_count, elem_size * buff_size);
            writeToSharedMemory(handle, DataPtr, elem_size * buff_size);
            HANDLER_TRACE_STREAM << getpid() << " === handler === Scale send host data" << std::endl;

            waitForReadCompletion(handle);
            cleanupSharedMemory(handle, elem_size * buff_size);
            HANDLER_TRACE_STREAM << getpid() << " === handler === Scale waitForReadCompletion" << std::endl;
          }
        }
        return MLastEvent; // 提供完不执行要返回
      }

      // ====【处理kernel的依赖数据】scale完就不需要走这段流程
      else {
        if (daemon_kernel_count != kernel_exec_info.kernel_count) {
          HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === kernel count not match" << std::endl;
          exit(1);
        }
        auto &req_counts = kernel_exec_info.req_counts;
        // 如果不执行 检查是否需要host->device 完成后再返回
        if (!kernel_exec_info.exec) {
          if (req_counts.size() > 0) {
            for (int i = 0; i < MRequirements.size(); i++) {
              int daemon_req_count = i + 1;
              Requirement *Req = MRequirements[i];
              if ((std::find(req_counts.begin(), req_counts.end(), daemon_req_count) != req_counts.end()) && (Req->MAccessMode == access::mode::read || Req->MAccessMode == access::mode::read_write || Req->MAccessMode == access::mode::atomic)) {
                Requirement *hostReq = new Requirement(*Req);
                EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
                hostEvent->wait(hostEvent);
                delete hostReq;
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== sender add host acc" << std::endl;

                using DATA_TYPE = std::byte;
                size_t elem_size = Req->MElemSize;
                size_t buff_size = Req->MMemoryRange.size();
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== sender elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;

                SYCLMemObjI *MemObj = Req->MSYCLMemObj;
                SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
                void *UserPtr = BufferObj->getUserPtr();
                DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);

                SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, daemon_kernel_count, daemon_req_count, elem_size * buff_size);
                writeToSharedMemory(handle, DataPtr, elem_size * buff_size);
                HANDLER_TRACE_STREAM << getpid() << " === handler === send host data" << std::endl;

                waitForReadCompletion(handle);
                cleanupSharedMemory(handle, elem_size * buff_size);
                HANDLER_TRACE_STREAM << getpid() << " === handler === waitForReadCompletion" << std::endl;
              }
            }
          }
          HANDLER_TRACE_STREAM << getpid() << " === handler === kernel_count: " << daemon_kernel_count << " MLastEvent: " << &MLastEvent << std::endl;
          return MLastEvent;
        }
        // 如果执行 需要等待可能的通信数据 也需要host 被用的时候再从host->device
        else {
          if (req_counts.size() > 0) {
            for (int i = 0; i < MRequirements.size(); i++) {
              int daemon_req_count = i + 1;
              Requirement *Req = MRequirements[i];
              if ((std::find(req_counts.begin(), req_counts.end(), daemon_req_count) != req_counts.end()) && (Req->MAccessMode == access::mode::read || Req->MAccessMode == access::mode::read_write || Req->MAccessMode == access::mode::atomic)) {
                Requirement *hostReq = new Requirement(*Req);
                EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
                hostEvent->wait(hostEvent);
                delete hostReq;
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== receiver add host acc" << std::endl;

                using DATA_TYPE = std::byte;
                size_t elem_size = Req->MElemSize;
                size_t buff_size = Req->MMemoryRange.size();
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== receiver elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;

                SYCLMemObjI *MemObj = Req->MSYCLMemObj;
                SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
                void *UserPtr = BufferObj->getUserPtr();
                DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);

                std::vector<DATA_TYPE> host_data(elem_size * buff_size);
                SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, daemon_kernel_count, daemon_req_count, elem_size * buff_size);
                readFromSharedMemory(handle, host_data.data(), elem_size * buff_size);
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== Data read successfully." << std::endl;
                cleanupSharedMemory(handle, elem_size * buff_size);
                std::memcpy(DataPtr, host_data.data(), elem_size * buff_size);

                HANDLER_TRACE_STREAM << getpid() << " === handler === mem copy" << std::endl;
              }
            }
          }
        }
      }

      // ====【执行进程rebind】
      if (kernel_exec_info.exec) {
        const int ActualDeviceIndex =
            clampOfflineDeviceIndex(kernel_exec_info.device_index);
        device exec_device = detail::ProgramManager::getInstance().globalDevices.at(ActualDeviceIndex);
        detail::DeviceImplPtr dp = detail::getSyclObjImpl(exec_device);
        HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === rebind_device is_gpu: " << exec_device.is_gpu() << std::endl;
        MQueue.reset(new detail::queue_impl(dp, detail::queue_impl::getDefaultOrNew(dp), MQueue->getAsyncHandler(), MQueue->getPropertyList()));
      }
    }
  }
#endif

// 先不考虑scale
#ifdef SCHEDULE_OFFLINE
  using namespace sycl::detail;
  const auto &cmdType = getType();
  if (cmdType == detail::CG::Kernel) {
    int &daemon_kernel_count = detail::ProgramManager::getInstance().kernel_count;
    daemon_kernel_count++;
    //【通用情况】
    std::vector<S2DKernelReqData> &kernel_reqs = detail::ProgramManager::getInstance().kernel_reqs;
    // ====【打包存储kenrel内req】
    S2DKernelReqData kernel_req_data;
    {
      detail::combineAccessModesOfReqs(MRequirements);

      fillOfflineKernelReqData(kernel_req_data, getpid(),
                               daemon_kernel_count, MKernelName, MNDRDesc,
                               MRequirements);
    }
    kernel_reqs.push_back(kernel_req_data);

    HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === set sycl_kernel_cg: " << detail::ProgramManager::getInstance().kernel_count << " MQueue: " << MQueue << " MRequirements: " << MRequirements.size() << " -";
    for (Requirement *req : MRequirements) {
      HANDLER_TRACE_STREAM << " " << req->MSYCLMemObj;
    }
    HANDLER_TRACE_STREAM << " MEvents: " << MEvents.size() << std::endl;
  }
#endif

#ifdef REBIND_DISCARD
#ifdef TEST
    // ====【DONE】【测试忽略Kernel】
    {
      detail::ProgramManager::getInstance().kernel_count++;
      if (detail::ProgramManager::getInstance().kernel_count == 2) {
        return MLastEvent;
      }
    }

    // ====【DONE】【测试数据移动SameCtx判断】
    {
      std::unique_ptr<detail::CG> cmdGroup; // ？
      // 需要cmdGroup和MQueue
      // graph_builder里用Mrequirements和MEvents构建依赖图
      detail::combineAccessModesOfReqs(MRequirements);
      std::vector<Command *> ToEnqueue; // scheduler.cpp中的AuxiliaryCmds
      HANDLER_TRACE_STREAM << getpid() << " === handler === BEFORE REQS" << std::endl;

      // 在gloabal_handler初始化时获取所有device，并循环判断如果使用每个device的queue会造成几次数据移动
      for (device tempD : detail::ProgramManager::getInstance().globalDevices) {
        detail::DeviceImplPtr tempDP = detail::getSyclObjImpl(tempD);
        MQueue.reset(new detail::queue_impl(tempDP, detail::queue_impl::getDefaultOrNew(tempDP), MQueue->getAsyncHandler(), MQueue->getPropList()));
        int notSameCtxCount = 0;

        HANDLER_TRACE_STREAM << getpid() << " === handler === TRY: " << tempD.get_info<info::device::name>() << std::endl;
        for (Requirement *Req : MRequirements) {
          // Req->MAccessMode
          MemObjRecord *record = nullptr;
          AllocaCommandBase *allocaCmd = nullptr;
          bool isSameCtx = false;
          // std::cout << getpid() << " === handler === req 1"<< std::endl;

          // SYCLMemObj生命周期由用户代码管理 MemObjRecord生命周期由syclrt管理
          // SYCLMemObjI *MemObject = Req->MSYCLMemObj;
          // MemObjRecord *Record = getMemObjRecord(MemObject);
          //                      = MemObject->MRecord.get();
          record = detail::Scheduler::getInstance().MGraphBuilder.getOrInsertMemObjRecord(MQueue, Req, ToEnqueue);
          HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== record: " << record << std::endl;

          // 不清楚具体逻辑 不需要
          // detail::Scheduler::getInstance().MGraphBuilder.markModifiedIfWrite(record, Req);
          
          // 如果找不到对应的allocaCmd就创建了 不对需要销毁 如何销毁？
          // allocaCmd = detail::Scheduler::getInstance().MGraphBuilder.getOrCreateAllocaForReq(record, Req, MQueue, ToEnqueue);
          // std::cout << getpid() << " === handler === test_mem ==== allocaCmd: " << allocaCmd << std::endl;
          
          // std::cout << getpid() << " === handler === req 2"<< std::endl;
          isSameCtx = detail::sameCtx(MQueue->getContextImplPtr(), record->MCurContext);
          if (!isSameCtx) {
            notSameCtxCount++;
          }
        }
        HANDLER_TRACE_STREAM << getpid() << " === handler === notSameCtx: " << notSameCtxCount << std::endl;
      }
    }
#endif
  for (device d : detail::ProgramManager::getInstance().globalDevices) {
    HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === global_device cpu: " << d.is_cpu() << " gpu: " << d.is_gpu() << " acc: " << d.is_accelerator() << std::endl;
  }

  // 应被替换为运行时调度设备选择
  device d = detail::select_device(gpu_selector_v, true);

  // ========【测试设备选择】
  // device d;
  // if (detail::ProgramManager::getInstance().kernel_count == 1) {
  //   d = detail::ProgramManager::getInstance().globalDevices.at(1);
  // } else if (detail::ProgramManager::getInstance().kernel_count == 2) {
  //   d = detail::ProgramManager::getInstance().globalDevices.at(2);
  // } else {
  //   d = detail::ProgramManager::getInstance().globalDevices.at(2);
  // }

  // ========【测试cpu子设备】
  // device dd = detail::ProgramManager::getInstance().globalDevices.at(0);
  // std::vector<int> cts = {8, 4};
  // std::vector<device> subDevices = dd.create_sub_devices<info::partition_property::partition_by_counts>(cts);
  // device d;
  // if (detail::ProgramManager::getInstance().kernel_count == 1) {
  //   d = subDevices.at(0);
  // } else if (detail::ProgramManager::getInstance().kernel_count == 2) {
  //   d = subDevices.at(1);
  // } else {
  //   d = dd;
  // }

  detail::DeviceImplPtr dp = detail::getSyclObjImpl(d);
  HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === rebind_device is_gpu: " << d.is_gpu() << std::endl;
  // MQueue->rebindDevice(dp);
  MQueue.reset(new detail::queue_impl(dp, detail::queue_impl::getDefaultOrNew(dp), MQueue->getAsyncHandler(), MQueue->getPropertyList()));

#ifdef TEST
  // ========【DONE】【测试device->host】
  {
    using namespace sycl::detail;
    const auto &cmdType = getType();
    if (cmdType == detail::CG::Kernel) {
      detail::combineAccessModesOfReqs(MRequirements);
      std::vector<Command *> ToEnqueue; // scheduler中的AuxiliaryCmds
      detail::ProgramManager::getInstance().kernel_count++;
      int notSameCtxCount = 0;
      int testReqCount = 0;
      for (Requirement *Req : MRequirements) {
        testReqCount++;
        // Req->MAccessMode
        MemObjRecord *record = nullptr;
        AllocaCommandBase *allocaCmd = nullptr;
        bool isSameCtx = false;

        // std::cout << getpid() << " === handler === req 1"<< std::endl;
        // 获取的就是内存需求 对应数组
        record = detail::Scheduler::getInstance().MGraphBuilder.getOrInsertMemObjRecord(MQueue, Req, ToEnqueue);
        HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== record: " << record << std::endl;
        HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== memobj: " << Req->MSYCLMemObj << std::endl;

        if (detail::ProgramManager::getInstance().kernel_count == 3) {
          // std::cout << getpid() << " === handler === test_mem ==== kernel count 3" << std::endl;
          if (Req->MAccessMode == access::mode::read && testReqCount == 1) {
            // 尝试将device仅复制到host
      
            // addCopyBack 不太行 会导致后面再copyback的时候崩溃
            // Command *NewCmd = detail::Scheduler::getInstance().MGraphBuilder.addCopyBack(Req, ToEnqueue);
            // std::cout << getpid() << " === handler === test_mem ==== read copy back: " << record << std::endl;

            // // addHostAccessor 会销毁device上的拷贝？
            // EventImplPtr Event = detail::Scheduler::getInstance().addHostAccessor(Req);
            // Event->wait(Event);
            // // 需要在生命周期结束销毁 不知道哪里不用？
            // detail::ProgramManager::getInstance().releaseReqs.push_back(Req);

            Requirement *hostReq = new Requirement(*Req);
            EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
            hostEvent->wait(hostEvent);
            delete hostReq;
            // 会导致notSameCtx 不知道是否是删除了device上的数据？下次就要再加载进device？未测试

            HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== add host acc" << std::endl;

            // 尝试直接获取用户的数据指针
            using DATA_TYPE = float;
            SYCLMemObjI *MemObj = Req->MSYCLMemObj;
            SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
            void *UserPtr = BufferObj->getUserPtr();
            DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);
            int size = 256;
            for(int i = 0; i < size; i++) {
              for(int j = 0; j < size; j++) {
                  std::cerr << DataPtr[i * size + j] << " ";
              }
              std::cerr << std::endl;
            }
            HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== get user ptr" << std::endl;
          }
        }
        
        // std::cout << getpid() << " === handler === req 2"<< std::endl;
        isSameCtx = detail::sameCtx(MQueue->getContextImplPtr(), record->MCurContext);
        if (!isSameCtx) {
          notSameCtxCount++;
        }
      }
      HANDLER_TRACE_STREAM << getpid() << " === handler === REBIND === notSameCtx: " << notSameCtxCount << std::endl;
    }    
  }
#endif
#endif

// 开REBIND测试单进程运行时 必须同时开TEST_WITH_CLEAN 即手动处理REBIND设备和kernel_count
// #define TEST_WITH_CLEAN
#ifdef TEST_WITH_CLEAN
  using namespace sycl::detail;
  const auto &cmdType = getType();
  if (cmdType == detail::CG::Kernel) {
    auto &PM = detail::ProgramManager::getInstance();
    PM.kernel_count++;
    HANDLER_TRACE_STREAM << "=== handler === kernel_count: " << PM.kernel_count << std::endl;
    detail::combineAccessModesOfReqs(MRequirements);

    // CHECKED 测试完毕 ===【获取Req实际数据指针】
    // for (Requirement *Req : MRequirements) {
    //   SYCLMemObjI *MemObj = Req->MSYCLMemObj;
    //   SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
    //   size_t elemSize = Req->MElemSize;
    //   std::cout << getpid() << " === handler === test_mem ==== elemSize: " << elemSize << std::endl;
    //   range<3> memoryRange = Req->MMemoryRange;
    //   // size_t totalSize = memoryRange[0] * memoryRange[1] * memoryRange[2];
    //   size_t totalSize = memoryRange.size();
    //   std::cout << getpid() << " === handler === test_mem ==== totalSize: " << totalSize << std::endl;
    //   using DATA_TYPE = std::byte;
    //   std::vector<DATA_TYPE> host_data(elemSize * totalSize);
    //   void *UserPtr = BufferObj->getUserPtr();
    //   DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);
    // }

    // ===【REBIND】
    device d = PM.globalDevices.at(1);
    detail::DeviceImplPtr dp = detail::getSyclObjImpl(d);
    HANDLER_TRACE_STREAM << getpid() << " === handler === Process " << getpid() << " === rebind_device is_gpu: " << d.is_gpu() << std::endl;
    MQueue.reset(new detail::queue_impl(dp, detail::queue_impl::getDefaultOrNew(dp), MQueue->getAsyncHandler(), MQueue->getPropertyList()));

    // ===【SPLIT】
    if (PM.kernel_count == 4) {
      // CHECKED 删除 1. 将需要的Req从最新device拷回host 使用hostacc
      // for (Requirement *Req : MRequirements) {
      //   if (Req->MAccessMode == access::mode::read || Req->MAccessMode == access::mode::read_write || Req->MAccessMode == access::mode::atomic) {
      //     size_t elem_size = Req->MElemSize;
      //     size_t buff_size = Req->MMemoryRange.size();
      //     std::cout << getpid() << " === handler === test_mem ==== Split elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;
      //     Requirement *hostReq = new Requirement(*Req);
      //     EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
      //     hostEvent->wait(hostEvent);
      //     delete hostReq;
      //     std::cout << getpid() << " === handler === test_mem ==== Split add host acc" << std::endl;
      //   }
      // }
      // 2. 把需要的Req从host拷到另一个调度计算的device上
      //   2.1. 构造device
      //   2.2. 构造Mqueue
      //   2.3. 构造完整的SYCLMemObj 以及MemObjRecord 与device有关还是无关
      //   2.4. 由运行时控制数据拷贝
      //   2.5. 划分Req及附属数据结构的Range

      // 1. 构造Split相关数据结构 Device/Queue
      size_t &NumParts = PM.NumParts;

      std::vector<detail::QueueImplPtr> &SplitQueues_Write = PM.SplitQueues_Write;
      for (int i = 1; i <= NumParts; i++) {
        device SplitDevice = PM.globalDevices.at(i);
        detail::DeviceImplPtr SplitDP = detail::getSyclObjImpl(SplitDevice);
        std::shared_ptr<detail::queue_impl> SplitQueue = std::make_shared<detail::queue_impl>(SplitDP, detail::queue_impl::getDefaultOrNew(SplitDP), MQueue->getAsyncHandler(), MQueue->getPropertyList());
        SplitQueues_Write.push_back(SplitQueue);
      }

      detail::QueueImplPtr hostQ = Scheduler::getInstance().getDefaultHostQueue();
      auto hostCtx = hostQ->getContextImplPtr();

      // 2. 构造SplitReq和PM中存储相关数据结构
      // 被写Req在提交给不同SplitKernel时必须保持完整 由NDR控制计算区间
      std::vector<Requirement *> SplitReqs_onlyRead; // 只读 EF
      std::vector<Requirement *> SplitReqs_hasWrite; // 存在写 G
      std::vector<std::vector<Requirement*>> SplitReqs_Copy; // 存在写 构造拷回Req
      SplitReqs_Copy.resize(NumParts);
      // std::unordered_map<Requirement *, Requirement *> &SplitReqs_Remap = PM.SplitReqs_Remap;

      // CHECKED 已更新逻辑 CurCtx在不在host的逻辑通用化
      //   3.1. CurCtx不在host
      //        (只要有读)只读/读写 尝试D2DCpy 如果不可行回退从CurCtx->host->SplitDevice
      //        只写 说明不需要拷回 应维持host的CurCtx和Leaves 直接在Device上创建Alloca
      //   3.2. CurCtx在host
      //        (只要有读)只读/读写 直接从host拷到SplitDevice
      //        只写 直接在SplitDevice上创建Alloca

      // 3. 手动控制每个相关Req的数据移动 构造CopyReq
      for (Requirement *Req : MRequirements) {
        auto Mode = Req->MAccessMode;
        const bool onlyRead = (Mode == access::mode::read);
        const bool hasRead = (Mode == access::mode::read) ||
                             (Mode == access::mode::read_write) ||
                             (Mode == access::mode::atomic);
        const bool onlyWrite = (Mode == access::mode::write) ||
                               (Mode == access::mode::discard_write) ||
                               (Mode == access::mode::discard_read_write);
        const bool hasWrite = (Mode == access::mode::write) ||
                             (Mode == access::mode::discard_write) ||
                             (Mode == access::mode::discard_read_write) ||
                             (Mode == access::mode::read_write) ||
                             (Mode == access::mode::atomic);

        // 可能Req还没建立Record 如此时只有host的G
        bool isRecorded = detail::Scheduler::getInstance().getMemObjRecord(Req) != nullptr;
        MemObjRecord *ReqRecord = isRecorded ? detail::Scheduler::getInstance().getMemObjRecord(Req) : nullptr;
        ContextImplPtr ReqCurCtx = isRecorded ? detail::Scheduler::getInstance().getMemObjRecord(Req)->MCurContext : hostCtx;
        if (ReqCurCtx != hostCtx) {
          HANDLER_TRACE_STREAM << " === handler === Split step3 Req:" << Req << "->" << Req->MSYCLMemObj << " CurCtx not host\n";
        } else {
          HANDLER_TRACE_STREAM << " === handler === Split step3 Req:" << Req << "->" << Req->MSYCLMemObj << " CurCtx is host\n";
        }

        // 3.1【有读/只写 只用来处理 数据移动】
        // 有读 SrcDevice->SplitDevice
        if (hasRead) {
          QueueImplPtr SrcQueue = hostQ;
          if (ReqCurCtx != hostCtx && ReqRecord != nullptr) {
            SrcQueue = nullptr;
            for (AllocaCommandBase *AllocaCmd : ReqRecord->MAllocaCommands) {
              if (AllocaCmd->getQueue() != nullptr && AllocaCmd->getQueue()->getContextImplPtr() == ReqCurCtx) {
                SrcQueue = AllocaCmd->getQueue();
                break;
              }
            }
            HANDLER_TRACE_STREAM << "=== handler === Split step3 SrcQueue: " << SrcQueue << std::endl;
          }

          for (const QueueImplPtr &SplitQueue : SplitQueues_Write) {
            if (ReqCurCtx != hostCtx && SplitQueue->getContextImplPtr() == ReqCurCtx) {
              HANDLER_TRACE_STREAM << "=== handler === Split step3 SplitQueue is SrcQueue, continue\n";
              continue;
            }

            bool moved_by_p2p = false;
            if (ReqCurCtx != hostCtx && SrcQueue != nullptr) {
              try {
                EventImplPtr ev_p2p = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, SrcQueue);
                ev_p2p->wait(ev_p2p);
                moved_by_p2p = true;
                HANDLER_TRACE_STREAM << "=== handler === Split step3 direct D2D success\n";
              } catch (const std::exception &e) {
                HANDLER_TRACE_STREAM << "=== handler === Split step3 direct D2D failed, fallback D2H->H2D, reason: " << e.what() << "\n";
              } catch (...) {
                HANDLER_TRACE_STREAM << "=== handler === Split step3 direct D2D failed, fallback D2H->H2D\n";
              }
            }

            if (!moved_by_p2p) {
              if (ReqCurCtx != hostCtx && SrcQueue != nullptr) {
                EventImplPtr ev_host = detail::Scheduler::getInstance().addMemoryMove(Req, hostQ, SrcQueue);
                ev_host->wait(ev_host);
                HANDLER_TRACE_STREAM << "=== handler === Split step3 copy back host\n";
              }

              EventImplPtr ev_split = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, hostQ);
              ev_split->wait(ev_split);
              HANDLER_TRACE_STREAM << "=== handler === Split step3 copy to split device\n";
            }
          }
        }
        // 只写 直接在SplitDevice上创建Alloca
        else {
          std::vector<Command *> ToEnqueue;
          MemObjRecord *SplitRecord = ReqRecord;
          if (SplitRecord == nullptr)
            SplitRecord = detail::Scheduler::getInstance().MGraphBuilder.getOrInsertMemObjRecord(hostQ, Req, ToEnqueue);

          for (const QueueImplPtr &SplitQueue : SplitQueues_Write) {
            detail::Scheduler::getInstance().MGraphBuilder.getOrCreateAllocaForSplitReq(SplitRecord, Req, SplitQueue, ToEnqueue);
          }
        }

        // 3.2【只读/有写 只用来处理 写回逻辑】
        // 只读 无需额外逻辑
        if (onlyRead) {
          SplitReqs_onlyRead.push_back(Req);
        }
        // 有写 计算完后数据需要写回CurCtx
        else {
          SplitReqs_hasWrite.push_back(Req);
          range<3> FullRange = Req->MMemoryRange;
          size_t dim0 = FullRange[0];
          size_t chunk = dim0 / NumParts;
          for (size_t p = 0; p < NumParts; p++) {
            size_t begin0 = p * chunk;
            size_t end0 = (p + 1 == NumParts) ? (dim0) : (begin0 + chunk);
            size_t part0 = end0 - begin0;
            HANDLER_TRACE_STREAM << "=== handler === Split step3 Write part " << p << " begin: " << begin0 << " end: " << end0 << " range: " << part0 << "," << FullRange[1] << "," << FullRange[2] << "\n";

            Requirement *CopyReq = new Requirement(*Req);
            CopyReq->MOffset = id<3>(begin0, 0, 0);
            CopyReq->MAccessRange = range<3>(part0, FullRange[1], FullRange[2]);
            CopyReq->MMemoryRange = FullRange;
            CopyReq->MOffsetInBytes = begin0 * FullRange[1] * FullRange[2] * Req->MElemSize;
            SplitReqs_Copy[p].push_back(CopyReq);
          }
        }

        // CHECKED 已更新逻辑 不以ReqCurCtx是否为host作外层区分
        // if (ReqCurCtx != hostCtx) {
        //   std::cout << " === handler === Split step3 Req:" << Req << "->" << Req->MSYCLMemObj << " ReqCurCtx not host\n";
        //   if (hasRead) { // 有读 拷到SplitDevice
        //     QueueImplPtr SrcQueue = nullptr;
        //     // 通过ReqCurCtx获取所在Queue
        //     for (AllocaCommandBase *AllocaCmd : ReqRecord->MAllocaCommands) {
        //       if (AllocaCmd->getQueue() != nullptr && AllocaCmd->getQueue()->getContextImplPtr() == ReqCurCtx) {
        //         SrcQueue = AllocaCmd->getQueue();
        //         break;
        //       }
        //     }
        //     std::cout << "=== handler === Split step3 SrcQueue: " << SrcQueue << std::endl;
        //     //【注意】SplitQueue是被构造出来的 不能通过Queue判断同一个设备
        //     for (const QueueImplPtr &SplitQueue : SplitQueues_Write) {
        //       if (SplitQueue->getContextImplPtr() == ReqCurCtx) {
        //         std::cout << "=== handler === Split step3 SplitQueue is SrcQueue, continue\n";
        //         continue;
        //       }
        //       // std::cout << "=== handler === Split step3 Try SplitQueue: " << SplitQueue << std::endl;
        //       // if (SplitQueue == SrcQueue) continue;
        //       bool moved_by_p2p = false;
        //       try {
        //         EventImplPtr ev_p2p = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, SrcQueue);
        //         ev_p2p->wait(ev_p2p);
        //         moved_by_p2p = true;
        //         std::cout << "=== handler === Split step3 direct D2D success\n";
        //       } catch (const std::exception &e) {
        //         std::cout << "=== handler === Split step3 direct D2D failed, fallback D2H->H2D, reason: " << e.what() << "\n";
        //       } catch (...) {
        //         std::cout << "=== handler === Split step3 direct D2D failed, fallback D2H->H2D\n";
        //       }
        //       if (!moved_by_p2p) {
        //         // 从CurCtx拷回host
        //         EventImplPtr ev_host = detail::Scheduler::getInstance().addMemoryMove(Req, hostQ, SrcQueue);
        //         ev_host->wait(ev_host);
        //         std::cout << "=== handler === Split step3 copy back host\n";
        //         // 从host拷到SplitDevice
        //         EventImplPtr ev_split = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, hostQ);
        //         ev_split->wait(ev_split);
        //         std::cout << "=== handler === Split step3 copy to split device\n";
        //       }
        //     }
        //     // CHECKED 通用化 直接进行D->D的数据拷贝 如果不行会退回D->H->D
        //     // bool moved_by_p2p = false;
        //     // try {
        //     //   EventImplPtr ev_p2p =
        //     //       detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, MQueue);
        //     //   ev_p2p->wait(ev_p2p);
        //     //   moved_by_p2p = true;
        //     //   std::cout << "=== handler === Split step3 direct D2D success\n";
        //     // } catch (const std::exception &e) {
        //     //   std::cout << "=== handler === Split step3 direct D2D failed, fallback D2H->H2D, reason: " << e.what() << "\n";
        //     // } catch (...) {
        //     //   std::cout << "=== handler === Split step3 direct D2D failed, fallback D2H->H2D\n";
        //     // }
        //     // if (!moved_by_p2p) {
        //     //   // 从CurCtx拷回host
        //     //   EventImplPtr ev_host = detail::Scheduler::getInstance().addMemoryMove(Req, hostQ, MQueue);
        //     //   ev_host->wait(ev_host);
        //     //   std::cout << "=== handler === Split step3 copy back host\n";
        //     //   // 从host拷到SplitDevice
        //     //   EventImplPtr ev_split = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, hostQ);
        //     //   ev_split->wait(ev_split);
        //     //   std::cout << "=== handler === Split step3 copy to split device\n";
        //     // }
        //   } else { // 只写 需要在SplitDevice上创建Alloca 不需要拷贝
        //     // TODO
        //   }
        //   if (onlyRead) { // 只读 无额外逻辑
        //     SplitReqs_onlyRead.push_back(Req);
        //   } else { // 有写 算完后CurCtx放哪？
        //     SplitReqs_hasWrite.push_back(Req);
        //     // TODO 如G=E*F后 G=G+1
        //     // 数据需要拷回原定CurCtx
        //     // **注意** 所以这两段也应该通用化
        //   }
        // } else { // ReqCurCtx == hostCtx
        //   std::cout << " === handler === Split step3 Req:" << Req << "->" << Req->MSYCLMemObj << " ReqCurCtx is host\n";
        //   if (hasRead) { // 有读 从host拷到SplitDevice
        //     for (const QueueImplPtr &SplitQueue : SplitQueues_Write) {
        //       EventImplPtr ev_split = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, hostQ);
        //       ev_split->wait(ev_split);
        //       std::cout << "=== handler === Split step3 copy to split device\n";
        //     }
        //     // CHECKED 通用化
        //     // EventImplPtr ev_mqueue = detail::Scheduler::getInstance().addMemoryMove(Req, MQueue, hostQ);
        //     // ev_mqueue->wait(ev_mqueue);
        //     // std::cout << "=== handler === Split step3 copy to mqueue\n";
        //     // EventImplPtr ev_split = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, hostQ);
        //     // ev_split->wait(ev_split);
        //     // std::cout << "=== handler === Split step3 copy to split device\n";
        //   } else { // 只写 需要在SplitDevice上创建Alloca 不需要拷贝
        //     // TODO 
        //   }
        //   if (onlyRead) { // 只读
        //     SplitReqs_onlyRead.push_back(Req);
        //   } else { // 有写
        //     SplitReqs_hasWrite.push_back(Req);
        //     range<3> FullRange = Req->MMemoryRange;
        //     size_t dim0 = FullRange[0];
        //     size_t chunk = dim0 / NumParts;
        //     for (size_t p = 0; p < NumParts; p++) {
        //       size_t begin0 = p * chunk;
        //       size_t end0 = (p + 1 == NumParts) ? (dim0) : (begin0 + chunk);
        //       size_t part0 = end0 - begin0;
        //       std::cout << "=== handler === Split step3 Write part " << p << " begin: " << begin0 << " end: " << end0 << " range: " << part0 << "," << FullRange[1] << "," << FullRange[2] << "\n";
        //       Requirement *CopyReq = new Requirement(*Req);
        //       CopyReq->MOffset = id<3>(begin0, 0, 0);
        //       CopyReq->MAccessRange = range<3>(part0, FullRange[1], FullRange[2]);
        //       CopyReq->MMemoryRange = FullRange;
        //       CopyReq->MOffsetInBytes = begin0 * FullRange[1] * FullRange[2] * Req->MElemSize;
        //       SplitReqs_Copy[p].push_back(CopyReq);
        //     }
        //   }
        // }
      }

      // CHECKED 已更新逻辑 变为有读/只写和只读/有写
      // for (Requirement *Req : MRequirements) {
      //   auto Mode = Req->MAccessMode;
      //   const bool isRead = (Mode == access::mode::read);
      //   const bool isWrite = (Mode == access::mode::write) ||
      //                        (Mode == access::mode::discard_write) ||
      //                        (Mode == access::mode::discard_read_write) ||
      //                        (Mode == access::mode::read_write) ||
      //                        (Mode == access::mode::atomic);
      //   const bool isReadWrite = (Mode == access::mode::read_write) || (Mode == access::mode::atomic);
      //   if (!isRead && !isWrite) continue;        
      //   size_t elem_size = Req->MElemSize;
      //   size_t buff_size = Req->MMemoryRange.size();
      //   std::cout << getpid() << " === handler === test_mem ==== Split step2 elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;
      //   // E F 都在GPU上 先拷回host 再拷到split上
      //   if (isRead) {
      //     // CHECKED 删除 这里不应该用hostacc 直接addmemmove就可以
      //     // EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(Req);
      //     // hostEvent->wait(hostEvent);
      //     EventImplPtr host_ev = detail::Scheduler::getInstance().addMemoryMove(Req, hostQ, MQueue);
      //     host_ev->wait(host_ev);
      //     std::cout << "=== handler === Split step2 EF copy host\n";
      //     // CHECKED 测试完毕 在这里通过获取真实MemObj的方法验证E的第二行前10个值是否正确
      //     // {
      //     //   using DATA_TYPE = float; // 3mm_3kernel.cpp 里是 float
      //     //   SYCLMemObjI *MemObj = Req->MSYCLMemObj;
      //     //   SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
      //     //   void *UserPtr = BufferObj->getUserPtr();
      //     //   DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);
      //     //   const range<3> &R = Req->MMemoryRange;
      //     //   const size_t rows = R[0];
      //     //   const size_t cols = R[1] * R[2]; // 对 2D buffer 通常等价于列数
      //     //   const size_t row = 1;             // 第二行（0-based）
      //     //   const size_t n = std::min<size_t>(10, cols);
      //     //   std::cout << "=== handler === verify Req " << Req
      //     //             << " MemObj " << MemObj
      //     //             << " row1 first " << n << " values: ";
      //     //   if (DataPtr && rows > row && cols > 0) {
      //     //     const size_t base = row * cols;
      //     //     for (size_t j = 0; j < n; ++j) {
      //     //       std::cout << DataPtr[base + j] << (j + 1 == n ? '\n' : ' ');
      //     //     }
      //     //   } else {
      //     //     std::cout << "[invalid ptr or shape] rows=" << rows
      //     //               << " cols=" << cols << '\n';
      //     //   }
      //     // }
      //     EventImplPtr ev_split = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, hostQ);
      //     ev_split->wait(ev_split);
      //     std::cout << "=== handler === Split step2 EF copy gpu2\n";
      //   }
      //   // G 还没有alloca 拷到两个GPU上 第一次自动创建hostalloca
      //   if (isReadWrite) {
      //     EventImplPtr ev = detail::Scheduler::getInstance().addMemoryMove(Req, MQueue, hostQ);
      //     ev->wait(ev);
      //     std::cout << "=== handler === Split step2 G copy gpu1\n";
      //     EventImplPtr ev_split = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, hostQ);
      //     ev_split->wait(ev_split);
      //     std::cout << "=== handler === Split step2 G copy gpu2\n";
      //   }
      //   // CHECKED 删除 这里ReadWrite好像有错 needInit不能保证已经在host上
      //   // 提前拷贝不能等待第三步 第三步直接建图
      //   // const bool needInit = (Mode != access::mode::write) &&
      //   //                       (Mode != access::mode::discard_write) &&
      //   //                       (Mode != access::mode::discard_read_write);
      //   // if (needInit) {
      //   //   std::cout << "=== handler === Split step2 need init\n";
      //   //   // CHECK 就是从MQueue->host的 暂且不用重复拷贝的逻辑
      //   //   // EventImplPtr ev = detail::Scheduler::getInstance().addMemoryMove(Req, MQueue, hostQ);
      //   //   // ev->wait(ev);
      //   //   // std::cout << "=== handler === Split step2 copy gpu1\n";
      //   //   EventImplPtr ev_split = detail::Scheduler::getInstance().addMemoryMove(Req, SplitQueue, hostQ);
      //   //   ev_split->wait(ev_split);
      //   //   std::cout << "=== handler === Split step2 copy gpu2\n";
      //   // }
      //   // EF
      //   if (isRead) {
      //     std::cout << "=== handler === Split step2 Read Req: " << Req << "->" << Req->MSYCLMemObj << "\n";
      //     SplitReqs_Read.push_back(Req);
      //     // CHECKED 删除 已经提前拷贝完了这里不应该继续
      //     // EventImplPtr ev = detail::Scheduler::getInstance().addMemoryMove(Req, MQueue, hostQ);
      //     // ev->wait(ev);
      //     // auto *SplitReq = new Requirement(*Req);
      //     // SplitReqs_Read.push_back(SplitReq);
      //     // EventImplPtr ev_split = detail::Scheduler::getInstance().addMemoryMove(SplitReq, SplitQueue, hostQ);
      //     // ev_split->wait(ev_split);
      //   }
      //   // G
      //   else {
      //     std::cout << "=== handler === Split step2 Write Req: " << Req << "->" << Req->MSYCLMemObj << "\n";
      //     range<3> FullRange = Req->MMemoryRange;
      //     size_t dim0 = FullRange[0];
      //     size_t chunk = dim0 / NumParts;
      //     // auto *SplitReq = new Requirement(*Req);
      //     for (size_t p = 0; p < NumParts; p++) {
      //       size_t begin0 = p * chunk;
      //       size_t end0 = (p + 1 == NumParts) ? (dim0) : (begin0 + chunk);
      //       size_t part0 = end0 - begin0;
      //       std::cout << "=== handler === Split step2 Write part " << p << " begin: " << begin0 << " end: " << end0 << " range: " << part0 << "," << FullRange[1] << "," << FullRange[2] << "\n";
      //       // 第一个分片用原Req
      //       if (p == 0) {
      //         SplitReqs_Write[0].push_back(Req);
      //         SplitQueues_Write[0] = MQueue;
      //         Requirement *CopyReq = new Requirement(*Req);
      //         CopyReq->MOffset = id<3>(begin0, 0, 0);
      //         CopyReq->MAccessRange = range<3>(part0, FullRange[1], FullRange[2]);
      //         CopyReq->MMemoryRange = FullRange;
      //         CopyReq->MOffsetInBytes = begin0 * FullRange[1] * FullRange[2] * Req->MElemSize;
      //         SplitReqs_Copy[0].push_back(CopyReq);
      //       }
      //       else {
      //         SplitReqs_Write[1].push_back(Req);
      //         SplitQueues_Write[1] = SplitQueue;
      //         // SplitReqs_Remap[Req] = SplitReq;
      //         std::cout << "=== handler === Split step2 Write Split Req: " << Req << "->" << Req->MSYCLMemObj << "\n";
      //         Requirement *CopyReq = new Requirement(*Req);
      //         CopyReq->MOffset = id<3>(begin0, 0, 0);
      //         CopyReq->MAccessRange = range<3>(part0, FullRange[1], FullRange[2]);
      //         CopyReq->MMemoryRange = FullRange;
      //         CopyReq->MOffsetInBytes = begin0 * FullRange[1] * FullRange[2] * Req->MElemSize;
      //         SplitReqs_Copy[1].push_back(CopyReq);
      //       }
      //     }
      //   }
      // }

      // 4. 避开CommandGroup 在GraphBuilder::addCG中 克隆CGExecKernel 创建ExecCGCommand
      std::unique_ptr<detail::CG> CommandGroup;
      CommandGroup.reset(new detail::CGExecKernel(
          std::move(MNDRDesc), std::move(MHostKernel), std::move(MKernel),
          std::move(MImpl->MKernelBundle), std::move(MArgsStorage),
          std::move(MAccStorage), std::move(MSharedPtrStorage),
          std::move(MRequirements), std::move(MEvents), std::move(MArgs),
          MKernelName, MOSModuleHandle, std::move(MStreamStorage),
          std::move(MImpl->MAuxiliaryResources), MCGType,
          MImpl->MKernelCacheConfig, MCodeLoc));

      detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
          std::move(CommandGroup), MQueue);

      HANDLER_TRACE_STREAM << "=== handler === Split before wait\n";
      Event->wait(Event);
      HANDLER_TRACE_STREAM << "=== handler === Split after wait\n";

      // CHECKED 测试完毕 在addMemMove前验证G的host是否都为初始化的0值
      // for (size_t p = 0; p < NumParts; ++p) {
      //   for (Requirement *PartReq : SplitReqs_Write[p]) {
      //     {
      //       using DATA_TYPE = float; // 3mm_3kernel.cpp 里是 float
      //       SYCLMemObjI *MemObj = PartReq->MSYCLMemObj;
      //       SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
      //       void *UserPtr = BufferObj->getUserPtr();
      //       DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);
      //       const range<3> &R = PartReq->MMemoryRange;
      //       const size_t rows = R[0];
      //       const size_t cols = R[1] * R[2]; // 对 2D buffer 通常等价于列数
      //       const size_t row1 = 1;
      //       const size_t row = 128;
      //       const size_t n = std::min<size_t>(10, cols);
      //       std::cout << "=== handler === verify PartReq G before addMemMove " << PartReq
      //                 << " MemObj " << MemObj
      //                 << " row1 first " << n << " values: ";
      //       if (DataPtr && rows > row && cols > 0) {
      //         const size_t base1 = row1 * cols;
      //         for (size_t j = 0; j < n; ++j) {
      //           std::cout << DataPtr[base1 + j] << (j + 1 == n ? '\n' : ' ');
      //         }
      //         const size_t base = row * cols;
      //         for (size_t j = 0; j < n; ++j) {
      //           std::cout << DataPtr[base + j] << (j + 1 == n ? '\n' : ' ');
      //         }
      //       } else {
      //         std::cout << "[invalid ptr or shape] rows=" << rows
      //                   << " cols=" << cols << '\n';
      //       }
      //     }
      //   }
      // }

      // 5. 把Kernel更新后的【有写】Req的对应计算分片从各SplitDevice拷回SrcCtx 一定是Record的CurCtx
      // 如果SrcCtx是Device 就有D2D和回退D2H2D 如果SrcCtx是host 就直接D2H
      for (size_t p = 0; p < NumParts; ++p) {
        for (int i = 0; i < SplitReqs_Copy[p].size(); ++i) {
          Requirement *CopyReq = SplitReqs_Copy[p][i];
          MemObjRecord *Rec = detail::Scheduler::getInstance().getMemObjRecord(CopyReq);
          ContextImplPtr SrcCtx = Rec->MCurContext;
          QueueImplPtr SrcQueue = hostQ;
          if (SrcCtx != hostCtx) {
            SrcQueue = nullptr;
            for (AllocaCommandBase *AllocaCmd : Rec->MAllocaCommands) {
              if (AllocaCmd->getQueue()->getContextImplPtr() == SrcCtx) {
                SrcQueue = AllocaCmd->getQueue();
                break;
              }
            }
          }
          HANDLER_TRACE_STREAM << "=== handler === Split step5 PartReq " << CopyReq << " Record: " << Rec << " SrcCtx: " << SrcCtx << " SrcQueue: " << SrcQueue << " is host: " << (SrcCtx == hostCtx ? "true" : "false") << "\n";

          bool moved_by_p2p = false;
          if (SrcCtx != hostCtx) {
            if (SplitQueues_Write[p]->getContextImplPtr() == SrcCtx) {
              HANDLER_TRACE_STREAM << "=== handler === Split step5 SplitQueue is SrcQueue, continue\n";
              continue;
            }
            
            try {
              EventImplPtr ev_p2p = detail::Scheduler::getInstance().addMemoryMove(CopyReq, SrcQueue, SplitQueues_Write[p]);
              ev_p2p->wait(ev_p2p);
              moved_by_p2p = true;
              HANDLER_TRACE_STREAM << "=== handler === Split step5 direct D2D success\n";
            } catch (const std::exception &e) {
              HANDLER_TRACE_STREAM << "=== handler === Split step5 direct D2D failed, fallback D2H->H2D, reason: " << e.what() << "\n";
            } catch (...) {
              HANDLER_TRACE_STREAM << "=== handler === Split step5 direct D2D failed, fallback D2H->H2D\n";
            }
          }

          if (!moved_by_p2p) {
            EventImplPtr ev_host = detail::Scheduler::getInstance().addMemoryMove(CopyReq, hostQ, SplitQueues_Write[p]);
            ev_host->wait(ev_host);
            HANDLER_TRACE_STREAM << "=== handler === Split step5 copy back host\n";

            if (SrcCtx != hostCtx) {
              EventImplPtr ev_src = detail::Scheduler::getInstance().addMemoryMove(CopyReq, SrcQueue, hostQ);
              ev_src->wait(ev_src);
              HANDLER_TRACE_STREAM << "=== handler === Split step5 copy to src device\n";
            }
          }

          // CHECKED 已更新逻辑 对比输出Req对应Record的MCurContext和hostQ的ContextImplPtr是否一致
          // const QueueImplPtr &HostQueue = detail::Scheduler::getInstance().getDefaultHostQueue();
          // MemObjRecord *Rec = detail::Scheduler::getInstance().getMemObjRecord(SplitReqs_Copy[p][i]);
          // std::cout << "=== handler === Split step5 PartReq " << SplitReqs_Copy[p][i] << " Record: " << Rec << " MCurContext: " << (Rec ? Rec->MCurContext : nullptr) << " HostQueue Context: " << (HostQueue ? HostQueue->getContextImplPtr() : nullptr) << "\n";
          // // 进行host写回的PartReq要从完整的Req区分开 kernel所需Req要是完整的 不然计算偏移会导致错误
          // Requirement *CopyReq = SplitReqs_Copy[p][i];
          // EventImplPtr ev = detail::Scheduler::getInstance().addMemoryMove(CopyReq, hostQ, SplitQueues_Write[p]);
          // ev->wait(ev);

          // CHECKED 测试完毕 在这里通过获取真实MemObj的方法验证G的第二行前10个值是否正确
          // Requirement *OrigReq = SplitReqs_hasWrite[i];
          // {
          //   using DATA_TYPE = float; // 3mm_3kernel.cpp 里是 float
          //   SYCLMemObjI *MemObj = OrigReq->MSYCLMemObj;
          //   SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
          //   void *UserPtr = BufferObj->getUserPtr();
          //   DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);
          //   const range<3> &R = OrigReq->MMemoryRange;
          //   const size_t rows = R[0];
          //   const size_t cols = R[1] * R[2]; // 对 2D buffer 通常等价于列数
          //   const size_t row1 = 1;
          //   const size_t row = 128;             // 第二行（0-based）
          //   const size_t n = std::min<size_t>(10, cols);
          //   std::cout << "=== handler === verify OrigReq G " << OrigReq
          //             << " MemObj " << MemObj
          //             << " row1 first " << n << " values: ";
          //   if (DataPtr && rows > row && cols > 0) {
          //     const size_t base1 = row1 * cols;
          //     for (size_t j = 0; j < n; ++j) {
          //       std::cout << DataPtr[base1 + j] << (j + 1 == n ? '\n' : ' ');
          //     }
          //     const size_t base = row * cols;
          //     for (size_t j = 0; j < n; ++j) {
          //       std::cout << DataPtr[base + j] << (j + 1 == n ? '\n' : ' ');
          //     }
          //   } else {
          //     std::cout << "[invalid ptr or shape] rows=" << rows
          //               << " cols=" << cols << '\n';
          //   }
          // }
        }
        // CHECKED 删除 G在创建Record时由hostQueue生成 而手动控制拷贝SplitDevice和写回都不改变CurCtx 自然保持host的SameCtx
        // 每次最后一个Part结束后更新此Req对应Record的MCurContext
        // if (MemObjRecord *Rec = detail::Scheduler::getInstance().getMemObjRecord(SplitReqs_Write[p][0])) {
        //   Rec->MCurContext = hostQ->getContextImplPtr();
        // }
        // CHECKED 删除 错误的对原始Req的拷回
        // for (Requirement *Req : SplitReqs_Write[p]) {
        //   EventImplPtr ev = detail::Scheduler::getInstance().addMemoryMove(Req, hostQ, SplitQueues_Write[p]);
        //   ev->wait(ev);
        //   if (MemObjRecord *Rec = detail::Scheduler::getInstance().getMemObjRecord(Req)) {
        //     Rec->MCurContext = hostQ->getContextImplPtr();
        //   }
        // }
      }

      MLastEvent = detail::createSyclObjFromImpl<event>(Event);
      return MLastEvent; 
    }

    //【TEST kernel切分计算】===【对第3个kernel 修改NDRange：只跑前一半行】===
    #ifdef TEST_NDR_SPLIT
    if (detail::ProgramManager::getInstance().kernel_count == 3) {
      auto &NDR = MNDRDesc;

      HANDLER_TRACE_STREAM << "=== handler === [K3] before split: Dims = "
                << NDR.Dims
                << " GlobalSize = {"
                << NDR.GlobalSize[0] << ", "
                << NDR.GlobalSize[1] << ", "
                << NDR.GlobalSize[2] << "}, "
                << "Offset = {"
                << NDR.GlobalOffset[0] << ", "
                << NDR.GlobalOffset[1] << ", "
                << NDR.GlobalOffset[2] << "}\n";

      if (NDR.Dims >= 1 && NDR.GlobalSize[0] > 1) {
        size_t N = NDR.GlobalSize[0];

        size_t Parts   = 4;   // 总共切 4 段
        size_t PartIdx = 2;   // 要第 3 段

        size_t base_chunk = N / Parts;
        size_t rem        = N % Parts;

        size_t begin = PartIdx * base_chunk;
        size_t end   = (PartIdx == Parts - 1) ? (N) : (begin + base_chunk);
        size_t len   = end - begin;

        NDR.GlobalSize[0]   = len;
        NDR.GlobalOffset[0] = NDR.GlobalOffset[0] + begin;  // 原来是 0 就等于 begin

        HANDLER_TRACE_STREAM << "=== handler === [K3] after split: "
                  << "rows [" << begin << ", " << end << ") of original range\n"
                  << "New GlobalSize[0] = " << NDR.GlobalSize[0]
                  << " New Offset[0] = " << NDR.GlobalOffset[0] << "\n";
      }
    }
    #endif
  }
#endif
// 【END】=======================================================
  


  // 单独对特殊情况的kernel处理 目前没有 按道理可以忽略 但必须在之前rebind 因为有调用
  // **注意** 因目前没有 忽略这段代码中对MQueue的操作 在此前MQueue为空未rebind
  const auto &type = getType();
  if (type == detail::CG::Kernel) {
    // If there were uses of set_specialization_constant build the kernel_bundle
    std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpPtr =
        getOrInsertHandlerKernelBundle(/*Insert=*/false);
    // 目前没有遇到kernel_bundle的情况
    // kernel_bundle主要用于用户控制kernel的编译和链接
    if (KernelBundleImpPtr) {
      #ifdef PRINT_TRACE
      HANDLER_TRACE_STREAM << "======handler.cpp KernelBundleImpPtr" << std::endl;
      #endif
      // Make sure implicit non-interop kernel bundles have the kernel
      if (!KernelBundleImpPtr->isInterop() &&
          !MImpl->isStateExplicitKernelBundle()) {
        kernel_id KernelID =
            detail::ProgramManager::getInstance().getSYCLKernelID(MKernelName);
        bool KernelInserted =
            KernelBundleImpPtr->add_kernel(KernelID, MQueue->get_device());
        // If kernel was not inserted and the bundle is in input mode we try
        // building it and trying to find the kernel in executable mode
        if (!KernelInserted &&
            KernelBundleImpPtr->get_bundle_state() == bundle_state::input) {
          auto KernelBundle =
              detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
                  KernelBundleImpPtr);
          kernel_bundle<bundle_state::executable> ExecKernelBundle =
              build(KernelBundle);
          KernelBundleImpPtr = detail::getSyclObjImpl(ExecKernelBundle);
          setHandlerKernelBundle(KernelBundleImpPtr);
          KernelInserted =
              KernelBundleImpPtr->add_kernel(KernelID, MQueue->get_device());
        }
        // If the kernel was not found in executable mode we throw an exception
        if (!KernelInserted)
          throw sycl::exception(make_error_code(errc::runtime),
                                "Failed to add kernel to kernel bundle.");
      }

      switch (KernelBundleImpPtr->get_bundle_state()) {
      case bundle_state::input: {
        // Underlying level expects kernel_bundle to be in executable state
        kernel_bundle<bundle_state::executable> ExecBundle = build(
            detail::createSyclObjFromImpl<kernel_bundle<bundle_state::input>>(
                KernelBundleImpPtr));
        KernelBundleImpPtr = detail::getSyclObjImpl(ExecBundle);
        setHandlerKernelBundle(KernelBundleImpPtr);
        break;
      }
      case bundle_state::executable:
        // Nothing to do
        break;
      case bundle_state::object:
        assert(0 && "Expected that the bundle is either in input or executable "
                    "states.");
        break;
      }
    }

    // 目前也没有这种快速kernel的情况
    // 无需求 无依赖 无流 快速路径提交kernel执行
    if (!MQueue->is_in_fusion_mode() &&
        MRequirements.size() + MEvents.size() + MStreamStorage.size() == 0) {
      // if user does not add a new dependency to the dependency graph, i.e.
      // the graph is not changed, and the queue is not in fusion mode, then
      // this faster path is used to submit kernel bypassing scheduler and
      // avoiding CommandGroup, Command objects creation.

      // #ifdef PRINT_TRACE
      // std::cout << "======handler.cpp size=0" << std::endl;
      // #endif

      std::vector<RT::PiEvent> RawEvents;
      detail::EventImplPtr NewEvent;
      RT::PiEvent *OutEvent = nullptr;

      auto EnqueueKernel = [&]() {
        // 'Result' for single point of return
        pi_int32 Result = PI_ERROR_INVALID_VALUE;

        if (MQueue->is_host()) {
          MHostKernel->call(MNDRDesc, (NewEvent)
                                          ? NewEvent->getHostProfilingInfo()
                                          : nullptr);
          Result = PI_SUCCESS;
        } else {
          if (MQueue->getPlugin().getBackend() ==
              backend::ext_intel_esimd_emulator) {
            MQueue->getPlugin().call<detail::PiApiKind::piEnqueueKernelLaunch>(
                nullptr, reinterpret_cast<pi_kernel>(MHostKernel->getPtr()),
                MNDRDesc.Dims, &MNDRDesc.GlobalOffset[0],
                &MNDRDesc.GlobalSize[0], &MNDRDesc.LocalSize[0], 0, nullptr,
                nullptr);
            Result = PI_SUCCESS;
          } else {
            // #ifdef PRINT_TRACE
            // std::cout << "handler.cpp -> enqueueImpKernel" << std::endl;
            // #endif
            Result = enqueueImpKernel(MQueue, MNDRDesc, MArgs,
                                      KernelBundleImpPtr, MKernel, MKernelName,
                                      MOSModuleHandle, RawEvents, OutEvent,
                                      nullptr, MImpl->MKernelCacheConfig);
          }
        }
        return Result;
      };

      bool DiscardEvent = false;
      if (MQueue->has_discard_events_support()) {
        // Kernel only uses assert if it's non interop one
        bool KernelUsesAssert =
            !(MKernel && MKernel->isInterop()) &&
            detail::ProgramManager::getInstance().kernelUsesAssert(
                MOSModuleHandle, MKernelName);
        DiscardEvent = !KernelUsesAssert;
      }

      if (DiscardEvent) {
        #ifdef PRINT_TRACE
        HANDLER_TRACE_STREAM << "======handler.cpp DiscardEvent" << std::endl;
        #endif
        if (PI_SUCCESS != EnqueueKernel())
          throw runtime_error("Enqueue process failed.",
                              PI_ERROR_INVALID_OPERATION);
      } else {
        // #ifdef PRINT_TRACE
        // std::cout << "======handler.cpp No DiscardEvent" << std::endl;
        // #endif
        NewEvent = std::make_shared<detail::event_impl>(MQueue);
        NewEvent->setContextImpl(MQueue->getContextImplPtr());
        NewEvent->setStateIncomplete();
        OutEvent = &NewEvent->getHandleRef();

        NewEvent->setSubmissionTime();

        if (PI_SUCCESS != EnqueueKernel())
          throw runtime_error("Enqueue process failed.",
                              PI_ERROR_INVALID_OPERATION);
        else if (NewEvent->is_host() || NewEvent->getHandleRef() == nullptr)
          NewEvent->setComplete();

        MLastEvent = detail::createSyclObjFromImpl<event>(NewEvent);
      }
      return MLastEvent;
    }
  }

  // **注意** 只有cmdType==detail::CG::Kernel会参与MQueue的rebind
  std::unique_ptr<detail::CG> CommandGroup;
  switch (type) {
  case detail::CG::Kernel:
  case detail::CG::RunOnHostIntel: {
    // Copy kernel name here instead of move so that it's available after
    // running of this method by reductions implementation. This allows for
    // assert feature to check if kernel uses assertions
    CommandGroup.reset(new detail::CGExecKernel(
        std::move(MNDRDesc), std::move(MHostKernel), std::move(MKernel),
        std::move(MImpl->MKernelBundle), std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), std::move(MArgs),
        MKernelName, MOSModuleHandle, std::move(MStreamStorage),
        std::move(MImpl->MAuxiliaryResources), MCGType,
        MImpl->MKernelCacheConfig, MCodeLoc));
    break;
  }
  case detail::CG::CodeplayInteropTask:
    CommandGroup.reset(new detail::CGInteropTask(
        std::move(MInteropTask), std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::CopyAccToPtr:
  case detail::CG::CopyPtrToAcc:
  case detail::CG::CopyAccToAcc:
    CommandGroup.reset(new detail::CGCopy(
        MCGType, MSrcPtr, MDstPtr, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::Fill:
    CommandGroup.reset(new detail::CGFill(
        std::move(MPattern), MDstPtr, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::UpdateHost:
    CommandGroup.reset(new detail::CGUpdateHost(
        MDstPtr, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::CopyUSM:
    CommandGroup.reset(new detail::CGCopyUSM(
        MSrcPtr, MDstPtr, MLength, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::FillUSM:
    CommandGroup.reset(new detail::CGFillUSM(
        std::move(MPattern), MDstPtr, MLength, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::PrefetchUSM:
    CommandGroup.reset(new detail::CGPrefetchUSM(
        MDstPtr, MLength, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::AdviseUSM:
    CommandGroup.reset(new detail::CGAdviseUSM(
        MDstPtr, MLength, MImpl->MAdvice, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::Copy2DUSM:
    CommandGroup.reset(new detail::CGCopy2DUSM(
        MSrcPtr, MDstPtr, MImpl->MSrcPitch, MImpl->MDstPitch, MImpl->MWidth,
        MImpl->MHeight, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::Fill2DUSM:
    CommandGroup.reset(new detail::CGFill2DUSM(
        std::move(MPattern), MDstPtr, MImpl->MDstPitch, MImpl->MWidth,
        MImpl->MHeight, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::Memset2DUSM:
    CommandGroup.reset(new detail::CGMemset2DUSM(
        MPattern[0], MDstPtr, MImpl->MDstPitch, MImpl->MWidth, MImpl->MHeight,
        std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::CodeplayHostTask:
    CommandGroup.reset(new detail::CGHostTask(
        std::move(MHostTask), MQueue, MQueue->getContextImplPtr(),
        std::move(MArgs), std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::Barrier:
  case detail::CG::BarrierWaitlist:
    CommandGroup.reset(new detail::CGBarrier(
        std::move(MEventsWaitWithBarrier), std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::CopyToDeviceGlobal: {
    CommandGroup.reset(new detail::CGCopyToDeviceGlobal(
        MSrcPtr, MDstPtr, MImpl->MIsDeviceImageScoped, MLength, MImpl->MOffset,
        std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MOSModuleHandle, MCodeLoc));
    break;
  }
  case detail::CG::CopyFromDeviceGlobal: {
    CommandGroup.reset(new detail::CGCopyFromDeviceGlobal(
        MSrcPtr, MDstPtr, MImpl->MIsDeviceImageScoped, MLength, MImpl->MOffset,
        std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MOSModuleHandle, MCodeLoc));
    break;
  }
  case detail::CG::None:
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL)) {
      HANDLER_TRACE_STREAM << "WARNING: An empty command group is submitted." << std::endl;
    }
    detail::EventImplPtr Event = std::make_shared<sycl::detail::event_impl>();
    MLastEvent = detail::createSyclObjFromImpl<event>(Event);
    return MLastEvent;
  }

  if (!CommandGroup) {
    throw sycl::runtime_error(
        "Internal Error. Command group cannot be constructed.",
        PI_ERROR_INVALID_OPERATION);
  }

  #ifdef PRINT_TRACE
  HANDLER_TRACE_STREAM << "======handler.cpp === type: " << type << " req: " << MRequirements.size() << " queue: " << MQueue << " event: " << MEvents.size() << " lastevent: " << &MLastEvent << std::endl;
  #endif



// 【START】=======================================================
#if defined(SCHEDULE_OFFLINE) || defined(SNMD_OFFLINE)
  // 只CommandGroup和MQueue是合理的 MRequirements和其他已被move进CommandGroup
  auto &PM = detail::ProgramManager::getInstance();
#ifndef SCHEDULE_OFFLINE
  PM.kernel_count++;
#endif
  SyclKernelCg *sycl_kernel_cg =
      new SyclKernelCg(PM.kernel_count, std::move(CommandGroup), MQueue);
  PM.kernel_cgs.push_back(sycl_kernel_cg);
  return MLastEvent;
#else
  detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), std::move(MQueue));

  MLastEvent = detail::createSyclObjFromImpl<event>(Event);

  #ifdef PRINT_TRACE
  HANDLER_TRACE_STREAM << "======handler.cpp after === queue: " << MQueue << " event: " << MEvents.size() << " lastevent: " << &MLastEvent << std::endl;
  #endif

  return MLastEvent;
#endif
// 【END】=======================================================
}

#ifdef SNMD_OFFLINE
event handler::resubmit(detail::SyclKernelCg &sycl_kernel_cg) {
  using namespace sycl::detail;
  auto &PM = detail::ProgramManager::getInstance();

  if (!sycl_kernel_cg.kernel_cg || !sycl_kernel_cg.kernel_queue) {
    throw sycl::runtime_error(
        "Internal Error. Offline kernel CG or queue is null.",
        PI_ERROR_INVALID_OPERATION);
  }

  auto *ExecCG = dynamic_cast<detail::CGExecKernel *>(sycl_kernel_cg.kernel_cg.get());
  if (!ExecCG) {
    throw sycl::runtime_error(
        "Internal Error. Expected CGExecKernel for offline resubmit.",
        PI_ERROR_INVALID_OPERATION);
  }

  detail::QueueImplPtr KernelQueue = sycl_kernel_cg.kernel_queue;
  std::vector<Requirement *> &KernelReqs = ExecCG->MRequirements;
  size_t &NumParts = PM.NumParts;
  std::vector<int> &SplitDevices = PM.SplitDevices;

  // SPLIT
  if (NumParts > 1) {
    std::vector<int> ValidSplitDevices;
    for (int DeviceIndex : SplitDevices) {
      if (DeviceIndex <= 0 ||
          DeviceIndex >= static_cast<int>(PM.globalDevices.size())) {
        HANDLER_TRACE_STREAM << "=== handler === Split device_index out of range: "
                  << DeviceIndex << std::endl;
        continue;
      }
      if (std::find(ValidSplitDevices.begin(), ValidSplitDevices.end(),
                    DeviceIndex) == ValidSplitDevices.end()) {
        ValidSplitDevices.push_back(DeviceIndex);
      }
    }
    if (ValidSplitDevices.size() < NumParts) {
      HANDLER_TRACE_STREAM << "=== handler === Split NumParts clamped from "
                << NumParts << " to " << ValidSplitDevices.size()
                << std::endl;
      NumParts = ValidSplitDevices.size();
    }
    SplitDevices = std::move(ValidSplitDevices);
    if (NumParts <= 1 || SplitDevices.size() <= 1) {
      NumParts = 1;
      SplitDevices.clear();
    }
  }

  if (NumParts > 1) {
    const size_t SplitDim0 = ExecCG->MNDRDesc.GlobalSize[0];
    size_t AdjustedParts = NumParts;
    while (AdjustedParts > 1 &&
           (AdjustedParts % 2 != 0 || SplitDim0 < AdjustedParts ||
            SplitDim0 % AdjustedParts != 0)) {
      --AdjustedParts;
    }

    if (AdjustedParts != NumParts) {
      HANDLER_TRACE_STREAM << "=== handler === Split NumParts adjusted from "
                << NumParts << " to " << AdjustedParts
                << " for global_size0: " << SplitDim0 << std::endl;
      NumParts = AdjustedParts;
      if (SplitDevices.size() > NumParts) {
        SplitDevices.resize(NumParts);
      }
    }

    if (NumParts <= 1 || SplitDevices.size() <= 1) {
      NumParts = 1;
      SplitDevices.clear();
    }
  }

  if (NumParts > 1 &&
      !offlineSplitCanUseDim0ContiguousWrites(ExecCG, NumParts)) {
    HANDLER_TRACE_STREAM << "=== handler === Split NumParts disabled for "
                         << "non-contiguous or indivisible write range"
                         << std::endl;
    NumParts = 1;
    SplitDevices.clear();
  }

  if (NumParts > 1) {
#ifdef SNMD_OFFLINE_SPLIT_STATS
    offlineSplitStats().SplitKernelCount++;
#endif
    HANDLER_TRACE_STREAM << "=== handler === Split NumParts: " << NumParts << std::endl;

    // 1
    std::vector<detail::QueueImplPtr> &SplitQueues_Write = PM.SplitQueues_Write;
    SplitQueues_Write.clear();
    for (size_t p = 0; p < NumParts; p++) {
      int SplitDeviceIndex = SplitDevices[p];
      device SplitDevice = PM.globalDevices.at(SplitDeviceIndex);
      detail::DeviceImplPtr SplitDP = detail::getSyclObjImpl(SplitDevice);
      std::shared_ptr<detail::queue_impl> SplitQueue =
          makeOfflineProfilingQueue(SplitDP, KernelQueue);
      SplitQueues_Write.push_back(SplitQueue);
      HANDLER_TRACE_STREAM << "=== handler === Split part " << p
                << " uses device_index: " << SplitDeviceIndex << std::endl;
    }
    detail::QueueImplPtr hostQ = Scheduler::getInstance().getDefaultHostQueue();
    auto hostCtx = hostQ->getContextImplPtr();
    HANDLER_TRACE_STREAM << "=== handler === Split step1 hostQ: " << hostQ << " hostCtx: " << hostCtx << std::endl;
    PendingOfflineSplitMerge PendingSplit;
    PendingSplit.KernelCount = sycl_kernel_cg.kernel_count;
    PendingSplit.HostQueue = hostQ;
    PendingSplit.HostContext = hostCtx;
    PendingSplit.SplitQueues = SplitQueues_Write;

    // 2
    std::vector<Requirement *> SplitReqs_onlyRead;
    std::vector<Requirement *> SplitReqs_hasWrite;
    std::vector<std::vector<Requirement*>> SplitReqs_Copy;
    std::vector<std::unique_ptr<Requirement>> SplitReqOwners;
    SplitReqs_Copy.resize(NumParts);
    HANDLER_TRACE_STREAM << "=== handler === Split step2 SplitReqs_Copy resized to NumParts: " << NumParts << std::endl;

    // 3
    for (Requirement *Req : KernelReqs) {
      auto Mode = Req->MAccessMode;
      const bool onlyRead = (Mode == access::mode::read);
      const bool hasRead = (Mode == access::mode::read) ||
                            (Mode == access::mode::read_write) ||
                            (Mode == access::mode::atomic);

      bool isRecorded = detail::Scheduler::getInstance().getMemObjRecord(Req) != nullptr;
      MemObjRecord *ReqRecord = isRecorded ? detail::Scheduler::getInstance().getMemObjRecord(Req) : nullptr;
      ContextImplPtr ReqCurCtx = isRecorded ? detail::Scheduler::getInstance().getMemObjRecord(Req)->MCurContext : hostCtx;
      if (ReqCurCtx != hostCtx) {
        HANDLER_TRACE_STREAM << " === handler === Split step3 Req:" << Req << "->" << Req->MSYCLMemObj << " CurCtx not host\n";
      } else {
        HANDLER_TRACE_STREAM << " === handler === Split step3 Req:" << Req << "->" << Req->MSYCLMemObj << " CurCtx is host\n";
      }

      // 3.1
      if (hasRead) {
        // Full trailing dimensions form one contiguous row-major interval.
        // Describe such replication as 1-D so host/device copies use the
        // buffer-copy APIs rather than CUDA CopyRect.  The logical accessor
        // retained by the kernel is unchanged.
        std::unique_ptr<Requirement> LinearReadReqOwner =
            makeOfflineLinearRowBlockReq(Req, 0, Req->MAccessRange[0]);
        Requirement *TransferReq =
            LinearReadReqOwner ? LinearReadReqOwner.get() : Req;
        QueueImplPtr SrcQueue = hostQ;
        if (ReqCurCtx != hostCtx && ReqRecord != nullptr) {
          SrcQueue = nullptr;
          for (AllocaCommandBase *AllocaCmd : ReqRecord->MAllocaCommands) {
            if (AllocaCmd->getQueue() != nullptr && AllocaCmd->getQueue()->getContextImplPtr() == ReqCurCtx) {
              SrcQueue = AllocaCmd->getQueue();
              break;
            }
          }
          HANDLER_TRACE_STREAM << "=== handler === Split step3 SrcQueue: " << SrcQueue << std::endl;
        }

        for (const QueueImplPtr &SplitQueue : SplitQueues_Write) {
          if (onlyRead && pendingOfflineSplitHasReadReplica(
                              Req->MSYCLMemObj, SplitQueue)) {
#ifdef SNMD_OFFLINE_SPLIT_STATS
            offlineSplitStats().ReusedReadReplicaBytes +=
                offlineRequirementBytes(TransferReq);
#endif
            HANDLER_TRACE_STREAM
                << "=== handler === Split step3 reuse read-only replica\n";
            continue;
          }
          if (ReqCurCtx != hostCtx && SplitQueue->getContextImplPtr() == ReqCurCtx) {
            HANDLER_TRACE_STREAM << "=== handler === Split step3 SplitQueue is SrcQueue, continue\n";
            continue;
          }

          bool moved_by_p2p = false;
          if (ReqCurCtx != hostCtx && SrcQueue != nullptr) {
            try {
              EventImplPtr ev_p2p =
                  detail::Scheduler::getInstance().addMemoryMove(
                      TransferReq, SplitQueue, SrcQueue);
              waitOfflineSplitEvent(ev_p2p, OfflineSplitWaitKind::Prepare);
#ifdef SNMD_OFFLINE_SPLIT_STATS
              offlineSplitStats().InputDirectD2DBytes +=
                  offlineRequirementBytes(TransferReq);
#endif
              moved_by_p2p = true;
              HANDLER_TRACE_STREAM << "=== handler === Split step3 direct D2D success\n";
            } catch (const std::exception &e) {
              HANDLER_TRACE_STREAM << "=== handler === Split step3 direct D2D failed, fallback D2H->H2D, reason: " << e.what() << "\n";
            } catch (...) {
              HANDLER_TRACE_STREAM << "=== handler === Split step3 direct D2D failed, fallback D2H->H2D\n";
            }
          }

          if (!moved_by_p2p) {
            if (ReqCurCtx != hostCtx && SrcQueue != nullptr) {
              EventImplPtr ev_host =
                  detail::Scheduler::getInstance().addMemoryMove(
                      TransferReq, hostQ, SrcQueue);
              waitOfflineSplitEvent(ev_host, OfflineSplitWaitKind::Prepare);
#ifdef SNMD_OFFLINE_SPLIT_STATS
              offlineSplitStats().InputD2HBytes +=
                  offlineRequirementBytes(TransferReq);
#endif
              HANDLER_TRACE_STREAM << "=== handler === Split step3 copy back host\n";
            }

            EventImplPtr ev_split =
                detail::Scheduler::getInstance().addMemoryMove(
                    TransferReq, SplitQueue, hostQ);
            waitOfflineSplitEvent(ev_split, OfflineSplitWaitKind::Prepare);
#ifdef SNMD_OFFLINE_SPLIT_STATS
            offlineSplitStats().InputH2DBytes +=
                offlineRequirementBytes(TransferReq);
#endif
            HANDLER_TRACE_STREAM << "=== handler === Split step3 copy to split device\n";
          }
        }
      } else {
        std::vector<Command *> ToEnqueue;
        MemObjRecord *SplitRecord = ReqRecord;
        if (SplitRecord == nullptr)
          SplitRecord = detail::Scheduler::getInstance().MGraphBuilder.getOrInsertMemObjRecord(hostQ, Req, ToEnqueue);

        for (const QueueImplPtr &SplitQueue : SplitQueues_Write) {
          detail::Scheduler::getInstance().MGraphBuilder.getOrCreateAllocaForSplitReq(SplitRecord, Req, SplitQueue, ToEnqueue);
        }
      }

      // 3.2
      if (onlyRead) {
        SplitReqs_onlyRead.push_back(Req);
        rememberOfflineSplitRead(PendingSplit, Req->MSYCLMemObj);
      } else {
        SplitReqs_hasWrite.push_back(Req);
        rememberOfflineSplitWrite(PendingSplit, Req->MSYCLMemObj);
        range<3> AccessRange = Req->MAccessRange;
        size_t dim0 = AccessRange[0];
        size_t chunk = dim0 / NumParts;
        for (size_t p = 0; p < NumParts; p++) {
          size_t begin0 = p * chunk;
          size_t end0 = (p + 1 == NumParts) ? (dim0) : (begin0 + chunk);
          size_t part0 = end0 - begin0;
          HANDLER_TRACE_STREAM << "=== handler === Split step3 Write part "
                    << p << " begin: " << begin0 << " end: " << end0
                    << " range: " << part0 << "," << AccessRange[1]
                    << "," << AccessRange[2] << "\n";

          auto CopyReqOwner =
              makeOfflineLinearRowBlockReq(Req, begin0, part0);
          if (!CopyReqOwner) {
            throw sycl::runtime_error(
                "Internal Error. Offline split write block is not contiguous.",
                PI_ERROR_INVALID_VALUE);
          }
          Requirement *CopyReq = CopyReqOwner.get();
          SplitReqs_Copy[p].push_back(CopyReq);
          SplitReqOwners.push_back(std::move(CopyReqOwner));
        }
      }
    }
    HANDLER_TRACE_STREAM << "=== handler === Split step3 onlyRead: " << SplitReqs_onlyRead.size() << " hasWrite: " << SplitReqs_hasWrite.size() << std::endl;

    // 4
    detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(std::move(sycl_kernel_cg.kernel_cg), std::move(KernelQueue));
    PendingSplit.Event = Event;
    PendingSplit.Events = PM.SplitEvents;
    if (PendingSplit.Events.empty() && Event) {
      PendingSplit.Events.push_back(Event);
    }
    PendingSplit.SplitReqsCopy = std::move(SplitReqs_Copy);
    PendingSplit.SplitReqOwners = std::move(SplitReqOwners);
    cacheOfflineSplitReadReplicas(PendingSplit);
    pendingOfflineSplitMerges().push_back(std::move(PendingSplit));
    HANDLER_TRACE_STREAM << "=== handler === Split submitted async, merge deferred for kernel_count: "
              << sycl_kernel_cg.kernel_count << std::endl;

    HANDLER_TRACE_STREAM << getpid() << " === handler === resubmit kernel: " << sycl_kernel_cg.kernel_count << std::endl;
    event MLastEvent = detail::createSyclObjFromImpl<event>(Event);
    return MLastEvent;
  }
  else {
    NumParts = 1;
#ifdef SNMD_OFFLINE_SPLIT_STATS
    offlineSplitStats().SingleKernelCount++;
#endif
    HANDLER_TRACE_STREAM << getpid() << " === handler === resubmit kernel: " << sycl_kernel_cg.kernel_count << std::endl;

    // A non-split read can consume the already prepared replica in its target
    // context. This does not skip split-to-split version preparation below;
    // it only prevents the ordinary scheduler from attempting an unsupported
    // cross-context D2D move for an unchanged read-only object.
    for (Requirement *Req : KernelReqs) {
      if (Req == nullptr || Req->MAccessMode != access::mode::read ||
          !pendingOfflineSplitHasReadReplica(Req->MSYCLMemObj, KernelQueue)) {
        continue;
      }
      if (MemObjRecord *Record =
              detail::Scheduler::getInstance().getMemObjRecord(Req)) {
        Record->MCurContext = KernelQueue->getContextImplPtr();
        HANDLER_TRACE_STREAM
            << "=== handler === reuse pending read replica for non-split "
            << "kernel\n";
      }
    }

#if !defined(SCHEDULE_OFFLINE)
    std::shared_ptr<detail::queue_impl> &kernel_queue = sycl_kernel_cg.kernel_queue;
    device exec_device = PM.globalDevices.at(1);
    detail::DeviceImplPtr dp = detail::getSyclObjImpl(exec_device);
    HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === rebind_device is_gpu: " << exec_device.is_gpu() << std::endl;
    kernel_queue = makeOfflineProfilingQueue(dp, kernel_queue);
    HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === rebind MQueue" << std::endl;
#endif

    detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(std::move(sycl_kernel_cg.kernel_cg), std::move(sycl_kernel_cg.kernel_queue));
    event MLastEvent = detail::createSyclObjFromImpl<event>(Event);
    return MLastEvent;
  }
}

#if !defined(SCHEDULE_OFFLINE)
// 测试DataParallel与OfflineKernel存储机制结合
// event::wait()调用此函数
event handler::scheduleOffline() {
  std::vector<detail::SyclKernelCg *> &kernel_cgs = detail::ProgramManager::getInstance().kernel_cgs;
  HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " scheduleOffline kernel_cgs.size: " << kernel_cgs.size() << std::endl;
  if (kernel_cgs.empty()) {
    return event{};
  }
  std::vector<OfflineProfileEvent> profile_events;
  event last_event;
  for (int i = 0; i < kernel_cgs.size(); ++i) {
    detail::SyclKernelCg *sycl_kernel_cg = kernel_cgs.at(i);
#ifdef SNMD_OFFLINE
    finalizePendingOfflineSplitsForKernel(sycl_kernel_cg);
#endif
    uint64_t HostStart = offlineNowNs();
    last_event = resubmit(*sycl_kernel_cg);
    uint64_t HostEnd = offlineNowNs();
    HANDLER_TRACE_STREAM << "=== handler === Offline submit kernel_count: "
                         << sycl_kernel_cg->kernel_count
                         << " host_duration_ns: "
                         << (HostEnd >= HostStart ? HostEnd - HostStart : 0)
                         << std::endl;
    profile_events.push_back(
        {sycl_kernel_cg->kernel_count, 1,
         offlineSubmittedNumParts(sycl_kernel_cg->kernel_count),
         HostStart, HostEnd, last_event,
         findOfflineKernelProfileKey(sycl_kernel_cg->kernel_count)});
  }
#ifdef SNMD_OFFLINE
  finalizeAllPendingOfflineSplits();
  applyOfflineSplitFinalizeTimes(profile_events);
#endif
  processOfflineProfilingBatch(detail::ProgramManager::getInstance().wait_count,
                               profile_events);
  clearOfflineBatch();
  return last_event;
}
#endif
#endif


// 【START】=======================================================
#ifdef SCHEDULE_OFFLINE
#if !defined(SNMD_OFFLINE)
event handler::resubmit(detail::SyclKernelCg &sycl_kernel_cg) {
  HANDLER_TRACE_STREAM << getpid() << " === handler === resubmit kernel: " << sycl_kernel_cg.kernel_count << std::endl;
  detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
      std::move(sycl_kernel_cg.kernel_cg), std::move(sycl_kernel_cg.kernel_queue));
  event MLastEvent = detail::createSyclObjFromImpl<event>(Event);
  return MLastEvent;
}
#endif

// **注意** 由event::wait()调用
// 所有与daemon通信都由handler完成
// 返回最后一个kernel的对应event给调用者event
// 无法跳过的串行代码视作冷启动开销的一部分 暂时不考虑
event handler::scheduleOffline() {
  using namespace sycl::detail;
  // **注意** 一组kernel只调用一次
  if (detail::ProgramManager::getInstance().kernel_cgs.empty()) {
    HANDLER_TRACE_STREAM << "=== handler === Process " << getpid()
              << " === scheduleOffline empty batch" << std::endl;
    return event{};
  }

  // 【流程】
  // master接收一组kernel 确定需要扩容
  // master通知其他scale 起对应daemom
  // daemon起对应syclapp 同时接收master的execinfo
  // syclapp的handler到达第一个wait 调用此函数
  // 调用此函数时 handler已知会
  // 1.一组kernel都不执行 跳过此wait
  // 2.一组kernel的执行信息已知 不需要重新收集kenrel信息 只需要跟随execinfo
  // 3.一组kernel的执行信息未知 需要重新收集kernel信息 即通用情况

  // **注意** online如何确定启动的syclapp是否为scale
  // 解释: daemon通知scale时传递scalecount
  //   在handler的通用流程的第一次与daemon通信时传输scalecount
  //   handler随即跳过第一次通信（必然为第一个kernel 不会扩容）
  //   handler跳过后续scalecount前所有kernel 在scalecount走入scale流程

  // 这里拿kernel_count没啥用
  int &daemon_wait_count = detail::ProgramManager::getInstance().wait_count;
  daemon_wait_count++;
  int &daemon_scale_count = detail::ProgramManager::getInstance().scale_count;
  std::vector<OfflineProfileEvent> profile_events;
  event last_event;

  HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === daemon_wait_count: " << daemon_wait_count << " daemon_scale_count: " << daemon_scale_count << std::endl;

  // 除了第一次扩容后的的后续扩容 即跳过前几个wait
  if (daemon_wait_count < daemon_scale_count) {
    HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === wait_count: " << daemon_wait_count << " skip first wait" << std::endl;
    event empty;
    clearOfflineBatch();
    return empty;
  }
  else if (daemon_wait_count == daemon_scale_count) {
    HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === wait_count: " << daemon_wait_count << " first wait" << std::endl;

    // TODO 要在这里从scale的daemon接收D2S 没有mq_recv
    // 先只考虑初始扩容
    // std::vector<D2SKernelExecInfo> &kernel_exec_infos = detail::ProgramManager::getInstance().kernel_scale_exec_infos;
    std::vector<D2SKernelExecInfo> kernel_exec_infos;
    {
      std::string received_data = receiveOfflineMqPayload(
          mq_id_program, MAX_MSG_PROGRAM_SIZE, "scale kernel_exec_infos");
      HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === scale mq_receive kernel_exec_infos" << std::endl;

      kernel_exec_infos = parseOfflineKernelExecInfos(received_data);
      if (kernel_exec_infos.empty()) {
        std::cerr << "Error: Process " << getpid()
                  << " received empty scale kernel_exec_infos" << std::endl;
        exit(1);
      }
      for (D2SKernelExecInfo &kernel_exec_info : kernel_exec_infos) {
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid()
                  << " === scale mq_receive kernel_exec_info === kernel_count: " << kernel_exec_info.kernel_count
                  << " exec: " << kernel_exec_info.exec
                  << " device_index: " << kernel_exec_info.device_index
                  << " num_parts: " << kernel_exec_info.num_parts
                  << " split_devices:";
        for (int DeviceIndex : kernel_exec_info.split_devices) {
          HANDLER_TRACE_STREAM << " " << DeviceIndex;
        }
        HANDLER_TRACE_STREAM
                  << " req_size: " << kernel_exec_info.req_counts.size() << std::endl;
      }
    }
    
    // return commDepend(kernel_exec_infos);
    // DONE ====【按kernel执行顺序 为每个kernel处理满足依赖 -> rebind -> resubmit】
    // 即使scale这一组kernel需要前一组kernel的数据 不需要单独的流程满足
    std::vector<detail::SyclKernelCg *> &kernel_cgs = detail::ProgramManager::getInstance().kernel_cgs;
    for (int exec_num = 0; exec_num < kernel_exec_infos.size(); exec_num++) {
      HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === scale commDepend exec_num: " << exec_num << std::endl;
      D2SKernelExecInfo &kernel_exec_info = kernel_exec_infos.at(exec_num);
      int kernel_count = kernel_exec_info.kernel_count;
      detail::SyclKernelCg *sycl_kernel_cg =
          findOfflineKernelCg(kernel_cgs, kernel_count);
#ifdef SNMD_OFFLINE
      finalizePendingOfflineSplitsForKernel(sycl_kernel_cg);
#endif

      // **注意** 满足依赖的逻辑仍与online一致
      // 不需要与daemon建立连接 因daemon对所有kernel的依赖都已知
      // 在一个kernel执行rebind前满足依赖
      // OPTI 并非前置kernel结束后就发给各个rank 因为可能新rank还没启动 暂时不考虑
      {
        auto &req_counts = kernel_exec_info.req_counts;
        if (!kernel_exec_info.exec) {
          if (req_counts.size() > 0) {
            for (int i = 0; i < sycl_kernel_cg->kernel_cg->MRequirements.size(); i++) {
              HANDLER_TRACE_STREAM << getpid() << " === handler === scale kernel_count: " << kernel_count << " req_counts.size(): " << req_counts.size();
              for (int req_count : req_counts) {
                HANDLER_TRACE_STREAM << " " << req_count;
              }
              HANDLER_TRACE_STREAM << std::endl;

              int daemon_req_count = i + 1;
              Requirement *Req = sycl_kernel_cg->kernel_cg->MRequirements[i];
              if ((std::find(req_counts.begin(), req_counts.end(), daemon_req_count) != req_counts.end()) && (Req->MAccessMode == access::mode::read || Req->MAccessMode == access::mode::read_write || Req->MAccessMode == access::mode::atomic)) {
                Requirement *hostReq = new Requirement(*Req);
                EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
                hostEvent->wait(hostEvent);
                delete hostReq;
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== scale sender add host acc" << std::endl;

                using DATA_TYPE = std::byte;
                size_t elem_size = Req->MElemSize;
                size_t buff_size = Req->MMemoryRange.size();
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== scale sender elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;

                SYCLMemObjI *MemObj = Req->MSYCLMemObj;
                SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
                void *UserPtr = BufferObj->getUserPtr();
                DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);

                SharedMemoryHandle handle = initSharedMemory(getpid(), kernel_count, daemon_req_count, elem_size * buff_size);
                writeToSharedMemory(handle, DataPtr, elem_size * buff_size);
                HANDLER_TRACE_STREAM << getpid() << " === handler === scale send host data" << std::endl;

                waitForReadCompletion(handle);
                cleanupSharedMemory(handle, elem_size * buff_size);
                HANDLER_TRACE_STREAM << getpid() << " === handler === scale waitForReadCompletion" << std::endl;
              }
            }
          }
          HANDLER_TRACE_STREAM << getpid() << " === handler === scale kernel_count: " << kernel_count << " end hostacc" << std::endl;
        }
        else {
          if (req_counts.size() > 0) {
            for (int i = 0; i < sycl_kernel_cg->kernel_cg->MRequirements.size(); i++) {
              int daemon_req_count = i + 1;
              Requirement *Req = sycl_kernel_cg->kernel_cg->MRequirements[i];
              if ((std::find(req_counts.begin(), req_counts.end(), daemon_req_count) != req_counts.end()) && (Req->MAccessMode == access::mode::read || Req->MAccessMode == access::mode::read_write || Req->MAccessMode == access::mode::atomic)) {
                Requirement *hostReq = new Requirement(*Req);
                EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
                hostEvent->wait(hostEvent);
                delete hostReq;
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== receiver add host acc" << std::endl;

                using DATA_TYPE = std::byte;
                size_t elem_size = Req->MElemSize;
                size_t buff_size = Req->MMemoryRange.size();
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== receiver elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;

                SYCLMemObjI *MemObj = Req->MSYCLMemObj;
                SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
                void *UserPtr = BufferObj->getUserPtr();
                DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);

                std::vector<DATA_TYPE> host_data(elem_size * buff_size);
                SharedMemoryHandle handle = initSharedMemory(getpid(), kernel_count, daemon_req_count, elem_size * buff_size);
                readFromSharedMemory(handle, host_data.data(), elem_size * buff_size);
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== Data read successfully." << std::endl;
                cleanupSharedMemory(handle, elem_size * buff_size);
                std::memcpy(DataPtr, host_data.data(), elem_size * buff_size);

                HANDLER_TRACE_STREAM << getpid() << " === handler === mem copy" << std::endl;
              }
            }
          }
        }
      }
      
      // **注意** MQueue在此才rebind 之前有暂未出现过的逻辑使用MQueue
      // 此时有刚启动的daemon
      if (kernel_exec_info.exec) {
        const int ActualDeviceIndex =
            clampOfflineDeviceIndex(kernel_exec_info.device_index);
#ifdef SNMD_OFFLINE
        applyOfflineSplitDecision(kernel_exec_info, ActualDeviceIndex);
#endif
        std::shared_ptr<detail::queue_impl> &kernel_queue = sycl_kernel_cg->kernel_queue;
        device exec_device = detail::ProgramManager::getInstance().globalDevices.at(ActualDeviceIndex);
        detail::DeviceImplPtr dp = detail::getSyclObjImpl(exec_device);
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === rebind_device is_gpu: " << exec_device.is_gpu() << std::endl;
        kernel_queue = makeOfflineProfilingQueue(dp, kernel_queue);

        // resubmit
        uint64_t HostStart = offlineNowNs();
        last_event = resubmit(*sycl_kernel_cg);
        uint64_t HostEnd = offlineNowNs();
        HANDLER_TRACE_STREAM << "=== handler === Offline submit kernel_count: "
                             << kernel_count << " host_duration_ns: "
                             << (HostEnd >= HostStart
                                     ? HostEnd - HostStart
                                     : 0)
                             << std::endl;
        int ProfileNumParts = 1;
#ifdef SNMD_OFFLINE
        ProfileNumParts = offlineSubmittedNumParts(kernel_count);
#endif
        profile_events.push_back({kernel_count, ActualDeviceIndex,
                                  std::max(1, ProfileNumParts), HostStart,
                                  HostEnd, last_event,
                                  findOfflineKernelProfileKey(kernel_count)});
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === resubmit kernel: " << kernel_count << std::endl;
      }
      else {
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === skip resubmit kernel: " << kernel_count << std::endl;
      }
    }
#ifdef SNMD_OFFLINE
    finalizeAllPendingOfflineSplits();
    applyOfflineSplitFinalizeTimes(profile_events);
#endif
    processOfflineProfilingBatch(daemon_wait_count, profile_events);
    clearOfflineBatch();
    return last_event;
  }
  //【通用情况】
  else {
    // DONE ====【发送每个kernel的reqs给daemon】
    std::vector<S2DKernelReqData> &kernel_req_datas = detail::ProgramManager::getInstance().kernel_reqs;
    // DEBUG用
    // for (S2DKernelReqData &kernel_req_data : kernel_req_datas) {
    //   std::cout << "=== handler === Process " << getpid() << " === kernel_req_data: " << kernel_req_data.serialize();
    // }
    {
      // **注意** 在最前面加上daemon_wait_count
      std::string serialized_data = std::to_string(daemon_wait_count) + "\n";
      for (const auto &kernel_req_data : kernel_req_datas) {
        serialized_data += kernel_req_data.serialize();
      }
      size_t message_size = serialized_data.size();

      sendOfflineMqPayload(mq_id_daemon, serialized_data,
                           MAX_MSG_DAEMON_SIZE, "kernel_req_datas");
      HANDLER_TRACE_STREAM << "=== handler === Process " << getpid()
                           << " === mq_send kernel_req_datas size: "
                           << kernel_req_datas.size()
                           << " bytes: " << message_size << std::endl;
    }

    // DONE ====【接收daemon对每个kernel的执行决策】
    std::vector<D2SKernelExecInfo> kernel_exec_infos;
    bool completion_driven_dispatch = false;
    bool completion_window_complete = false;
    bool completion_window_failed = false;
    {
      std::string received_data = receiveOfflineMqPayload(
          mq_id_program, MAX_MSG_PROGRAM_SIZE, "kernel_exec_infos");
      HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === mq_receive kernel_exec_infos" << std::endl;

      ParsedOfflineDispatchBatch parsed_dispatch =
          parseOfflineDispatchBatch(received_data);
      completion_driven_dispatch = parsed_dispatch.CompletionDriven;
      completion_window_complete = parsed_dispatch.WindowComplete;
      completion_window_failed = parsed_dispatch.WindowFailed;
      kernel_exec_infos = std::move(parsed_dispatch.KernelExecInfos);
      if (kernel_exec_infos.empty() && !completion_driven_dispatch) {
        std::cerr << "Error: Process " << getpid()
                  << " received empty kernel_exec_infos" << std::endl;
        exit(1);
      }
      for (D2SKernelExecInfo &kernel_exec_info : kernel_exec_infos) {
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid()
                  << " === mq_receive kernel_exec_info === kernel_count: " << kernel_exec_info.kernel_count
                  << " exec: " << kernel_exec_info.exec
                  << " device_index: " << kernel_exec_info.device_index
                  << " num_parts: " << kernel_exec_info.num_parts
                  << " split_devices:";
        for (int DeviceIndex : kernel_exec_info.split_devices) {
          HANDLER_TRACE_STREAM << " " << DeviceIndex;
        }
        HANDLER_TRACE_STREAM
                  << " req_size: " << kernel_exec_info.req_counts.size() << std::endl;
      }

      // **注意** 这里与online不同 不会有scaledevice 也不会返回跳过
      // offline的scale与通用流程有什么不一样 为什么online需要分开做handler流程
      // 解释: online的scale流程针对一个kernel 只有scale新起的handler执行 master要为这一个daemon服务
      //   offline的流程针对这一组kernel 所有的handler都要执行 逻辑相同

      // 如果info中有scale_count 说明此daemon是scale起的
      // 全局视图 此时通用流程的daemon不会接收到scale_count
      if (!kernel_exec_infos.empty() &&
          kernel_exec_infos.at(0).scale_count >= 1) {
        daemon_scale_count = kernel_exec_infos.at(0).scale_count;
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === kernel_exec_info scale_count: " << daemon_scale_count << std::endl;
      
        // 只属于被scale的daemon的流程
        // 如果wait_count与scale_count不同 说明不是从第一个wait开始scale 需要跳过第一个走skip分支流程
        //   直到相同 也需要重走通用流程 就不会接收到scale_count
        // 如果wait_count与scale_count相同（且一定是1）说明此wait是第一个 后续通用流程直接处理

        // 其他daemon: scale_count==0 不会执行
        // 被count==1时scale的daemon: scale_count==1时 不满足 不会执行
        // 被count>1时scale的daemon: 满足 返回跳过第一个
        if (daemon_wait_count != daemon_scale_count) {
          HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === wait_count: " << daemon_wait_count << " skip first wait" << std::endl;
          detail::ProgramManager::getInstance().kernel_scale_exec_infos = kernel_exec_infos;

          event empty;
          clearOfflineBatch();
          return empty;
        }
      }
    }

#ifdef SNMD_OFFLINE_COMPLETION_DRIVEN_QUEUE
    if (completion_driven_dispatch) {
      std::map<int, OfflineProfileEvent> in_flight;

      try {
        while (true) {
          if (completion_window_failed) {
            throw sycl::runtime_error(
                "Internal Error. Completion queue daemon aborted the active "
                "window.",
                PI_ERROR_INVALID_OPERATION);
          }
          for (D2SKernelExecInfo &kernel_exec_info : kernel_exec_infos) {
            if (!kernel_exec_info.exec ||
                !kernel_exec_info.req_counts.empty()) {
              throw sycl::runtime_error(
                  "Internal Error. Completion queue received a non-local "
                  "dispatch.",
                  PI_ERROR_INVALID_OPERATION);
            }

            const int kernel_count = kernel_exec_info.kernel_count;
            detail::SyclKernelCg *sycl_kernel_cg = findOfflineKernelCg(
                detail::ProgramManager::getInstance().kernel_cgs,
                kernel_count);
            const int ActualDeviceIndex =
                clampOfflineDeviceIndex(kernel_exec_info.device_index);

#ifdef SNMD_OFFLINE
            // Install the new decision before materializing an older Split. A
            // future partition-resident path can use both decisions to retain
            // compatible partitions; the current safe path still canonicalizes
            // any conflicting producer before this submission.
            applyOfflineSplitDecision(kernel_exec_info, ActualDeviceIndex);
            finalizePendingOfflineSplitsForKernel(sycl_kernel_cg);
#endif

            std::shared_ptr<detail::queue_impl> &kernel_queue =
                sycl_kernel_cg->kernel_queue;
            device exec_device =
                detail::ProgramManager::getInstance().globalDevices.at(
                    ActualDeviceIndex);
            detail::DeviceImplPtr dp = detail::getSyclObjImpl(exec_device);
            kernel_queue = makeOfflineProfilingQueue(dp, kernel_queue);

            const uint64_t HostStart = offlineNowNs();
            last_event = resubmit(*sycl_kernel_cg);
            const uint64_t HostEnd = offlineNowNs();
            const int ProfileNumParts =
                offlineSubmittedNumParts(kernel_count);
            OfflineProfileEvent profile_event{
                kernel_count, ActualDeviceIndex, ProfileNumParts,
                HostStart,   HostEnd,          last_event,
                findOfflineKernelProfileKey(kernel_count)};
            if (!in_flight.emplace(kernel_count, std::move(profile_event))
                     .second) {
              throw sycl::runtime_error(
                  "Internal Error. Completion queue dispatched a kernel "
                  "twice.",
                  PI_ERROR_INVALID_OPERATION);
            }
            HANDLER_TRACE_STREAM
                << "=== handler === CompletionQueue dispatched kernel_count: "
                << kernel_count << " device_index: " << ActualDeviceIndex
                << " num_parts: " << ProfileNumParts
                << " host_duration_ns: "
                << (HostEnd >= HostStart ? HostEnd - HostStart : 0)
                << std::endl;
          }

        if (completion_window_complete) {
          if (!in_flight.empty()) {
            throw sycl::runtime_error(
                "Internal Error. Completion queue closed with kernels in "
                "flight.",
                PI_ERROR_INVALID_OPERATION);
          }
#ifdef SNMD_OFFLINE
          finalizeAllPendingOfflineSplits();
#endif
          clearOfflineBatch();
          return last_event;
        }

        std::vector<int> completed_kernel_counts;
        while (completed_kernel_counts.empty()) {
          for (const auto &entry : in_flight) {
            if (offlineProfileEventComplete(entry.second)) {
              completed_kernel_counts.push_back(entry.first);
            }
          }
          if (completed_kernel_counts.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }

        S2DCompletionBatchData completion_batch;
        completion_batch.wait_count = daemon_wait_count;
        for (int kernel_count : completed_kernel_counts) {
          auto profile_it = in_flight.find(kernel_count);
          if (profile_it == in_flight.end()) {
            continue;
          }
          OfflineProfileEvent completed_event = profile_it->second;
          S2DKernelProfileData completion;
          if (completed_event.NumParts > 1) {
            completed_event.HostEndNs = offlineNowNs();
            markOfflineSplitPartsComplete(kernel_count,
                                          completed_event.HostEndNs);
            finalizeCompletedOfflineSplit(kernel_count);
            std::vector<OfflineProfileEvent> completed_split_profiles{
                completed_event};
            applyOfflineSplitFinalizeTimes(completed_split_profiles);
            if (!collectOfflineProfilingInfo(
                    daemon_wait_count, completed_split_profiles.front(),
                    completion)) {
              fillOfflineProfileData(daemon_wait_count,
                                     completed_split_profiles.front(), 0,
                                     completion);
            }
          } else {
            sycl::event CompletedEventCopy = completed_event.Event;
            CompletedEventCopy.wait();
            if (!collectOfflineProfilingInfo(
                    daemon_wait_count, completed_event, completion)) {
              fillOfflineProfileData(daemon_wait_count, completed_event, 0,
                                     completion);
            }
          }
          completion_batch.completions.push_back(std::move(completion));
          in_flight.erase(profile_it);
        }

        sendOfflineMqPayload(mq_id_daemon, completion_batch.serialize(),
                             MAX_MSG_DAEMON_SIZE,
                             "completion acknowledgement");

        const std::string next_payload = receiveOfflineMqPayload(
            mq_id_program, MAX_MSG_PROGRAM_SIZE,
            "completion dispatch batch");
        ParsedOfflineDispatchBatch next_dispatch =
            parseOfflineDispatchBatch(next_payload);
        if (!next_dispatch.CompletionDriven) {
          throw sycl::runtime_error(
              "Internal Error. Completion queue protocol downgraded inside "
              "a window.",
              PI_ERROR_INVALID_OPERATION);
        }
          completion_window_complete = next_dispatch.WindowComplete;
          completion_window_failed = next_dispatch.WindowFailed;
          kernel_exec_infos = std::move(next_dispatch.KernelExecInfos);
        }
      } catch (...) {
        // Tell the daemon to abandon its reservation state before propagating
        // the real SYCL exception to the user's wait(). If the daemon was the
        // side that failed, it has already left the protocol and needs no echo.
        if (!completion_window_failed) {
          S2DCompletionBatchData failure_batch;
          failure_batch.wait_count = daemon_wait_count;
          failure_batch.window_failed = true;
          sendOfflineMqPayload(mq_id_daemon, failure_batch.serialize(),
                               MAX_MSG_DAEMON_SIZE,
                               "completion failure acknowledgement");
        }
        throw;
      }
    }
#endif
    
    // return commDepend(kernel_exec_infos);
    // DONE ====【按kernel执行顺序 为每个kernel处理满足依赖 -> rebind -> resubmit】
    // 即使scale这一组kernel需要前一组kernel的数据 不需要单独的流程满足
    std::vector<detail::SyclKernelCg *> &kernel_cgs = detail::ProgramManager::getInstance().kernel_cgs;
    HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === kernel_exec_infos.size(): " << kernel_exec_infos.size() << std::endl;
    for (int exec_num = 0; exec_num < kernel_exec_infos.size(); exec_num++) {
      D2SKernelExecInfo &kernel_exec_info = kernel_exec_infos.at(exec_num);
      int kernel_count = kernel_exec_info.kernel_count;
      detail::SyclKernelCg *sycl_kernel_cg =
          findOfflineKernelCg(kernel_cgs, kernel_count);
      HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === kernel_count: " << kernel_count << std::endl;
#ifdef SNMD_OFFLINE
      finalizePendingOfflineSplitsForKernel(sycl_kernel_cg);
#endif

      // **注意** 满足依赖的逻辑仍与online一致
      // 不需要与daemon建立连接 因daemon对所有kernel的依赖都已知
      // 在一个kernel执行rebind前满足依赖
      // OPTI 并非前置kernel结束后就发给各个rank 因为可能新rank还没启动 暂时不考虑
      {
        auto &req_counts = kernel_exec_info.req_counts;
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === req_counts.size(): " << req_counts.size() << std::endl;
        for (int i = 0; i < req_counts.size(); i++) {
          HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === req_counts[" << i << "]: " << req_counts[i] << std::endl;
        }

        if (!kernel_exec_info.exec) {
          HANDLER_TRACE_STREAM << "=== NOEXEC ===" << std::endl;
          if (req_counts.size() > 0) {
            HANDLER_TRACE_STREAM << "=== NEED SEND SHMEM ===" << std::endl;
            for (int i = 0; i < sycl_kernel_cg->kernel_cg->MRequirements.size(); i++) {
              int daemon_req_count = i + 1;
              Requirement *Req = sycl_kernel_cg->kernel_cg->MRequirements[i];
              if ((std::find(req_counts.begin(), req_counts.end(), daemon_req_count) != req_counts.end()) && (Req->MAccessMode == access::mode::read || Req->MAccessMode == access::mode::read_write || Req->MAccessMode == access::mode::atomic)) {
                Requirement *hostReq = new Requirement(*Req);
                EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
                hostEvent->wait(hostEvent);
                delete hostReq;
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== sender add host acc" << std::endl;

                using DATA_TYPE = std::byte;
                size_t elem_size = Req->MElemSize;
                size_t buff_size = Req->MMemoryRange.size();
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== sender elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;

                SYCLMemObjI *MemObj = Req->MSYCLMemObj;
                SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
                void *UserPtr = BufferObj->getUserPtr();
                DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);

                SharedMemoryHandle handle = initSharedMemory(getpid(), kernel_count, daemon_req_count, elem_size * buff_size);
                writeToSharedMemory(handle, DataPtr, elem_size * buff_size);
                HANDLER_TRACE_STREAM << getpid() << " === handler === send host data" << std::endl;

                waitForReadCompletion(handle);
                cleanupSharedMemory(handle, elem_size * buff_size);
                HANDLER_TRACE_STREAM << getpid() << " === handler === waitForReadCompletion" << std::endl;
              }
            }
          }
          HANDLER_TRACE_STREAM << "=== SEND SHMEM DONE ===" << std::endl;
        }
        else {
          HANDLER_TRACE_STREAM << "=== EXEC ===" << std::endl;
          if (req_counts.size() > 0) {
            HANDLER_TRACE_STREAM << "=== NEED RECV SHMEM ===" << std::endl;
            for (int i = 0; i < sycl_kernel_cg->kernel_cg->MRequirements.size(); i++) {
              int daemon_req_count = i + 1;
              Requirement *Req = sycl_kernel_cg->kernel_cg->MRequirements[i];
              HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === daemon_req_count: " << daemon_req_count << " Req: " << Req << std::endl;

              if ((std::find(req_counts.begin(), req_counts.end(), daemon_req_count) != req_counts.end()) && (Req->MAccessMode == access::mode::read || Req->MAccessMode == access::mode::read_write || Req->MAccessMode == access::mode::atomic)) {
                Requirement *hostReq = new Requirement(*Req);
                EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
                hostEvent->wait(hostEvent);
                delete hostReq;
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== receiver add host acc" << std::endl;

                using DATA_TYPE = std::byte;
                size_t elem_size = Req->MElemSize;
                size_t buff_size = Req->MMemoryRange.size();
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== receiver elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;

                SYCLMemObjI *MemObj = Req->MSYCLMemObj;
                SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
                void *UserPtr = BufferObj->getUserPtr();
                DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);

                std::vector<DATA_TYPE> host_data(elem_size * buff_size);
                SharedMemoryHandle handle = initSharedMemory(getpid(), kernel_count, daemon_req_count, elem_size * buff_size);
                readFromSharedMemory(handle, host_data.data(), elem_size * buff_size);
                HANDLER_TRACE_STREAM << getpid() << " === handler === test_mem ==== Data read successfully." << std::endl;
                cleanupSharedMemory(handle, elem_size * buff_size);
                std::memcpy(DataPtr, host_data.data(), elem_size * buff_size);

                HANDLER_TRACE_STREAM << getpid() << " === handler === mem copy" << std::endl;
              }
            }
            HANDLER_TRACE_STREAM << "=== RECV SHMEM DONE ===" << std::endl;
          }
        }
      }
      
      // **注意** MQueue在此才rebind 之前有暂未出现过的逻辑使用MQueue
      // 此时有刚启动的daemon
      if (kernel_exec_info.exec) {
        const int ActualDeviceIndex =
            clampOfflineDeviceIndex(kernel_exec_info.device_index);
#ifdef SNMD_OFFLINE
        applyOfflineSplitDecision(kernel_exec_info, ActualDeviceIndex);
#endif
        std::shared_ptr<detail::queue_impl> &kernel_queue = sycl_kernel_cg->kernel_queue;
        device exec_device = detail::ProgramManager::getInstance().globalDevices.at(ActualDeviceIndex);
        detail::DeviceImplPtr dp = detail::getSyclObjImpl(exec_device);
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === rebind_device is_gpu: " << exec_device.is_gpu() << std::endl;
        kernel_queue = makeOfflineProfilingQueue(dp, kernel_queue);
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === rebind MQueue" << std::endl;

        // resubmit
        uint64_t HostStart = offlineNowNs();
        last_event = resubmit(*sycl_kernel_cg);
        uint64_t HostEnd = offlineNowNs();
        HANDLER_TRACE_STREAM << "=== handler === Offline submit kernel_count: "
                             << kernel_count << " host_duration_ns: "
                             << (HostEnd >= HostStart
                                     ? HostEnd - HostStart
                                     : 0)
                             << std::endl;
        int ProfileNumParts = 1;
#ifdef SNMD_OFFLINE
        ProfileNumParts = offlineSubmittedNumParts(kernel_count);
#endif
        profile_events.push_back({kernel_count, ActualDeviceIndex,
                                  std::max(1, ProfileNumParts), HostStart,
                                  HostEnd, last_event,
                                  findOfflineKernelProfileKey(kernel_count)});
        HANDLER_TRACE_STREAM << getpid() << " === handler === resubmitted kernel: " << kernel_count << std::endl;
        if (exec_num == kernel_exec_infos.size() - 1)
          HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === resubmit last kernel: " << kernel_count << std::endl;
      }
      else {
        HANDLER_TRACE_STREAM << "=== handler === Process " << getpid() << " === skip resubmit kernel: " << kernel_count << std::endl;
      }
    }
#ifdef SNMD_OFFLINE
    finalizeAllPendingOfflineSplits();
    applyOfflineSplitFinalizeTimes(profile_events);
#endif
    processOfflineProfilingBatch(daemon_wait_count, profile_events);
    clearOfflineBatch();
    return last_event;
  }
}



#endif
// 【END】=======================================================



void handler::addReduction(const std::shared_ptr<const void> &ReduObj) {
  MImpl->MAuxiliaryResources.push_back(ReduObj);
}

void handler::associateWithHandler(detail::AccessorBaseHost *AccBase,
                                   access::target AccTarget) {
  detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
  detail::Requirement *Req = AccImpl.get();
  // Add accessor to the list of requirements.
  MRequirements.push_back(Req);
  // Store copy of the accessor.
  MAccStorage.push_back(std::move(AccImpl));
  // Add an accessor to the handler list of associated accessors.
  // For associated accessors index does not means nothing.
  MAssociatedAccesors.emplace_back(detail::kernel_param_kind_t::kind_accessor,
                                   Req, static_cast<int>(AccTarget),
                                   /*index*/ 0);
}

static void addArgsForGlobalAccessor(detail::Requirement *AccImpl, size_t Index,
                                     size_t &IndexShift, int Size,
                                     bool IsKernelCreatedFromSource,
                                     size_t GlobalSize,
                                     std::vector<detail::ArgDesc> &Args,
                                     bool isESIMD) {
  using detail::kernel_param_kind_t;
  if (AccImpl->PerWI)
    AccImpl->resize(GlobalSize);

  Args.emplace_back(kernel_param_kind_t::kind_accessor, AccImpl, Size,
                    Index + IndexShift);

  // TODO ESIMD currently does not suport offset, memory and access ranges -
  // accessor::init for ESIMD-mode accessor has a single field, translated
  // to a single kernel argument set above.
  if (!isESIMD && !IsKernelCreatedFromSource) {
    // Dimensionality of the buffer is 1 when dimensionality of the
    // accessor is 0.
    const size_t SizeAccField =
        sizeof(size_t) * (AccImpl->MDims == 0 ? 1 : AccImpl->MDims);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout,
                      &AccImpl->MAccessRange[0], SizeAccField,
                      Index + IndexShift);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout,
                      &AccImpl->MMemoryRange[0], SizeAccField,
                      Index + IndexShift);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout,
                      &AccImpl->MOffset[0], SizeAccField, Index + IndexShift);
  }
}

void handler::processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                         const int Size, const size_t Index, size_t &IndexShift,
                         bool IsKernelCreatedFromSource, bool IsESIMD) {
  using detail::kernel_param_kind_t;

  switch (Kind) {
  case kernel_param_kind_t::kind_std_layout:
  case kernel_param_kind_t::kind_pointer: {
    MArgs.emplace_back(Kind, Ptr, Size, Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_stream: {
    // Stream contains several accessors inside.
    stream *S = static_cast<stream *>(Ptr);

    detail::AccessorBaseHost *GBufBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalBuf);
    detail::AccessorImplPtr GBufImpl = detail::getSyclObjImpl(*GBufBase);
    detail::Requirement *GBufReq = GBufImpl.get();
    addArgsForGlobalAccessor(GBufReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource,
                             MNDRDesc.GlobalSize.size(), MArgs, IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GOffsetBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalOffset);
    detail::AccessorImplPtr GOfssetImpl = detail::getSyclObjImpl(*GOffsetBase);
    detail::Requirement *GOffsetReq = GOfssetImpl.get();
    addArgsForGlobalAccessor(GOffsetReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource,
                             MNDRDesc.GlobalSize.size(), MArgs, IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GFlushBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalFlushBuf);
    detail::AccessorImplPtr GFlushImpl = detail::getSyclObjImpl(*GFlushBase);
    detail::Requirement *GFlushReq = GFlushImpl.get();

    size_t GlobalSize = MNDRDesc.GlobalSize.size();
    // If work group size wasn't set explicitly then it must be recieved
    // from kernel attribute or set to default values.
    // For now we can't get this attribute here.
    // So we just suppose that WG size is always default for stream.
    // TODO adjust MNDRDesc when device image contains kernel's attribute
    if (GlobalSize == 0) {
      // Suppose that work group size is 1 for every dimension
      GlobalSize = MNDRDesc.NumWorkGroups.size();
    }
    addArgsForGlobalAccessor(GFlushReq, Index, IndexShift, Size,
                             IsKernelCreatedFromSource, GlobalSize, MArgs,
                             IsESIMD);
    ++IndexShift;
    MArgs.emplace_back(kernel_param_kind_t::kind_std_layout,
                       &S->FlushBufferSize, sizeof(S->FlushBufferSize),
                       Index + IndexShift);

    break;
  }
  case kernel_param_kind_t::kind_accessor: {
    // For args kind of accessor Size is information about accessor.
    // The first 11 bits of Size encodes the accessor target.
    const access::target AccTarget = static_cast<access::target>(Size & 0x7ff);
    switch (AccTarget) {
    case access::target::device:
    case access::target::constant_buffer: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
      addArgsForGlobalAccessor(AccImpl, Index, IndexShift, Size,
                               IsKernelCreatedFromSource,
                               MNDRDesc.GlobalSize.size(), MArgs, IsESIMD);
      break;
    }
    case access::target::local: {
      detail::LocalAccessorImplHost *LAcc =
          static_cast<detail::LocalAccessorImplHost *>(Ptr);

      range<3> &Size = LAcc->MSize;
      const int Dims = LAcc->MDims;
      int SizeInBytes = LAcc->MElemSize;
      for (int I = 0; I < Dims; ++I)
        SizeInBytes *= Size[I];
      // Some backends do not accept zero-sized local memory arguments, so we
      // make it a minimum allocation of 1 byte.
      SizeInBytes = std::max(SizeInBytes, 1);
      MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, nullptr,
                         SizeInBytes, Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
        ++IndexShift;
        const size_t SizeAccField = Dims * sizeof(Size[0]);
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, &Size,
                           SizeAccField, Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, &Size,
                           SizeAccField, Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, &Size,
                           SizeAccField, Index + IndexShift);
      }
      break;
    }
    case access::target::image:
    case access::target::image_array: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
      MArgs.emplace_back(Kind, AccImpl, Size, Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
        // TODO Handle additional kernel arguments for image class
        // if the compiler front-end adds them.
      }
      break;
    }
    case access::target::host_image:
    case access::target::host_task:
    case access::target::host_buffer: {
      throw sycl::invalid_parameter_error("Unsupported accessor target case.",
                                          PI_ERROR_INVALID_OPERATION);
      break;
    }
    }
    break;
  }
  case kernel_param_kind_t::kind_sampler: {
    MArgs.emplace_back(kernel_param_kind_t::kind_sampler, Ptr, sizeof(sampler),
                       Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_specialization_constants_buffer: {
    MArgs.emplace_back(
        kernel_param_kind_t::kind_specialization_constants_buffer, Ptr, Size,
        Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_invalid:
    throw runtime_error("Invalid kernel param kind", PI_ERROR_INVALID_VALUE);
    break;
  }
}

// The argument can take up more space to store additional information about
// MAccessRange, MMemoryRange, and MOffset added with addArgsForGlobalAccessor.
// We use the worst-case estimate because the lifetime of the vector is short.
// In processArg the kind_stream case introduces the maximum number of
// additional arguments. The case adds additional 12 arguments to the currently
// processed argument, hence worst-case estimate is 12+1=13.
// TODO: the constant can be removed if the size of MArgs will be calculated at
// compile time.
inline constexpr size_t MaxNumAdditionalArgs = 13;

void handler::extractArgsAndReqs() {
  assert(MKernel && "MKernel is not initialized");
  std::vector<detail::ArgDesc> UnPreparedArgs = std::move(MArgs);
  MArgs.clear();

  std::sort(
      UnPreparedArgs.begin(), UnPreparedArgs.end(),
      [](const detail::ArgDesc &first, const detail::ArgDesc &second) -> bool {
        return (first.MIndex < second.MIndex);
      });

  const bool IsKernelCreatedFromSource = MKernel->isCreatedFromSource();
  MArgs.reserve(MaxNumAdditionalArgs * UnPreparedArgs.size());

  size_t IndexShift = 0;
  for (size_t I = 0; I < UnPreparedArgs.size(); ++I) {
    void *Ptr = UnPreparedArgs[I].MPtr;
    const detail::kernel_param_kind_t &Kind = UnPreparedArgs[I].MType;
    const int &Size = UnPreparedArgs[I].MSize;
    const int Index = UnPreparedArgs[I].MIndex;
    processArg(Ptr, Kind, Size, Index, IndexShift, IsKernelCreatedFromSource,
               false);
  }
}

void handler::extractArgsAndReqsFromLambda(
    char *LambdaPtr, size_t KernelArgsNum,
    const detail::kernel_param_desc_t *KernelArgs, bool IsESIMD) {
  const bool IsKernelCreatedFromSource = false;
  size_t IndexShift = 0;
  MArgs.reserve(MaxNumAdditionalArgs * KernelArgsNum);

  for (size_t I = 0; I < KernelArgsNum; ++I) {
    void *Ptr = LambdaPtr + KernelArgs[I].offset;
    const detail::kernel_param_kind_t &Kind = KernelArgs[I].kind;
    const int &Size = KernelArgs[I].info;
    if (Kind == detail::kernel_param_kind_t::kind_accessor) {
      // For args kind of accessor Size is information about accessor.
      // The first 11 bits of Size encodes the accessor target.
      const access::target AccTarget =
          static_cast<access::target>(Size & 0x7ff);
      if ((AccTarget == access::target::device ||
           AccTarget == access::target::constant_buffer) ||
          (AccTarget == access::target::image ||
           AccTarget == access::target::image_array)) {
        detail::AccessorBaseHost *AccBase =
            static_cast<detail::AccessorBaseHost *>(Ptr);
        Ptr = detail::getSyclObjImpl(*AccBase).get();
      } else if (AccTarget == access::target::local) {
        detail::LocalAccessorBaseHost *LocalAccBase =
            static_cast<detail::LocalAccessorBaseHost *>(Ptr);
        Ptr = detail::getSyclObjImpl(*LocalAccBase).get();
      }
    }
    processArg(Ptr, Kind, Size, I, IndexShift, IsKernelCreatedFromSource,
               IsESIMD);
  }
}

// Calling methods of kernel_impl requires knowledge of class layout.
// As this is impossible in header, there's a function that calls necessary
// method inside the library and returns the result.
std::string handler::getKernelName() {
  return MKernel->get_info<info::kernel::function_name>();
}

void handler::verifyUsedKernelBundle(const std::string &KernelName) {
  auto UsedKernelBundleImplPtr =
      getOrInsertHandlerKernelBundle(/*Insert=*/false);
  if (!UsedKernelBundleImplPtr)
    return;

  // Implicit kernel bundles are populated late so we ignore them
  if (!MImpl->isStateExplicitKernelBundle())
    return;

  kernel_id KernelID = detail::get_kernel_id_impl(KernelName);
  device Dev = detail::getDeviceFromHandler(*this);
  if (!UsedKernelBundleImplPtr->has_kernel(KernelID, Dev))
    throw sycl::exception(
        make_error_code(errc::kernel_not_supported),
        "The kernel bundle in use does not contain the kernel");
}

void handler::ext_oneapi_barrier(const std::vector<event> &WaitList) {
  throwIfActionIsCreated();
  MCGType = detail::CG::BarrierWaitlist;
  MEventsWaitWithBarrier.resize(WaitList.size());
  std::transform(
      WaitList.begin(), WaitList.end(), MEventsWaitWithBarrier.begin(),
      [](const event &Event) { return detail::getSyclObjImpl(Event); });
}

__SYCL2020_DEPRECATED("use 'ext_oneapi_barrier' instead")
void handler::barrier(const std::vector<event> &WaitList) {
  handler::ext_oneapi_barrier(WaitList);
}

using namespace sycl::detail;
bool handler::DisableRangeRounding() {
  return SYCLConfig<SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING>::get();
}

bool handler::RangeRoundingTrace() {
  return SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE>::get();
}

void handler::GetRangeRoundingSettings(size_t &MinFactor, size_t &GoodFactor,
                                       size_t &MinRange) {
  SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS>::GetSettings(
      MinFactor, GoodFactor, MinRange);
}

void handler::memcpy(void *Dest, const void *Src, size_t Count) {
  throwIfActionIsCreated();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;
  MLength = Count;
  setType(detail::CG::CopyUSM);
}

void handler::memset(void *Dest, int Value, size_t Count) {
  throwIfActionIsCreated();
  MDstPtr = Dest;
  MPattern.push_back(static_cast<char>(Value));
  MLength = Count;
  setType(detail::CG::FillUSM);
}

void handler::prefetch(const void *Ptr, size_t Count) {
  throwIfActionIsCreated();
  MDstPtr = const_cast<void *>(Ptr);
  MLength = Count;
  setType(detail::CG::PrefetchUSM);
}

void handler::mem_advise(const void *Ptr, size_t Count, int Advice) {
  throwIfActionIsCreated();
  MDstPtr = const_cast<void *>(Ptr);
  MLength = Count;
  MImpl->MAdvice = static_cast<pi_mem_advice>(Advice);
  setType(detail::CG::AdviseUSM);
}

void handler::ext_oneapi_memcpy2d_impl(void *Dest, size_t DestPitch,
                                       const void *Src, size_t SrcPitch,
                                       size_t Width, size_t Height) {
  // Checks done in callers.
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;
  MImpl->MSrcPitch = SrcPitch;
  MImpl->MDstPitch = DestPitch;
  MImpl->MWidth = Width;
  MImpl->MHeight = Height;
  setType(detail::CG::Copy2DUSM);
}

void handler::ext_oneapi_fill2d_impl(void *Dest, size_t DestPitch,
                                     const void *Value, size_t ValueSize,
                                     size_t Width, size_t Height) {
  // Checks done in callers.
  MDstPtr = Dest;
  MPattern.resize(ValueSize);
  std::memcpy(MPattern.data(), Value, ValueSize);
  MImpl->MDstPitch = DestPitch;
  MImpl->MWidth = Width;
  MImpl->MHeight = Height;
  setType(detail::CG::Fill2DUSM);
}

void handler::ext_oneapi_memset2d_impl(void *Dest, size_t DestPitch, int Value,
                                       size_t Width, size_t Height) {
  // Checks done in callers.
  MDstPtr = Dest;
  MPattern.push_back(static_cast<char>(Value));
  MImpl->MDstPitch = DestPitch;
  MImpl->MWidth = Width;
  MImpl->MHeight = Height;
  setType(detail::CG::Memset2DUSM);
}

void handler::use_kernel_bundle(
    const kernel_bundle<bundle_state::executable> &ExecBundle) {

  std::shared_ptr<detail::queue_impl> PrimaryQueue =
      MImpl->MSubmissionPrimaryQueue;
  if (PrimaryQueue->get_context() != ExecBundle.get_context())
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Context associated with the primary queue is different from the "
        "context associated with the kernel bundle");

  std::shared_ptr<detail::queue_impl> SecondaryQueue =
      MImpl->MSubmissionSecondaryQueue;
  if (SecondaryQueue &&
      SecondaryQueue->get_context() != ExecBundle.get_context())
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Context associated with the secondary queue is different from the "
        "context associated with the kernel bundle");

  setStateExplicitKernelBundle();
  setHandlerKernelBundle(detail::getSyclObjImpl(ExecBundle));
}

void handler::depends_on(event Event) {
  auto EventImpl = detail::getSyclObjImpl(Event);
  if (EventImpl->isDiscarded()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Queue operation cannot depend on discarded event.");
  }
  MEvents.push_back(EventImpl);
}

void handler::depends_on(const std::vector<event> &Events) {
  for (const event &Event : Events) {
    auto EventImpl = detail::getSyclObjImpl(Event);
    if (EventImpl->isDiscarded()) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Queue operation cannot depend on discarded event.");
    }
    MEvents.push_back(EventImpl);
  }
}

static bool
checkContextSupports(const std::shared_ptr<detail::context_impl> &ContextImpl,
                     detail::RT::PiContextInfo InfoQuery) {
  auto &Plugin = ContextImpl->getPlugin();
  pi_bool SupportsOp = false;
  Plugin.call<detail::PiApiKind::piContextGetInfo>(ContextImpl->getHandleRef(),
                                                   InfoQuery, sizeof(pi_bool),
                                                   &SupportsOp, nullptr);
  return SupportsOp;
}

bool handler::supportsUSMMemcpy2D() {
  for (const std::shared_ptr<detail::queue_impl> &QueueImpl :
       {MImpl->MSubmissionPrimaryQueue, MImpl->MSubmissionSecondaryQueue}) {
    if (QueueImpl &&
        !checkContextSupports(QueueImpl->getContextImplPtr(),
                              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT))
      return false;
  }
  return true;
}

bool handler::supportsUSMFill2D() {
  for (const std::shared_ptr<detail::queue_impl> &QueueImpl :
       {MImpl->MSubmissionPrimaryQueue, MImpl->MSubmissionSecondaryQueue}) {
    if (QueueImpl &&
        !checkContextSupports(QueueImpl->getContextImplPtr(),
                              PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT))
      return false;
  }
  return true;
}

bool handler::supportsUSMMemset2D() {
  for (const std::shared_ptr<detail::queue_impl> &QueueImpl :
       {MImpl->MSubmissionPrimaryQueue, MImpl->MSubmissionSecondaryQueue}) {
    if (QueueImpl &&
        !checkContextSupports(QueueImpl->getContextImplPtr(),
                              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT))
      return false;
  }
  return true;
}

id<2> handler::computeFallbackKernelBounds(size_t Width, size_t Height) {
  device Dev = MQueue->get_device();
  id<2> ItemLimit = Dev.get_info<info::device::max_work_item_sizes<2>>() *
                    Dev.get_info<info::device::max_compute_units>();
  return id<2>{std::min(ItemLimit[0], Height), std::min(ItemLimit[1], Width)};
}

void handler::memcpyToDeviceGlobal(const void *DeviceGlobalPtr, const void *Src,
                                   bool IsDeviceImageScoped, size_t NumBytes,
                                   size_t Offset) {
  throwIfActionIsCreated();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = const_cast<void *>(DeviceGlobalPtr);
  MImpl->MIsDeviceImageScoped = IsDeviceImageScoped;
  MLength = NumBytes;
  MImpl->MOffset = Offset;
  setType(detail::CG::CopyToDeviceGlobal);
}

void handler::memcpyFromDeviceGlobal(void *Dest, const void *DeviceGlobalPtr,
                                     bool IsDeviceImageScoped, size_t NumBytes,
                                     size_t Offset) {
  throwIfActionIsCreated();
  MSrcPtr = const_cast<void *>(DeviceGlobalPtr);
  MDstPtr = Dest;
  MImpl->MIsDeviceImageScoped = IsDeviceImageScoped;
  MLength = NumBytes;
  MImpl->MOffset = Offset;
  setType(detail::CG::CopyFromDeviceGlobal);
}

const std::shared_ptr<detail::context_impl> &
handler::getContextImplPtr() const {
  return MQueue->getContextImplPtr();
}

void handler::setKernelCacheConfig(
    detail::RT::PiKernelCacheConfig Config) {
  MImpl->MKernelCacheConfig = Config;
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
