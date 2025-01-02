//==-------- handler.cpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>

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
#include <detail/daemon/daemon.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/pi.h>
#include <sycl/detail/pi.hpp>
#include <sycl/access/access.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/event.hpp>
#include <sycl/handler.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/stream.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <mqueue.h>
#include <unistd.h>

// #define PRINT_TRACE 1
#define REBIND 1
#define SCHEDULE 1

extern mqd_t mq_id_daemon, mq_id_program;

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
    // scale_count也避免通信
    // TODO 暂时不考虑scale的存在依赖
    else if (daemon_kernel_count == daemon_scale_count) {
      device exec_device = detail::ProgramManager::getInstance().globalDevices.at(detail::ProgramManager::getInstance().scale_device);
      detail::DeviceImplPtr dp = detail::getSyclObjImpl(exec_device);
      std::cout << "=== handler === Process " << getpid() << " === rebind_device is_gpu: " << exec_device.is_gpu() << std::endl;
      MQueue.reset(new detail::queue_impl(dp, detail::queue_impl::getDefaultOrNew(dp), MQueue->getAsyncHandler(), MQueue->getPropertyList()));
    } else {
      // ====【打包kernel内req发送给daemon】
      // 因为scheduler维护了kernel历史执行 所以不需要数据移动SameCtx判断
      // 即使SameCtx判断 有的Req可能并不在同一节点上 没有必要
      // TODO Req对应内存的具体信息 如大小
      S2DKernelReqData kernel_req_data;
      {
        detail::combineAccessModesOfReqs(MRequirements);

        kernel_req_data.pid = getpid();
        kernel_req_data.kernel_count = daemon_kernel_count;
        kernel_req_data.req_size = MRequirements.size();

        for (int i = 0; i < MRequirements.size(); i++) {
          Requirement *Req = MRequirements[i];

          SyclReqData req_data;
          req_data.mem_pointer = Req->MSYCLMemObj;
          req_data.kernel_count = kernel_req_data.kernel_count;
          req_data.req_count = i + 1;
          req_data.req_accmode = static_cast<acc_mode>(Req->MAccessMode);

          kernel_req_data.reqs.push_back(req_data);
        }

        std::string serialized_data = kernel_req_data.serialize();
        size_t message_size = serialized_data.size();

        mq_send(mq_id_daemon, serialized_data.c_str(), message_size, 0);
        std::cout << "=== handler === Process " << getpid() << " === mq_send kernel_req_data" << std::endl;
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
          detail::ProgramManager::getInstance().scale_device = kernel_exec_info.device_index;
          std::cout << "=== handler === Process " << getpid() << " === scale_count: " << daemon_scale_count << " device_index: " << kernel_exec_info.device_index << std::endl;
          return MLastEvent;
        } else {
          std::cout << "=== handler === Process " << getpid() << " === mq_receive kernel_exec_info === kernel_count: " << kernel_exec_info.kernel_count << " exec: " << kernel_exec_info.exec << " device_index: " << kernel_exec_info.device_index << " req_size: " << kernel_exec_info.req_counts.size() << std::endl;
        }
      }

      // ====【处理kernel的依赖数据】
      {
        if (daemon_kernel_count != kernel_exec_info.kernel_count) {
          std::cout << "=== handler === Process " << getpid() << " === kernel count not match" << std::endl;
          exit(1);
        }
        auto &req_counts = kernel_exec_info.req_counts;
        // 如果不执行 检查是否需要host->device 完成后再返回
        if (!kernel_exec_info.exec) {
          if (req_counts.size() > 0) {
            for (int i = 0; i < MRequirements.size(); i++) {
              int daemon_req_count = i + 1;
              Requirement *Req = MRequirements[i];
              if ((std::find(req_counts.begin(), req_counts.end(), daemon_req_count) != req_counts.end()) && Req->MAccessMode == access::mode::read) {
                Requirement *hostReq = new Requirement(*Req);
                EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
                hostEvent->wait(hostEvent);
                delete hostReq;
                std::cout << "=== handler === test_mem ==== sender add host acc" << std::endl;

                // ========【固定测试】
                using DATA_TYPE = float;
                SYCLMemObjI *MemObj = Req->MSYCLMemObj;
                SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
                void *UserPtr = BufferObj->getUserPtr();
                DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);
                // int size = 256;
                // for(int i = 0; i < size; i++) {
                //   std::cerr << DataPtr[i * size] << " ";
                //   std::cerr << std::endl;
                // }
                std::cout << "=== handler === test_mem ==== sender get user ptr" << std::endl;

                SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, daemon_kernel_count, daemon_req_count);
                writeToSharedMemory(handle, DataPtr);

                std::cout << "=== handler === send host data" << std::endl;

                waitForReadCompletion(handle);
                cleanupSharedMemory(handle);

                std::cout << "=== handler === waitForReadCompletion" << std::endl;
              }
            }
          }
          std::cout << "=== handler === kernel_count: " << daemon_kernel_count << " MLastEvent: " << &MLastEvent << std::endl;
          return MLastEvent;
        }
        // 如果执行 需要等待可能的通信数据 也需要host 被用的时候再从host->device
        else {
          if (req_counts.size() > 0) {
            for (int i = 0; i < MRequirements.size(); i++) {
              int daemon_req_count = i + 1;
              Requirement *Req = MRequirements[i];
              if ((std::find(req_counts.begin(), req_counts.end(), daemon_req_count) != req_counts.end()) && Req->MAccessMode == access::mode::read) {
                Requirement *hostReq = new Requirement(*Req);
                EventImplPtr hostEvent = detail::Scheduler::getInstance().addHostAccessor(hostReq);
                hostEvent->wait(hostEvent);
                delete hostReq;
                std::cout << "=== handler === test_mem ==== receiver add host acc" << std::endl;

                using DATA_TYPE = float;
                SYCLMemObjI *MemObj = Req->MSYCLMemObj;
                SYCLMemObjT *BufferObj = static_cast<SYCLMemObjT *>(MemObj);
                void *UserPtr = BufferObj->getUserPtr();
                DATA_TYPE *DataPtr = static_cast<DATA_TYPE *>(UserPtr);
                std::cout << "=== handler === test_mem ==== receiver get user ptr" << std::endl;

                std::vector<DATA_TYPE> host_data(VECTOR_SIZE);
                SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, daemon_kernel_count, daemon_req_count);
                readFromSharedMemory(handle, host_data.data());
                std::cout << "=== handler === test_mem ==== Data read successfully." << std::endl;
                cleanupSharedMemory(handle);
                std::memcpy(DataPtr, host_data.data(), MEMORY_SIZE);

                std::cout << "=== handler === mem copy" << std::endl;
              }
            }
          }
        }
      }

      // ====【执行进程rebind】
      if (kernel_exec_info.exec) {
        device exec_device = detail::ProgramManager::getInstance().globalDevices.at(kernel_exec_info.device_index);
        detail::DeviceImplPtr dp = detail::getSyclObjImpl(exec_device);
        std::cout << "=== handler === Process " << getpid() << " === rebind_device is_gpu: " << exec_device.is_gpu() << std::endl;
        MQueue.reset(new detail::queue_impl(dp, detail::queue_impl::getDefaultOrNew(dp), MQueue->getAsyncHandler(), MQueue->getPropertyList()));
      }
    }
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
      std::cout << "=== handler === BEFORE REQS" << std::endl;

      // 在gloabal_handler初始化时获取所有device，并循环判断如果使用每个device的queue会造成几次数据移动
      for (device tempD : detail::ProgramManager::getInstance().globalDevices) {
        detail::DeviceImplPtr tempDP = detail::getSyclObjImpl(tempD);
        MQueue.reset(new detail::queue_impl(tempDP, detail::queue_impl::getDefaultOrNew(tempDP), MQueue->getAsyncHandler(), MQueue->getPropList()));
        int notSameCtxCount = 0;

        std::cout << "=== handler === TRY: " << tempD.get_info<info::device::name>() << std::endl;
        for (Requirement *Req : MRequirements) {
          // Req->MAccessMode
          MemObjRecord *record = nullptr;
          AllocaCommandBase *allocaCmd = nullptr;
          bool isSameCtx = false;
          // std::cout << "=== handler === req 1"<< std::endl;

          // SYCLMemObj生命周期由用户代码管理 MemObjRecord生命周期由syclrt管理
          // SYCLMemObjI *MemObject = Req->MSYCLMemObj;
          // MemObjRecord *Record = getMemObjRecord(MemObject);
          //                      = MemObject->MRecord.get();
          record = detail::Scheduler::getInstance().MGraphBuilder.getOrInsertMemObjRecord(MQueue, Req, ToEnqueue);
          std::cout << "=== handler === test_mem ==== record: " << record << std::endl;

          // 不清楚具体逻辑 不需要
          // detail::Scheduler::getInstance().MGraphBuilder.markModifiedIfWrite(record, Req);
          
          // 如果找不到对应的allocaCmd就创建了 不对需要销毁 如何销毁？
          // allocaCmd = detail::Scheduler::getInstance().MGraphBuilder.getOrCreateAllocaForReq(record, Req, MQueue, ToEnqueue);
          // std::cout << "=== handler === test_mem ==== allocaCmd: " << allocaCmd << std::endl;
          
          // std::cout << "=== handler === req 2"<< std::endl;
          isSameCtx = detail::sameCtx(MQueue->getContextImplPtr(), record->MCurContext);
          if (!isSameCtx) {
            notSameCtxCount++;
          }
        }
        std::cout << "=== handler === notSameCtx: " << notSameCtxCount << std::endl;
      }
    }
#endif
  for (device d : detail::ProgramManager::getInstance().globalDevices) {
    std::cout << "=== handler === Process " << getpid() << " === global_device cpu: " << d.is_cpu() << " gpu: " << d.is_gpu() << " acc: " << d.is_accelerator() << std::endl;
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
  std::cout << "=== handler === Process " << getpid() << " === rebind_device is_gpu: " << d.is_gpu() << std::endl;
  // MQueue->rebindDevice(dp);
  MQueue.reset(new detail::queue_impl(dp, detail::queue_impl::getDefaultOrNew(dp), MQueue->getAsyncHandler(), MQueue->getPropList()));

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

        // std::cout << "=== handler === req 1"<< std::endl;
        // 获取的就是内存需求 对应数组
        record = detail::Scheduler::getInstance().MGraphBuilder.getOrInsertMemObjRecord(MQueue, Req, ToEnqueue);
        std::cout << "=== handler === test_mem ==== record: " << record << std::endl;
        std::cout << "=== handler === test_mem ==== memobj: " << Req->MSYCLMemObj << std::endl;

        if (detail::ProgramManager::getInstance().kernel_count == 3) {
          // std::cout << "=== handler === test_mem ==== kernel count 3" << std::endl;
          if (Req->MAccessMode == access::mode::read && testReqCount == 1) {
            // 尝试将device仅复制到host
      
            // addCopyBack 不太行 会导致后面再copyback的时候崩溃
            // Command *NewCmd = detail::Scheduler::getInstance().MGraphBuilder.addCopyBack(Req, ToEnqueue);
            // std::cout << "=== handler === test_mem ==== read copy back: " << record << std::endl;

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

            std::cout << "=== handler === test_mem ==== add host acc" << std::endl;

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
            std::cout << "=== handler === test_mem ==== get user ptr" << std::endl;
          }
        }
        
        // std::cout << "=== handler === req 2"<< std::endl;
        isSameCtx = detail::sameCtx(MQueue->getContextImplPtr(), record->MCurContext);
        if (!isSameCtx) {
          notSameCtxCount++;
        }
      }
      std::cout << "=== handler === REBIND === notSameCtx: " << notSameCtxCount << std::endl;
    }    
  }
#endif
#endif
// 【END】=======================================================
  


  // 单独对特殊情况的kernel处理 目前没有 按道理可以忽略 但必须在之前rebind 因为有调用
  const auto &type = getType();
  if (type == detail::CG::Kernel) {
    // If there were uses of set_specialization_constant build the kernel_bundle
    std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpPtr =
        getOrInsertHandlerKernelBundle(/*Insert=*/false);
    // 目前没有遇到kernel_bundle的情况
    // kernel_bundle主要用于用户控制kernel的编译和链接
    if (KernelBundleImpPtr) {
      #ifdef PRINT_TRACE
      std::cout << "======handler.cpp KernelBundleImpPtr" << std::endl;
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
        std::cout << "======handler.cpp DiscardEvent" << std::endl;
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
      std::cout << "WARNING: An empty command group is submitted." << std::endl;
    }
    detail::EventImplPtr Event = std::make_shared<sycl::detail::event_impl>();
    MLastEvent = detail::createSyclObjFromImpl<event>(Event);
    return MLastEvent;
  }

  if (!CommandGroup)
    throw sycl::runtime_error(
        "Internal Error. Command group cannot be constructed.",
        PI_ERROR_INVALID_OPERATION);

  #ifdef PRINT_TRACE
  std::cout << "======handler.cpp === type: " << type << " req: " << MRequirements.size() << " queue: " << MQueue << " event: " << MEvents.size() << " lastevent: " << &MLastEvent << std::endl;
  #endif

  detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), std::move(MQueue));

  MLastEvent = detail::createSyclObjFromImpl<event>(Event);

  // #ifdef PRINT_TRACE
  // std::cout << "======handler.cpp after === queue: " << MQueue << " event: " << MEvents.size() << " lastevent: " << &MLastEvent << std::endl;
  // #endif

  return MLastEvent;
}

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
