//==------------- cg.cpp - SYCL command group implementation -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/accessor_impl.hpp>
#include <sycl/detail/cg.hpp>

#include <cstring>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

std::unique_ptr<CGExecKernel>
CGExecKernel::cloneForSplit(const NDRDescT &NewNDR) const {
  // CG base owns accessor implementations through MAccStorage, while MArgs
  // stores raw pointers to each accessor and its layout fields.  Split CGs
  // must not share those raw pointers because each split command may be
  // wired to a different allocation and context.
  auto ArgsStorageCopy = MArgsStorage; // vector<vector<char>>
  std::vector<detail::AccessorImplPtr> AccStorageCopy;
  AccStorageCopy.reserve(MAccStorage.size() + MRequirements.size());
  std::vector<std::pair<AccessorImplHost *, AccessorImplHost *>> ReqMap;

  auto FindMappedReq = [&ReqMap](AccessorImplHost *OldReq) {
    for (const auto &Entry : ReqMap) {
      if (Entry.first == OldReq)
        return Entry.second;
    }
    return static_cast<AccessorImplHost *>(nullptr);
  };

  for (const detail::AccessorImplPtr &Acc : MAccStorage) {
    if (!Acc) {
      AccStorageCopy.push_back(nullptr);
      continue;
    }
    auto AccCopy = std::make_shared<AccessorImplHost>(*Acc);
    ReqMap.push_back({Acc.get(), AccCopy.get()});
    AccStorageCopy.push_back(std::move(AccCopy));
  }

  for (AccessorImplHost *Req : MRequirements) {
    if (Req == nullptr || FindMappedReq(Req) != nullptr)
      continue;
    auto ReqCopy = std::make_shared<AccessorImplHost>(*Req);
    ReqMap.push_back({Req, ReqCopy.get()});
    AccStorageCopy.push_back(std::move(ReqCopy));
  }

  auto SharedPtrStorageCopy = MSharedPtrStorage;
  auto EventsCopy = MEvents;
  std::unique_ptr<HostKernelBase> HostKernelCopy;
  char *OldHostKernelPtr = nullptr;
  char *NewHostKernelPtr = nullptr;
  size_t HostKernelSize = 0;

  if (MHostKernel) {
    OldHostKernelPtr = MHostKernel->getPtr();
    HostKernelSize = MHostKernel->getSize();
    HostKernelCopy = MHostKernel->clone();
    NewHostKernelPtr = HostKernelCopy ? HostKernelCopy->getPtr() : nullptr;
  }

  auto RemapArgStoragePtr = [this, &ArgsStorageCopy](void *Ptr) -> void * {
    if (Ptr == nullptr)
      return nullptr;

    const std::uintptr_t PtrValue = reinterpret_cast<std::uintptr_t>(Ptr);
    for (size_t I = 0; I < MArgsStorage.size(); ++I) {
      if (MArgsStorage[I].empty())
        continue;

      const std::uintptr_t OldBegin =
          reinterpret_cast<std::uintptr_t>(MArgsStorage[I].data());
      const std::uintptr_t OldEnd = OldBegin + MArgsStorage[I].size();
      if (PtrValue >= OldBegin && PtrValue < OldEnd) {
        const std::uintptr_t Offset = PtrValue - OldBegin;
        return ArgsStorageCopy[I].data() + Offset;
      }
    }

    return Ptr;
  };

  auto RemapHostKernelPtr = [OldHostKernelPtr, NewHostKernelPtr,
                             HostKernelSize](void *Ptr) -> void * {
    if (Ptr == nullptr || OldHostKernelPtr == nullptr ||
        NewHostKernelPtr == nullptr || HostKernelSize == 0)
      return Ptr;

    const std::uintptr_t PtrValue = reinterpret_cast<std::uintptr_t>(Ptr);
    const std::uintptr_t OldBegin =
        reinterpret_cast<std::uintptr_t>(OldHostKernelPtr);
    const std::uintptr_t OldEnd = OldBegin + HostKernelSize;
    if (PtrValue >= OldBegin && PtrValue < OldEnd) {
      const std::uintptr_t Offset = PtrValue - OldBegin;
      return NewHostKernelPtr + Offset;
    }

    return Ptr;
  };

  auto ReqsCopy = MRequirements; // vector<AccessorImplHost *>
  for (AccessorImplHost *&Req : ReqsCopy) {
    if (AccessorImplHost *MappedReq = FindMappedReq(Req))
      Req = MappedReq;
  }
  std::vector<AccessorImplHost *> PartitionLocalReqsCopy;
  PartitionLocalReqsCopy.reserve(MSNMDPartitionLocalReqs.size());
  for (AccessorImplHost *Req : MSNMDPartitionLocalReqs) {
    if (AccessorImplHost *MappedReq = FindMappedReq(Req))
      PartitionLocalReqsCopy.push_back(MappedReq);
  }

  auto ArgsCopy = MArgs; // vector<ArgDesc>
  auto RemapReqArgPtr = [&ReqMap](void *Ptr) -> void * {
    for (const auto &Entry : ReqMap) {
      AccessorImplHost *OldReq = Entry.first;
      AccessorImplHost *NewReq = Entry.second;
      if (Ptr == OldReq)
        return NewReq;
      if (Ptr == static_cast<void *>(&OldReq->MAccessRange[0]))
        return &NewReq->MAccessRange[0];
      if (Ptr == static_cast<void *>(&OldReq->MMemoryRange[0]))
        return &NewReq->MMemoryRange[0];
      if (Ptr == static_cast<void *>(&OldReq->MOffset[0]))
        return &NewReq->MOffset[0];
    }
    return Ptr;
  };
  for (ArgDesc &Arg : ArgsCopy) {
    void *MappedPtr = RemapReqArgPtr(Arg.MPtr);
    if (MappedPtr == Arg.MPtr)
      MappedPtr = RemapArgStoragePtr(Arg.MPtr);
    if (MappedPtr == Arg.MPtr)
      MappedPtr = RemapHostKernelPtr(Arg.MPtr);

    // Lambda kernels taking item/id/nd_item are normalized through a
    // std::function.  In that case extractArgsAndReqsFromLambda records plain
    // captured arguments using pointers into std::function's heap-allocated
    // target, not necessarily into HostKernelBase::getPtr()/getSize().  A copy
    // of the std::function owns a different target, so address-range remapping
    // alone can leave split CGs with dangling scalar arguments.  Snapshot
    // value and pointer arguments into storage owned by this clone.  Accessor
    // arguments still use ReqMap above because the scheduler needs their
    // independently owned Requirement objects.
    if ((Arg.MType == kernel_param_kind_t::kind_std_layout ||
         Arg.MType == kernel_param_kind_t::kind_pointer) &&
        MappedPtr != nullptr && Arg.MSize > 0) {
      std::vector<char> ArgValue(static_cast<size_t>(Arg.MSize));
      std::memcpy(ArgValue.data(), MappedPtr, ArgValue.size());
      ArgsStorageCopy.push_back(std::move(ArgValue));
      MappedPtr = ArgsStorageCopy.back().data();
    }
    Arg.MPtr = MappedPtr;
  }

  return std::make_unique<CGExecKernel>(
      NewNDR,
      std::move(HostKernelCopy), // unique_ptr HostKernel
      MSyclKernel, // shared_ptr
      MKernelBundle, // shared_ptr
      std::move(ArgsStorageCopy),
      std::move(AccStorageCopy),
      std::move(SharedPtrStorageCopy),
      std::move(ReqsCopy),
      std::move(EventsCopy),
      std::move(ArgsCopy), // MArgs.MPtr直接作为Req*
      MKernelName,
      MOSModuleHandle,
      MStreams, // vector<shared_ptr<...>> 值拷贝 共享资源
      MAuxiliaryResources, // vector<shared_ptr<const void>> 共享资源
      MType,
      MKernelCacheConfig,
      code_location{},
      std::move(PartitionLocalReqsCopy));
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
