//==------------------------- plugin.hpp - SYCL platform -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/config.hpp>
#include <detail/plugin_printers.hpp>
#include <memory>
#include <mutex>
#include <sycl/backend_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/stl.hpp>
#include <map>
#include <string>

#define PRINT_PI 1

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting traces using the trace framework
#include "xpti/xpti_trace_framework.h"
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
extern xpti::trace_event_data_t *GPICallEvent;
extern xpti::trace_event_data_t *GPIArgCallEvent;
#endif

template <PiApiKind Kind, size_t Idx, typename... Args>
struct array_fill_helper;

template <PiApiKind Kind> struct PiApiArgTuple;

#define _PI_API(api)                                                           \
  template <> struct PiApiArgTuple<PiApiKind::api> {                           \
    using type = typename function_traits<decltype(api)>::args_type;           \
  };

#include <sycl/detail/pi.def>
#undef _PI_API

template <PiApiKind Kind, size_t Idx, typename T>
struct array_fill_helper<Kind, Idx, T> {
  static void fill(unsigned char *Dst, T &&Arg) {
    using ArgsTuple = typename PiApiArgTuple<Kind>::type;
    // C-style cast is required here.
    auto RealArg = (std::tuple_element_t<Idx, ArgsTuple>)(Arg);
    *(std::remove_cv_t<std::tuple_element_t<Idx, ArgsTuple>> *)Dst = RealArg;
  }
};

template <PiApiKind Kind, size_t Idx, typename T, typename... Args>
struct array_fill_helper<Kind, Idx, T, Args...> {
  static void fill(unsigned char *Dst, const T &&Arg, Args &&...Rest) {
    using ArgsTuple = typename PiApiArgTuple<Kind>::type;
    // C-style cast is required here.
    auto RealArg = (std::tuple_element_t<Idx, ArgsTuple>)(Arg);
    *(std::remove_cv_t<std::tuple_element_t<Idx, ArgsTuple>> *)Dst = RealArg;
    array_fill_helper<Kind, Idx + 1, Args...>::fill(
        Dst + sizeof(decltype(RealArg)), std::forward<Args>(Rest)...);
  }
};

template <typename... Ts>
constexpr size_t totalSize(const std::tuple<Ts...> &) {
  return (sizeof(Ts) + ...);
}

template <PiApiKind Kind, typename... ArgsT>
auto packCallArguments(ArgsT &&...Args) {
  using ArgsTuple = typename PiApiArgTuple<Kind>::type;

  constexpr size_t TotalSize = totalSize(ArgsTuple{});

  std::array<unsigned char, TotalSize> ArgsData;
  array_fill_helper<Kind, 0, ArgsT...>::fill(ArgsData.data(),
                                             std::forward<ArgsT>(Args)...);

  return ArgsData;
}

/// The plugin class provides a unified interface to the underlying low-level
/// runtimes for the device-agnostic SYCL runtime.
///
/// \ingroup sycl_pi
class plugin {
public:
  plugin() = delete;
  plugin(const std::shared_ptr<RT::PiPlugin> &Plugin, backend UseBackend,
         void *LibraryHandle)
      : MPlugin(Plugin), MBackend(UseBackend), MLibraryHandle(LibraryHandle),
        TracingMutex(std::make_shared<std::mutex>()),
        MPluginMutex(std::make_shared<std::mutex>()) {}

  plugin &operator=(const plugin &) = default;
  plugin(const plugin &) = default;
  plugin &operator=(plugin &&other) noexcept = default;
  plugin(plugin &&other) noexcept = default;

  ~plugin() = default;

  const RT::PiPlugin &getPiPlugin() const { return *MPlugin; }
  RT::PiPlugin &getPiPlugin() { return *MPlugin; }
  const std::shared_ptr<RT::PiPlugin> &getPiPluginPtr() const {
    return MPlugin;
  }

  /// Checks return value from PI calls.
  ///
  /// \throw Exception if pi_result is not a PI_SUCCESS.
  template <typename Exception = sycl::runtime_error>
  void checkPiResult(RT::PiResult pi_result) const {
    char *message = nullptr;
    if (pi_result == PI_ERROR_PLUGIN_SPECIFIC_ERROR) {
      pi_result = call_nocheck<PiApiKind::piPluginGetLastError>(&message);

      // If the warning level is greater then 2 emit the message
      if (detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() >= 2)
        std::clog << message << std::endl;

      // If it is a warning do not throw code
      if (pi_result == PI_SUCCESS)
        return;
    }
    __SYCL_CHECK_OCL_CODE_THROW(pi_result, Exception, message);
  }

  /// \throw SYCL 2020 exception(errc) if pi_result is not PI_SUCCESS
  template <sycl::errc errc> void checkPiResult(RT::PiResult pi_result) const {
    if (pi_result == PI_ERROR_PLUGIN_SPECIFIC_ERROR) {
      char *message = nullptr;
      pi_result = call_nocheck<PiApiKind::piPluginGetLastError>(&message);

      // If the warning level is greater then 2 emit the message
      if (detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() >= 2)
        std::clog << message << std::endl;

      // If it is a warning do not throw code
      if (pi_result == PI_SUCCESS)
        return;
    }
    __SYCL_CHECK_CODE_THROW_VIA_ERRC(pi_result, errc);
  }

  void reportPiError(RT::PiResult pi_result, const char *context) const {
    if (pi_result != PI_SUCCESS) {
      throw sycl::runtime_error(std::string(context) +
                                    " API failed with error: " +
                                    sycl::detail::codeToString(pi_result),
                                pi_result);
    }
  }

  /// Calls the PiApi, traces the call, and returns the result.
  ///
  /// Usage:
  /// \code{cpp}
  /// PiResult Err = plugin.call<PiApiKind::pi>(Args);
  /// Plugin.checkPiResult(Err); // Checks Result and throws a runtime_error
  /// // exception.
  /// \endcode
  ///
  /// \sa plugin::checkPiResult

  template <PiApiKind PiApiOffset, typename... ArgsT>
  RT::PiResult call_nocheck(ArgsT... Args) const {
    RT::PiFuncInfo<PiApiOffset> PiCallInfo;
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Emit a function_begin trace for the PI API before the call is executed.
    // If arguments need to be captured, then a data structure can be sent in
    // the per_instance_user_data field.
    const char *PIFnName = PiCallInfo.getFuncName();
    uint64_t CorrelationID = pi::emitFunctionBeginTrace(PIFnName);
    uint64_t CorrelationIDWithArgs = 0;
    unsigned char *ArgsDataPtr = nullptr;
    // TODO check if stream is observed when corresponding API is present.
    if (xptiTraceEnabled()) {
      auto ArgsData =
          packCallArguments<PiApiOffset>(std::forward<ArgsT>(Args)...);
      ArgsDataPtr = ArgsData.data();
      CorrelationIDWithArgs = pi::emitFunctionWithArgsBeginTrace(
          static_cast<uint32_t>(PiApiOffset), PIFnName, ArgsDataPtr, *MPlugin);
    }
#endif
    RT::PiResult R;
    if (pi::trace(pi::TraceLevel::PI_TRACE_CALLS)) {
      std::lock_guard<std::mutex> Guard(*TracingMutex);
      const char *FnName = PiCallInfo.getFuncName();
      std::cout << "---> " << FnName << "(" << std::endl;
      RT::printArgs(Args...);
      R = PiCallInfo.getFuncPtr(*MPlugin)(Args...);
      std::cout << ") ---> ";
      RT::printArgs(R);
      RT::printOuts(Args...);
      std::cout << std::endl;
    } else {
      R = PiCallInfo.getFuncPtr(*MPlugin)(Args...);
      
      #if PRINT_PI
      int key = static_cast<int>(PiApiOffset);
      std::string value;
      auto it = PiApiKindString.find(key);
      if (it != PiApiKindString.end()) {
          value = it->second;
      }
      std::cout << "Call PiApi: " << key << " " << value << "\n";
      // (std::cout << ... << typeid(ArgsT).name()) << " ";
      // (std::cout << ... << Args) << '\n';
      #endif
    }
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Close the function begin with a call to function end
    pi::emitFunctionEndTrace(CorrelationID, PIFnName);
    pi::emitFunctionWithArgsEndTrace(CorrelationIDWithArgs,
                                     static_cast<uint32_t>(PiApiOffset),
                                     PIFnName, ArgsDataPtr, R, *MPlugin);
#endif
    return R;
  }

  /// Calls the API, traces the call, checks the result
  ///
  /// \throw sycl::runtime_exception if the call was not successful.
  template <PiApiKind PiApiOffset, typename... ArgsT>
  void call(ArgsT... Args) const {
    RT::PiResult Err = call_nocheck<PiApiOffset>(Args...);
    checkPiResult(Err);
  }

  /// \throw sycl::exceptions(errc) if the call was not successful.
  template <sycl::errc errc, PiApiKind PiApiOffset, typename... ArgsT>
  void call(ArgsT... Args) const {
    RT::PiResult Err = call_nocheck<PiApiOffset>(Args...);
    checkPiResult<errc>(Err);
  }

  backend getBackend(void) const { return MBackend; }
  void *getLibraryHandle() const { return MLibraryHandle; }
  void *getLibraryHandle() { return MLibraryHandle; }
  int unload() { return RT::unloadPlugin(MLibraryHandle); }

  // return the index of PiPlatforms.
  // If not found, add it and return its index.
  // The function is expected to be called in a thread safe manner.
  int getPlatformId(RT::PiPlatform Platform) {
    auto It = std::find(PiPlatforms.begin(), PiPlatforms.end(), Platform);
    if (It != PiPlatforms.end())
      return It - PiPlatforms.begin();

    PiPlatforms.push_back(Platform);
    LastDeviceIds.push_back(0);
    return PiPlatforms.size() - 1;
  }

  // Device ids are consecutive across platforms within a plugin.
  // We need to return the same starting index for the given platform.
  // So, instead of returing the last device id of the given platform,
  // return the last device id of the predecessor platform.
  // The function is expected to be called in a thread safe manner.
  int getStartingDeviceId(RT::PiPlatform Platform) {
    int PlatformId = getPlatformId(Platform);
    if (PlatformId == 0)
      return 0;
    return LastDeviceIds[PlatformId - 1];
  }

  // set the id of the last device for the given platform
  // The function is expected to be called in a thread safe manner.
  void setLastDeviceId(RT::PiPlatform Platform, int Id) {
    int PlatformId = getPlatformId(Platform);
    LastDeviceIds[PlatformId] = Id;
  }

  // Adjust the id of the last device for the given platform.
  // Involved when there is no device on that platform at all.
  // The function is expected to be called in a thread safe manner.
  void adjustLastDeviceId(RT::PiPlatform Platform) {
    int PlatformId = getPlatformId(Platform);
    if (PlatformId > 0 &&
        LastDeviceIds[PlatformId] < LastDeviceIds[PlatformId - 1])
      LastDeviceIds[PlatformId] = LastDeviceIds[PlatformId - 1];
  }

  bool containsPiPlatform(RT::PiPlatform Platform) {
    auto It = std::find(PiPlatforms.begin(), PiPlatforms.end(), Platform);
    return It != PiPlatforms.end();
  }

  std::shared_ptr<std::mutex> getPluginMutex() { return MPluginMutex; }

private:
  std::shared_ptr<RT::PiPlugin> MPlugin;
  backend MBackend;
  void *MLibraryHandle; // the handle returned from dlopen
  std::shared_ptr<std::mutex> TracingMutex;
  // Mutex to guard PiPlatforms and LastDeviceIds.
  // Note that this is a temporary solution until we implement the global
  // Device/Platform cache later.
  std::shared_ptr<std::mutex> MPluginMutex;
  // vector of PiPlatforms that belong to this plugin
  std::vector<RT::PiPlatform> PiPlatforms;
  // represents the unique ids of the last device of each platform
  // index of this vector corresponds to the index in PiPlatforms vector.
  std::vector<int> LastDeviceIds;
  std::map<int, std::string> PiApiKindString = {
    {0, "piPlatformsGet"},
    {1, "piPlatformGetInfo"},
    {2, "piextPlatformGetNativeHandle"},
    {3, "piextPlatformCreateWithNativeHandle"},
    {4, "piDevicesGet"},
    {5, "piDeviceGetInfo"},
    {6, "piDevicePartition"},
    {7, "piDeviceRetain"},
    {8, "piDeviceRelease"},
    {9, "piextDeviceSelectBinary"},
    {10, "piextGetDeviceFunctionPointer"},
    {11, "piextDeviceGetNativeHandle"},
    {12, "piextDeviceCreateWithNativeHandle"},
    {13, "piContextCreate"},
    {14, "piContextGetInfo"},
    {15, "piContextRetain"},
    {16, "piContextRelease"},
    {17, "piextContextSetExtendedDeleter"},
    {18, "piextContextGetNativeHandle"},
    {19, "piextContextCreateWithNativeHandle"},
    {20, "piQueueCreate"},
    {21, "piextQueueCreate"},
    {22, "piQueueGetInfo"},
    {23, "piQueueFinish"},
    {24, "piQueueFlush"},
    {25, "piQueueRetain"},
    {26, "piQueueRelease"},
    {27, "piextQueueGetNativeHandle"},
    {28, "piextQueueCreateWithNativeHandle"},
    {29, "piMemBufferCreate"},
    {30, "piMemImageCreate"},
    {31, "piMemGetInfo"},
    {32, "piMemImageGetInfo"},
    {33, "piMemRetain"},
    {34, "piMemRelease"},
    {35, "piMemBufferPartition"},
    {36, "piextMemGetNativeHandle"},
    {37, "piextMemCreateWithNativeHandle"},
    {38, "piProgramCreate"},
    {39, "piclProgramCreateWithSource"},
    {40, "piProgramCreateWithBinary"},
    {41, "piProgramGetInfo"},
    {42, "piProgramCompile"},
    {43, "piProgramBuild"},
    {44, "piProgramLink"},
    {45, "piProgramGetBuildInfo"},
    {46, "piProgramRetain"},
    {47, "piProgramRelease"},
    {48, "piextProgramSetSpecializationConstant"},
    {49, "piextProgramGetNativeHandle"},
    {50, "piextProgramCreateWithNativeHandle"},
    {51, "piKernelCreate"},
    {52, "piKernelSetArg"},
    {53, "piKernelGetInfo"},
    {54, "piKernelGetGroupInfo"},
    {55, "piKernelGetSubGroupInfo"},
    {56, "piKernelRetain"},
    {57, "piKernelRelease"},
    {58, "piextKernelSetArgPointer"},
    {59, "piKernelSetExecInfo"},
    {60, "piextKernelCreateWithNativeHandle"},
    {61, "piextKernelGetNativeHandle"},
    {62, "piEventCreate"},
    {63, "piEventGetInfo"},
    {64, "piEventGetProfilingInfo"},
    {65, "piEventsWait"},
    {66, "piEventSetCallback"},
    {67, "piEventSetStatus"},
    {68, "piEventRetain"},
    {69, "piEventRelease"},
    {70, "piextEventGetNativeHandle"},
    {71, "piextEventCreateWithNativeHandle"},
    {72, "piSamplerCreate"},
    {73, "piSamplerGetInfo"},
    {74, "piSamplerRetain"},
    {75, "piSamplerRelease"},
    {76, "piEnqueueKernelLaunch"},
    {77, "piEnqueueNativeKernel"},
    {78, "piEnqueueEventsWait"},
    {79, "piEnqueueEventsWaitWithBarrier"},
    {80, "piEnqueueMemBufferRead"},
    {81, "piEnqueueMemBufferReadRect"},
    {82, "piEnqueueMemBufferWrite"},
    {83, "piEnqueueMemBufferWriteRect"},
    {84, "piEnqueueMemBufferCopy"},
    {85, "piEnqueueMemBufferCopyRect"},
    {86, "piEnqueueMemBufferFill"},
    {87, "piEnqueueMemImageRead"},
    {88, "piEnqueueMemImageWrite"},
    {89, "piEnqueueMemImageCopy"},
    {90, "piEnqueueMemImageFill"},
    {91, "piEnqueueMemBufferMap"},
    {92, "piEnqueueMemUnmap"},
    {93, "piextUSMHostAlloc"},
    {94, "piextUSMDeviceAlloc"},
    {95, "piextUSMSharedAlloc"},
    {96, "piextUSMFree"},
    {97, "piextUSMEnqueueMemset"},
    {98, "piextUSMEnqueueMemcpy"},
    {99, "piextUSMEnqueuePrefetch"},
    {100, "piextUSMEnqueueMemAdvise"},
    {101, "piextUSMGetMemAllocInfo"},
    {102, "piextKernelSetArgMemObj"},
    {103, "piextKernelSetArgSampler"},
    {104, "piextPluginGetOpaqueData"},
    {105, "piPluginGetLastError"},
    {106, "piTearDown"},
    {107, "piextUSMEnqueueFill2D"},
    {108, "piextUSMEnqueueMemset2D"},
    {109, "piextUSMEnqueueMemcpy2D"}
  };
}; // class plugin
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
