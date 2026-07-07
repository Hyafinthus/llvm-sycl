#pragma once

// #define PRINT_CREATE 1 // pi_opencl.cpp中create相关
// #define PRINT_ELSE 1 // pi_opencl.cpp中除create以外调用
// #define PRINT_PI 1 // plugin.hpp中所有picall
// #define PRINT_KERNEL 1 // commands.cpp中kernel所有参数
// #define PRINT_TRACE 1 // 整个运行时trace
// #define MODIFY 1 // DAG和MPI相关 暂时弃用
// #define CUDA_RT 1 // pi_cuda.cpp中调用cuda运行时
// #define PRINT_DAEMON_TRACE 1 // daemon.cpp调度/通信trace
// #define PRINT_HANDLER_TRACE 1 // handler.cpp离线调度trace
// #define PRINT_DAG_TRACE 1 // scheduler/graph_builder/commands DAG trace

#include <iostream>

struct OfflineTraceNullStream {
  template <typename T>
  const OfflineTraceNullStream &operator<<(const T &) const noexcept {
    return *this;
  }

  const OfflineTraceNullStream &
  operator<<(std::ostream &(*)(std::ostream &)) const noexcept {
    return *this;
  }

  const OfflineTraceNullStream &
  operator<<(std::ios_base &(*)(std::ios_base &)) const noexcept {
    return *this;
  }
};

static constexpr OfflineTraceNullStream OFFLINE_TRACE_NULL_STREAM{};

#ifdef PRINT_DAEMON_TRACE
#define DAEMON_TRACE_STREAM std::cout
#else
#define DAEMON_TRACE_STREAM OFFLINE_TRACE_NULL_STREAM
#endif

#ifdef PRINT_HANDLER_TRACE
#define HANDLER_TRACE_STREAM std::cout
#else
#define HANDLER_TRACE_STREAM OFFLINE_TRACE_NULL_STREAM
#endif

#ifdef PRINT_DAG_TRACE
#define DAG_TRACE_STREAM std::cout
#else
#define DAG_TRACE_STREAM OFFLINE_TRACE_NULL_STREAM
#endif

#define REBIND 1 // 重绑定queue与device
// #define SCHEDULE 1 // 重绑定的调度决策
#define SCHEDULE_OFFLINE 1 // 延迟提交kernel的离线调度

#define SNMD_OFFLINE 1 // 单节点多设备的数据并行 且kernel接入先wait后调度的离线逻辑
