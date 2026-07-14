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

// ==== SNMD offline performance fixes and isolated test switches ====
// Each switch is intentionally independent so Reactive and later miniapps can
// run an ablation without changing the Split execution/merge implementation.
#if defined(SCHEDULE_OFFLINE) && defined(SNMD_OFFLINE)

// P1: A Split producer exposes one complete version on a deterministic merge
// device. The daemon and handler must use the same source-of-truth device.
#define SNMD_OFFLINE_CANONICAL_MERGE 1

// P2: Collect a real num_parts=1 profile before allowing a Split probe.
#define SNMD_OFFLINE_SINGLE_FIRST 1

// P2: Once both profiles exist, retain Split only when its end-to-end profile
// beats the single-device profile by at least the configured percentage.
#define SNMD_OFFLINE_SPLIT_HYSTERESIS 1
#define SNMD_OFFLINE_SPLIT_MIN_GAIN_PERCENT 15
#if SNMD_OFFLINE_SPLIT_MIN_GAIN_PERCENT < 0 ||                             \
    SNMD_OFFLINE_SPLIT_MIN_GAIN_PERCENT >= 100
#error "SNMD_OFFLINE_SPLIT_MIN_GAIN_PERCENT must be in [0, 100)"
#endif

// P3: When a complete DAG depth already has enough independent tasks to fill
// the GPUs, prefer task parallelism. A measured superlinear-throughput Split
// can still pass the guard.
#define SNMD_OFFLINE_WIDE_DAG_GUARD 1
#define SNMD_OFFLINE_SPLIT_THROUGHPUT_MARGIN_PERCENT 15
#if SNMD_OFFLINE_SPLIT_THROUGHPUT_MARGIN_PERCENT < 0 ||                    \
    SNMD_OFFLINE_SPLIT_THROUGHPUT_MARGIN_PERCENT >= 100
#error "SNMD_OFFLINE_SPLIT_THROUGHPUT_MARGIN_PERCENT must be in [0, 100)"
#endif

// P0 control: uncomment to retain offline HEFT/dual-GPU placement and disable
// only num_parts>1 candidates.
#define SNMD_OFFLINE_TEST_DISABLE_SPLIT 1

// P5 diagnostics: uncomment for one daemon decision summary and one handler
// data-movement summary per wait window. Keep disabled for formal timing.
// #define SNMD_OFFLINE_SPLIT_STATS 1

#endif
