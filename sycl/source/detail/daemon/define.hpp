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

#ifndef SNMD_OFFLINE_SPLIT_DEFAULT_ENABLED
#define SNMD_OFFLINE_SPLIT_DEFAULT_ENABLED 1
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

// P1: Ordinary Split exposes one complete version on a deterministic merge
// device. An explicitly contracted persistent Split may instead expose the
// ordered set of part devices until an incompatible edge or user fence forces
// that same canonical fallback.
#define SNMD_OFFLINE_CANONICAL_MERGE 1

// P2: Permit one bounded cold Split mode only for a long, narrow-DAG kernel
// whose modeled end-to-end gain is substantial. Short/uncertain work remains
// single-device; later windows use measured single/Split profiles.
#define SNMD_OFFLINE_COLD_SPLIT_PROBE 1
#define SNMD_OFFLINE_COLD_SPLIT_MIN_SINGLE_COST 500000
#define SNMD_OFFLINE_COLD_SPLIT_MIN_GAIN_PERCENT 30
#if SNMD_OFFLINE_COLD_SPLIT_MIN_SINGLE_COST < 0
#error "SNMD_OFFLINE_COLD_SPLIT_MIN_SINGLE_COST must be nonnegative"
#endif
#if SNMD_OFFLINE_COLD_SPLIT_MIN_GAIN_PERCENT < 0 ||                       \
    SNMD_OFFLINE_COLD_SPLIT_MIN_GAIN_PERCENT >= 100
#error "SNMD_OFFLINE_COLD_SPLIT_MIN_GAIN_PERCENT must be in [0, 100)"
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

// Unified risk objective. Every feasible Single/Split placement produces the
// same (mean, predictive uncertainty) estimate. The scheduler minimizes its
// upper confidence bound after monitor scaling and communication uncertainty;
// no component is allowed to apply an independent placement penalty.
#define SNMD_OFFLINE_UNIFIED_RISK_OBJECTIVE 1
#define SNMD_OFFLINE_PROFILE_PRIOR_ERROR_PERCENT 5
#define SNMD_OFFLINE_SCALED_PROFILE_ERROR_PERCENT 20
#define SNMD_OFFLINE_COLD_MODEL_ERROR_PERCENT 50
#define SNMD_OFFLINE_DERIVED_SPLIT_ERROR_PERCENT 30
#define SNMD_OFFLINE_TRANSFER_ERROR_PERCENT 15
#define SNMD_OFFLINE_RISK_CONFIDENCE_PERCENT 196
#define SNMD_OFFLINE_PERSISTED_PROFILE_ERROR_PERCENT 25
#define SNMD_OFFLINE_LEARNED_IDENTITY_ERROR_PERCENT 20
#define SNMD_OFFLINE_LEARNED_STRUCTURAL_ERROR_PERCENT 60
#define SNMD_OFFLINE_PROFILE_STORE_MAX_AGE_DAYS 90
#define SNMD_OFFLINE_PROFILE_STORE_QUEUE_LIMIT 4096
#define SNMD_OFFLINE_PROFILE_STORE_DEFAULT_ENABLED 1
#if SNMD_OFFLINE_PROFILE_PRIOR_ERROR_PERCENT < 0 ||                       \
    SNMD_OFFLINE_PROFILE_PRIOR_ERROR_PERCENT >= 100
#error "SNMD_OFFLINE_PROFILE_PRIOR_ERROR_PERCENT must be in [0, 100)"
#endif
#if SNMD_OFFLINE_SCALED_PROFILE_ERROR_PERCENT < 0 ||                       \
    SNMD_OFFLINE_SCALED_PROFILE_ERROR_PERCENT >= 100 ||                    \
    SNMD_OFFLINE_COLD_MODEL_ERROR_PERCENT < 0 ||                           \
    SNMD_OFFLINE_COLD_MODEL_ERROR_PERCENT >= 100 ||                        \
    SNMD_OFFLINE_DERIVED_SPLIT_ERROR_PERCENT < 0 ||                        \
    SNMD_OFFLINE_DERIVED_SPLIT_ERROR_PERCENT >= 100 ||                     \
    SNMD_OFFLINE_TRANSFER_ERROR_PERCENT < 0 ||                             \
    SNMD_OFFLINE_TRANSFER_ERROR_PERCENT >= 100 ||                          \
    SNMD_OFFLINE_PERSISTED_PROFILE_ERROR_PERCENT < 0 ||                    \
    SNMD_OFFLINE_PERSISTED_PROFILE_ERROR_PERCENT >= 100 ||                 \
    SNMD_OFFLINE_LEARNED_IDENTITY_ERROR_PERCENT < 0 ||                     \
    SNMD_OFFLINE_LEARNED_IDENTITY_ERROR_PERCENT >= 100 ||                  \
    SNMD_OFFLINE_LEARNED_STRUCTURAL_ERROR_PERCENT < 0 ||                   \
    SNMD_OFFLINE_LEARNED_STRUCTURAL_ERROR_PERCENT >= 100
#error "SNMD offline uncertainty percentages must be in [0, 100)"
#endif
#if SNMD_OFFLINE_RISK_CONFIDENCE_PERCENT < 0
#error "SNMD_OFFLINE_RISK_CONFIDENCE_PERCENT must be nonnegative"
#endif
#if SNMD_OFFLINE_PROFILE_STORE_MAX_AGE_DAYS <= 0 ||                         \
    SNMD_OFFLINE_PROFILE_STORE_QUEUE_LIMIT <= 0
#error "SNMD offline profile-store limits must be positive"
#endif

// Completion-driven dispatch is currently enabled only for one daemon rank.
// The handler reports actual completions and the daemon releases gang/device
// reservations before selecting the next ready task. Multi-rank keeps the
// batch-static fallback until remote completion acknowledgement is available.
#define SNMD_OFFLINE_COMPLETION_DRIVEN_QUEUE 1

// Split is enabled by default. Set SYCL_SNMD_ENABLE_SPLIT=0 for a runtime
// fallback, or define SNMD_OFFLINE_TEST_DISABLE_SPLIT as a compile-time hard
// stop for correctness isolation.
// #define SNMD_OFFLINE_TEST_DISABLE_SPLIT 1

// P5 diagnostics: uncomment for one daemon decision summary and one handler
// data-movement summary per wait window. Keep disabled for formal timing.
// #define SNMD_OFFLINE_SPLIT_STATS 1

#endif
