#include <iostream>
#include <fstream>
#include <iomanip>
#include <signal.h>
#include <unistd.h>
#include <mqueue.h>
#include <fcntl.h>
#include <pthread.h>
#include <mpi.h>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <deque>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <cuda_runtime_api.h>
#include <nvml.h>

#include "daemon.hpp"
#include "define.hpp"
#include <sycl/backend.hpp>
#include <sycl/device.hpp>
// #include <sycl/access/access.hpp>

volatile bool is_interrupted = false;

// signal pthread MPI mq shmem

// 一个SYCLAPP的全局信息
int global_syclapp_count = 0; // 对于整个集群的SYCLAPP计数 因都会在Submit的Bcast前阻塞 每个节点的计数保持相等
std::map<int, std::vector<std::string>> globalcount_to_submit_args; // SYCLAPP计数_启动参数(argv)
std::map<int, pid_t> globalcount_to_pid; // SYCLAPP计数_每个进程上不同的pid
std::map<pid_t, ProgramInfo> pid_to_program; // 当前rank上的pid_整个SYCLAPP的进程信息
// [master]
std::map<int, std::set<int>> globalcount_to_onrun; // globalcount_在哪些rank上运行
// [非master]
std::map<int, int> globalcount_to_scalecount; // globalcount_对于本rank从哪个kernel开始
// [master] daemon向scale传递信息
std::unordered_map<pid_t, std::shared_ptr<std::queue<std::pair<int, int>>>> pid_to_scalecount_queue; // pid_扩容kernelcount_执行rank
std::unordered_map<pid_t, std::shared_ptr<std::mutex>> pid_to_scalecount_mutex; // pid_扩容kernelcount_mutex
std::unordered_map<pid_t, std::shared_ptr<std::condition_variable>> pid_to_scalecount_cv; // pid_扩容kernelcount_cv

// ====【Monitor】
static constexpr int MAX_MONITOR_DEVICES = 16;
static constexpr int MONITOR_PACKED_FIELDS = 5; // valid, util, mem, fp32, fp64
struct ComputeCapability {
  double fp32 = 1.0;
  double fp64 = 0.5;
};
std::map<int, int> index_sycl_nvml; // 根据busid确定sycl::device到gpu映射
std::map<int, int> index_nvml_sycl;
std::map<int, std::string> index_sycl_device_identity;
std::vector<MonitorInfo> device_monitor_info(1); // 每个设备的监控信息, 0号设备固定是CPU
std::vector<ComputeCapability> device_capability(1); // 本rank设备原始能力, 编号与handler的globalDevices一致
std::vector<std::vector<MonitorInfo>> cluster_monitor_info;
std::vector<std::vector<ComputeCapability>> cluster_device_capability;
std::vector<int> cluster_comm_profile_ids;
std::vector<int> ranks_idle;
std::mutex monitor_state_mutex;

// ====【Algorithm】
// Each application daemon thread owns one scheduling context. Shared monitor
// and profile tables are sampled into these thread-local vectors; keeping the
// mutable HEFT/completion calendar process-global would either race between
// applications or require holding a global lock for the full kernel window.
thread_local std::vector<std::vector<ComputeCapability>> gpu_capability;
thread_local std::vector<std::vector<double>> gpu_available_time;
thread_local std::vector<std::vector<double>> gpu_service_time_scale;
thread_local std::vector<std::vector<double>> gpu_memory_available_kib;
thread_local std::vector<int> gpu_comm_profile_ids;
int local_comm_profile_id = -1;

static std::vector<MonitorInfo> snapshotLocalMonitorInfo() {
  std::lock_guard<std::mutex> lock(monitor_state_mutex);
  return device_monitor_info;
}

static std::vector<int> snapshotRankIdleState() {
  std::lock_guard<std::mutex> lock(monitor_state_mutex);
  return ranks_idle;
}

static std::vector<std::string> DeserializeSubmitArgs(const char *buffer,
                                                      size_t buffer_size) {
  if (buffer_size < sizeof(uint32_t)) {
    throw std::runtime_error("submit payload is too small");
  }

  uint32_t arg_count = 0;
  std::memcpy(&arg_count, buffer, sizeof(arg_count));
  if (arg_count == 0) {
    throw std::runtime_error("submit payload has no binary path");
  }

  std::vector<std::string> args;
  args.reserve(arg_count);
  size_t offset = sizeof(arg_count);
  for (uint32_t i = 0; i < arg_count; ++i) {
    if (offset >= buffer_size) {
      throw std::runtime_error("submit payload ended before all args");
    }

    const void *arg_end = std::memchr(buffer + offset, '\0',
                                     buffer_size - offset);
    if (arg_end == nullptr) {
      throw std::runtime_error("submit payload contains an unterminated arg");
    }

    const char *arg_end_char = static_cast<const char *>(arg_end);
    args.emplace_back(buffer + offset, arg_end_char - (buffer + offset));
    offset += args.back().size() + 1;
  }

  if (args[0].empty()) {
    throw std::runtime_error("submit binary path is empty");
  }

  return args;
}

static std::vector<char *> BuildExecArgv(const std::vector<std::string> &args) {
  std::vector<char *> exec_argv;
  exec_argv.reserve(args.size() + 1);
  for (const std::string &arg : args) {
    exec_argv.push_back(const_cast<char *>(arg.c_str()));
  }
  exec_argv.push_back(nullptr);
  return exec_argv;
}

static std::string JoinSubmitArgs(const std::vector<std::string> &args) {
  std::ostringstream oss;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i != 0) {
      oss << " ";
    }
    oss << args[i];
  }
  return oss.str();
}

struct ProfileCostKey {
  std::string kernel_key;
  int rank = 0;
  int device = 0;
  std::string device_identity;
  int num_parts = 1;
  bool persistent_split = false;

  bool operator<(const ProfileCostKey &other) const {
    return std::tie(kernel_key, rank, device_identity, num_parts,
                    persistent_split) <
           std::tie(other.kernel_key, other.rank, other.device_identity,
                    other.num_parts, other.persistent_split);
  }
};

struct ProfileCostEntry {
  double ewma_cost = 0.0;
  double mean_cost = 0.0;
  double m2_cost = 0.0;
  double min_cost = std::numeric_limits<double>::infinity();
  double source_fp32_capability = 0.0;
  double source_fp64_capability = 0.0;
  uint64_t last_observed_unix_sec = 0;
  int samples = 0;
  int live_samples = 0;
};

struct KernelFeatureSignature {
  uint64_t kernel_identity = 0;
  int work_dim = 0;
  double global_items = 1.0;
  double total_access_elems = 1.0;
  double read_bytes = 0.0;
  double write_bytes = 0.0;
  double analytical_work = 1.0;
  int req_count = 0;
  int read_req_count = 0;
  int write_req_count = 0;
  int dominant_elem_size = 0;
  uint32_t access_mode_mask = 0;
  bool partition_local_read = false;
  bool partition_local_write = false;
};

struct PersistedProfileObservation {
  std::string profile_namespace;
  ProfileCostKey key;
  double sample_cost = 0.0;
  double source_fp32_capability = 0.0;
  double source_fp64_capability = 0.0;
  uint64_t observed_unix_sec = 0;
  bool feature_valid = false;
  KernelFeatureSignature feature;
};

std::map<ProfileCostKey, ProfileCostEntry> profile_cost_table;
std::map<std::string, KernelFeatureSignature> kernel_feature_table;
std::map<std::tuple<pid_t, int, int>, uint64_t> profile_device_update_ns;
std::mutex profile_cost_mutex;

std::mutex profile_store_mutex;
std::condition_variable profile_store_cv;
std::deque<PersistedProfileObservation> profile_store_queue;
std::thread profile_store_thread;
std::string profile_store_path;
std::string profile_store_namespace = "default";
bool profile_store_enabled = false;
bool profile_store_stop = false;
uint64_t profile_store_dropped_records = 0;

// ====【MPI】
int mpi_rank, mpi_size; // main
MPI_Comm comm_submit; // SystemSchedulerSubmit
int submit_rank, submit_size; // SystemSchedulerSubmit
MPI_Comm comm_monitor; // SystemSchedulerMonitor
int monitor_rank, monitor_size; // SystemSchedulerMonitor

// ====【mq】
mqd_t mq_id_submit;

// DISCARD MPI_THREAD_MULTIPLE会劫持SIGINT
// SIG_BLOCK和sigwait和export OMPI_MCA_mpi_signal=0都不行
// void SignalHandler(int signum) {
//   if (signum == SIGINT) {
//     std::cout << "Interrupted!" << std::endl;
//     is_interrupted = true;
//   }
// }

mqd_t EstablishDaemon(pid_t pid) {
  struct mq_attr mq_attr;
  mq_attr.mq_flags = 0;
  mq_attr.mq_maxmsg = MAX_MSG_NUM;
  mq_attr.mq_msgsize = MAX_MSG_DAEMON_SIZE;

  char MESSAGE_QUEUE_DAEMON_NAME[MESSAGE_QUEUE_DAEMON_NAME_MAX];
  sprintf(MESSAGE_QUEUE_DAEMON_NAME, MESSAGE_QUEUE_DAEMON_PATTERN, pid);
  mqd_t mq_id_daemon = mq_open(MESSAGE_QUEUE_DAEMON_NAME, O_CREAT | O_RDONLY, 0666, &mq_attr);
  if (mq_id_daemon == -1) {
    std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " mq_id_daemon open failed";
    perror(errorMsg.c_str());
    exit(1);
  }

  DAEMON_TRACE_STREAM << "EstablishDaemon: Rank " << mpi_rank << " created mq_id_daemon: " << mq_id_daemon << std::endl;

  return mq_id_daemon;
}

void EstablishSubmit() {
  struct mq_attr mq_attr;
  mq_attr.mq_flags = 0;
  mq_attr.mq_maxmsg = MAX_MSG_NUM;
  mq_attr.mq_msgsize = MAX_MSG_SUBMIT_SIZE;

  mq_id_submit = mq_open(MESSAGE_QUEUE_SUBMIT_NAME, O_CREAT | O_RDONLY, 0666, &mq_attr);
  if (mq_id_submit == -1) {
    std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " mq_id_submit open failed";
    perror(errorMsg.c_str());
    exit(1);
  }
}

// DISCARD 现在在program_manager发送端关闭
// void CloseDaemon(pid_t pid) {
//   mq_close(mq_id_daemon);
//   char MESSAGE_QUEUE_DAEMON_NAME[MESSAGE_QUEUE_DAEMON_NAME_MAX];
//   sprintf(MESSAGE_QUEUE_DAEMON_NAME, MESSAGE_QUEUE_DAEMON_PATTERN, pid);
//   mq_unlink(MESSAGE_QUEUE_DAEMON_NAME);
// }

void CloseSubmit() {
  mq_close(mq_id_submit);
  mq_unlink(MESSAGE_QUEUE_SUBMIT_NAME);
}

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

  DAEMON_TRACE_STREAM << Description << " multipart mq_send payload_size: "
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

  DAEMON_TRACE_STREAM << Description << " multipart mq_receive payload_size: "
                      << Payload.size() << " chunks: " << ChunkCount
                      << std::endl;
  return Payload;
}

void SendD2DKernelSchedInfo(MPI_Comm comm_daemon, int master_rank, int daemon_rank, const std::set<int>& onrun_ranks, D2DKernelSchedInfo& kernel_sched_info) {
  if (daemon_rank == master_rank) {
    std::string serialized_data = kernel_sched_info.serialize();
    int str_length = static_cast<int>(serialized_data.size());
    for (int rank : onrun_ranks) {
      if (rank != master_rank) {
        MPI_Send(&str_length, 1, MPI_INT, rank, 0, comm_daemon);
        DAEMON_TRACE_STREAM << "SendD2DKernelSchedInfo: Rank " << daemon_rank << " to Rank " << rank << " with str_length:" << str_length << std::endl;
        MPI_Send(serialized_data.c_str(), str_length, MPI_CHAR, rank, 0, comm_daemon);
        DAEMON_TRACE_STREAM << "SendD2DKernelSchedInfo: Rank " << daemon_rank << " to Rank " << rank << " with serialized_data" << std::endl;
      }
    }
  } else {
    int str_length;
    MPI_Recv(&str_length, 1, MPI_INT, master_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
    DAEMON_TRACE_STREAM << "SendD2DKernelSchedInfo: Rank " << daemon_rank << " received str_length:" << str_length << std::endl;

    char* buffer = new char[str_length + 1];
    MPI_Recv(buffer, str_length, MPI_CHAR, master_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
    buffer[str_length] = '\0';

    std::string serialized_data(buffer);
    kernel_sched_info = D2DKernelSchedInfo::deserialize(serialized_data);
    DAEMON_TRACE_STREAM << "SendD2DKernelSchedInfo: Rank " << daemon_rank << " received serialized_data" << std::endl;

    delete[] buffer;
  }
}

// OFFLINE ONLY
void SendD2DKernelSchedInfos(MPI_Comm comm_daemon, int master_rank, int daemon_rank, const std::set<int>& onrun_ranks, std::vector<D2DKernelSchedInfo>& kernel_sched_order_infos) {
  if (daemon_rank == master_rank) {
    std::string serialized_data;
    for (const auto &info : kernel_sched_order_infos) {
        serialized_data += info.serialize();
    }
    int str_length = static_cast<int>(serialized_data.size());

    for (int rank : onrun_ranks) {
      if (rank != master_rank) {
        MPI_Send(&str_length, 1, MPI_INT, rank, 0, comm_daemon);
        DAEMON_TRACE_STREAM << "SendD2DKernelSchedInfos: Rank " << daemon_rank << " to Rank " << rank << " with str_length:" << str_length << std::endl;

        MPI_Send(serialized_data.c_str(), str_length, MPI_CHAR, rank, 0, comm_daemon);
        DAEMON_TRACE_STREAM << "SendD2DKernelSchedInfos: Rank " << daemon_rank << " to Rank " << rank << " with serialized_data" << std::endl;
      }
    }
  } else {
    int str_length;
    MPI_Recv(&str_length, 1, MPI_INT, master_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
    DAEMON_TRACE_STREAM << "SendD2DKernelSchedInfos: Rank " << daemon_rank << " received str_length:" << str_length << std::endl;

    char* buffer = new char[str_length + 1];
    MPI_Recv(buffer, str_length, MPI_CHAR, master_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
    buffer[str_length] = '\0';

    std::string received_data(buffer);
    delete[] buffer;

    std::vector<D2DKernelSchedInfo> received_sched_infos;
    std::istringstream stream(received_data);
    std::string line;

    while (std::getline(stream, line)) {
      std::string obj_data = line + "\n";  // kernel_count
      std::getline(stream, line);
      obj_data += line + "\n";  // exec_order
      std::getline(stream, line);
      obj_data += line + "\n";  // exec_rank
      std::getline(stream, line);
      obj_data += line + "\n";  // exec_device
      std::getline(stream, line);
      obj_data += line + "\n";  // num_parts
      std::getline(stream, line);
      obj_data += line + "\n";  // persistent_split
      std::getline(stream, line);
      obj_data += line + "\n";  // split_devices.size()
      int split_device_count = std::stoi(line);
      for (int i = 0; i < split_device_count; ++i) {
        std::getline(stream, line);
        obj_data += line + "\n";  // split device index
      }
      std::getline(stream, line);
      obj_data += line + "\n";  // req_rank.size()
      int map_size = std::stoi(line);
      for (int i = 0; i < map_size; ++i) {
        for (int j = 0; j < SYCL_REQ_DATA_SERIALIZED_LINES; ++j) {
          std::getline(stream, line);
          obj_data += line + "\n";  // SyclReqData
        }
        std::getline(stream, line);
        obj_data += line + "\n";  // int: rank
      }

      received_sched_infos.push_back(D2DKernelSchedInfo::deserialize(obj_data));
    }

    DAEMON_TRACE_STREAM << "SendD2DKernelSchedInfos: Rank " << daemon_rank << " received " << received_sched_infos.size() << " kernel_sched_order_infos" << std::endl;
    kernel_sched_order_infos = std::move(received_sched_infos);
  }
}

void BcastD2DKernelSchedInfo(MPI_Comm comm_daemon, int master_rank, int daemon_rank, D2DKernelSchedInfo &kernel_sched_info) {
  std::string serialized_data;
  if (daemon_rank == master_rank) {
    serialized_data = kernel_sched_info.serialize();
  }

  size_t str_length = serialized_data.size();
  MPI_Bcast(&str_length, 1, MPI_INT, master_rank, comm_daemon);

  char *buffer = new char[str_length + 1];
  if (daemon_rank == master_rank) {
    std::copy(serialized_data.begin(), serialized_data.end(), buffer);
    buffer[str_length] = '\0';
  }

  MPI_Bcast(buffer, str_length + 1, MPI_CHAR, master_rank, comm_daemon);

  if (daemon_rank != master_rank) {
    serialized_data = std::string(buffer);
    kernel_sched_info = D2DKernelSchedInfo::deserialize(serialized_data);
  }

  delete[] buffer;
}

// DISCARD Submit确定执行的ranks
int master_rank_syclapp(const std::vector<int>& exec_flags) {
  std::vector<int> non_zero_index;
  for (int i = 0; i < exec_flags.size(); i++) {
    if (exec_flags[i] == 2) {
      return non_zero_index.size();
    } else if (exec_flags[i] != 0) {
      non_zero_index.push_back(i);
    }
  }
  return -1;
}

// DISCARD Submit确定执行的ranks
std::map<int, int> map_rank_syclapp_submit(const std::vector<int>& exec_flags) {
  std::map<int, int> rank_syclapp_to_submit;  
  for (int i = 0; i < exec_flags.size(); i++) {
    if (exec_flags[i] != 0) {
      int syclapp_rank = rank_syclapp_to_submit.size();
      int submit_rank = i;
      rank_syclapp_to_submit[syclapp_rank] = submit_rank;
    }
  }
  return rank_syclapp_to_submit;
}

// ========【Online Start】

std::map<SyclReqData, std::set<int>> generateDAG(std::vector<DAGNode *> &kernel_dag_nodes, DAGNode *node) {
  // 一个kernel内的依赖关系对分析没有影响 重点在于kernel间相同mem的依赖关系 (在保证执行顺序的前提下)
  // read: 必然无依赖 - 依赖前序kernel相同mem的写
  // write: 可能读其他mem (仍需要加载到设备内存) - 不依赖前序kernel (依赖的mem已在read中)
  // read_write: 更新自身/先读后写=更新自身/先写后读=初始化后读 - 依赖前序kernel相同mem的写
  // discard_write: 丢弃之前并初始化 - 不依赖前序kernel
  // discard_read_write: 丢弃之前并读写=先初始化后读 - 不依赖前序kernel
  // atomic: 单线程执行=视作读写 - 依赖前序kernel相同mem的写

  // **注意** SyclReqData相memptr会视作同一个对象 单kernelcount和reqcount不同 必须使用当前kernel生成的对象 后续会用到两个count
  std::map<SyclReqData, std::set<int>> req_ranks;

  // 【优化】因需要读一个数组会从执行的rank拷贝 所以数据最新拷贝可能在多个rank存在
  // write: 在最近被写过的kernel上必然是最新的
  // read: 同时在最近被写 之后读过的所有rank 因会从写处拷回host并通信传输 也是最新的
  for (SyclReqData &req : node->req_data) {
    // 依赖前序kernel相同mem的写 read | read_write | atomic
    if (req.req_accmode == acc_mode::read || req.req_accmode == acc_mode::read_write || req.req_accmode == acc_mode::atomic) {
      // 由后向前遍历kernel 找到相同mem最近的写作为依赖
      bool found_write = false;
      for (auto it = kernel_dag_nodes.rbegin(); it != kernel_dag_nodes.rend(); ++it) {
        DAGNode *prev_node = *it;
        for (SyclReqData &prev_req : prev_node->req_data) {
          if (prev_req.mem_pointer == req.mem_pointer) {
            if (prev_req.req_accmode != acc_mode::read) {
              node->depend_on.push_back(prev_node);
              prev_node->depend_by.push_back(node);
              node->depth = std::max(node->depth, prev_node->depth + 1);

              req_ranks[req].insert(prev_node->exec_rank);
              found_write = true;
              break;
            }
            else { //【优化】写后被读的rank可以作为依赖
              req_ranks[req].insert(prev_node->exec_rank);
            }
          }
        }
        if (found_write) {
          break;
        }
      }
    }
  }
  // DISCARD 【被优化】写后被读的rank可以作为依赖
  // std::map<SyclReqData, int> req_rank;
  // for (auto it = kernel_dag_nodes.rbegin(); it != kernel_dag_nodes.rend(); ++it) {
  //   DAGNode *prev_node = *it;
  //   bool found = false;
  //   for (SyclReqData &prev_req : prev_node->req_data) {
  //     std::cout << "generateDAG: Rank " << node->exec_rank << " prev_req_mem_pointer: " << prev_req.mem_pointer << " prev_req_accmode: " << static_cast<int>(prev_req.req_accmode) << std::endl;
  //     if (prev_req.mem_pointer == req.mem_pointer && prev_req.req_accmode != acc_mode::read) {
  //       node->depend_on.push_back(prev_node);
  //       prev_node->depend_by.push_back(node);
  //       req_rank[req] = prev_node->exec_rank;
  //       found = true;
  //       break;
  //     }
  //   }
  //   if (found) {
  //     break;
  //   }
  // }

  kernel_dag_nodes.push_back(node);
  return req_ranks;
}

// OPTI 根据数据量判断不同rank所需的总体通信量
int mostDepdRank(std::map<SyclReqData, std::set<int>> &req_ranks) {
  std::map<int, int> rank_count;
  for (auto pair : req_ranks) {
    for (int rank : pair.second) {
      rank_count[rank]++;
    }
  }
  int max_count = 0;
  int max_rank = -1;
  for (auto pair : rank_count) {
    if (pair.second > max_count) {
      max_count = pair.second;
      max_rank = pair.first;
    }
  }
  return max_rank;
}

// OPTI 同一个req在多个rank上有最新时的选择方案 尽量集中还是分散？
std::map<SyclReqData, int> chooseReqRank(std::map<SyclReqData, std::set<int>> &req_ranks, int rank) {
  std::map<SyclReqData, int> req_rank;
  for (auto pair : req_ranks) {
    if (pair.second.find(rank) != pair.second.end()) {
      req_rank[pair.first] = rank;
    }
    // 在集合中随机选一个
    else {
      int rand_index = rand() % pair.second.size();
      auto it = pair.second.begin();
      std::advance(it, rand_index);
      req_rank[pair.first] = *it;
    }
  }
  return req_rank;
}

// ========【Online End】

// ========【Offline Start】

static bool isReadAccess(acc_mode mode) {
  return mode == acc_mode::read || mode == acc_mode::read_write ||
         mode == acc_mode::atomic;
}

static bool isWriteAccess(acc_mode mode) {
  return mode == acc_mode::write || mode == acc_mode::read_write ||
         mode == acc_mode::discard_write ||
         mode == acc_mode::discard_read_write ||
         mode == acc_mode::atomic;
}

static bool containsNode(const std::vector<DAGNode *> &nodes, DAGNode *target) {
  return std::find(nodes.begin(), nodes.end(), target) != nodes.end();
}

static void addDAGDependency(DAGNode *pre_node, DAGNode *node,
                             const SyclReqData &req, bool data_dependency) {
  if (!containsNode(node->depend_on, pre_node)) {
    node->depend_on.push_back(pre_node);
  }
  if (!containsNode(pre_node->depend_by, node)) {
    pre_node->depend_by.push_back(node);
  }

  node->depth = std::max(node->depth, pre_node->depth + 1);

  // Only RAW-like edges move data. WAR/WAW edges are ordering constraints and
  // should not become communication or req_rank dependencies.
  if (data_dependency) {
    node->depend_on_mem[pre_node].insert(req);
    node->depend_on_node[req] = pre_node;
    pre_node->depend_by_mem[node].insert(req);
  }
}

static double getCommBytes(DAGNode *node, DAGNode *pre_node) {
  auto it = node->depend_on_mem.find(pre_node);
  if (it == node->depend_on_mem.end()) {
    return 0.0;
  }

  double bytes = 0.0;
  for (const SyclReqData &req : it->second) {
    bytes += static_cast<double>(req.elem_size) *
             static_cast<double>(req.buff_size);
  }
  return bytes;
}

static std::vector<DAGNode *> reverseTopologicalOrder(
    const std::vector<DAGNode *> &nodes) {
  std::unordered_set<DAGNode *> current_nodes(nodes.begin(), nodes.end());
  std::map<DAGNode *, int> indegree;
  for (DAGNode *node : nodes) {
    indegree[node] = 0;
  }

  for (DAGNode *node : nodes) {
    for (DAGNode *pre_node : node->depend_on) {
      if (current_nodes.count(pre_node)) {
        indegree[node]++;
      }
    }
  }

  std::queue<DAGNode *> ready;
  for (DAGNode *node : nodes) {
    if (indegree[node] == 0) {
      ready.push(node);
    }
  }

  std::vector<DAGNode *> topo;
  while (!ready.empty()) {
    DAGNode *node = ready.front();
    ready.pop();
    topo.push_back(node);

    for (DAGNode *succ_node : node->depend_by) {
      if (!current_nodes.count(succ_node)) {
        continue;
      }
      indegree[succ_node]--;
      if (indegree[succ_node] == 0) {
        ready.push(succ_node);
      }
    }
  }

  if (topo.size() != nodes.size()) {
    std::cerr << "algorithmHEFT: DAG cycle detected or incomplete topo order, "
              << "fallback to input order" << std::endl;
    topo = nodes;
  }

  return std::vector<DAGNode *>(topo.rbegin(), topo.rend());
}

static void regenerateReqRanksAfterHEFT(
    const std::vector<DAGNode *> &nodes,
    std::vector<D2DKernelSchedInfo> &kernel_sched_order_infos) {
  std::map<int, D2DKernelSchedInfo *> kernel_to_sched;
  for (D2DKernelSchedInfo &sched_info : kernel_sched_order_infos) {
    sched_info.req_rank.clear();
    kernel_to_sched[sched_info.kernel_count] = &sched_info;
  }

  for (DAGNode *node : nodes) {
    auto sched_it = kernel_to_sched.find(node->kernel_count);
    if (sched_it == kernel_to_sched.end()) {
      std::cerr << "regenerateReqRanksAfterHEFT: missing sched_info for kernel "
                << node->kernel_count << std::endl;
      continue;
    }

    D2DKernelSchedInfo *sched_info = sched_it->second;
    for (const SyclReqData &req : node->req_data) {
      if (!isReadAccess(req.req_accmode)) {
        continue;
      }

      auto producer_it = node->depend_on_node.find(req);
      if (producer_it == node->depend_on_node.end()) {
        continue;
      }

      DAGNode *producer = producer_it->second;
      sched_info->req_rank[req] = producer->exec_rank;
      // DAEMON_TRACE_STREAM << "regenerateReqRanksAfterHEFT: Kernel "
      //           << node->kernel_count << " req " << req.req_count
      //           << " source rank " << producer->exec_rank
      //           << " from Kernel " << producer->kernel_count << std::endl;
    }
  }
}

// INPUT: 已有的kernel组成的DAG 新的一组wait中所有的kernel
// 仅通过req的mem依赖建立DAG 不涉及req_rank和exec_rank
// 分析见NOTION
void generateDAGs(std::vector<DAGNode *> &kernel_dag_nodes, std::vector<DAGNode *> &nodes) {
  for (DAGNode *node : nodes) {
    for (SyclReqData &req : node->req_data) {
      const bool current_reads = isReadAccess(req.req_accmode);
      const bool current_writes = isWriteAccess(req.req_accmode);
      if (!current_reads && !current_writes) {
        continue;
      }

      // 向后查找同一mem的最近冲突访问：
      // RAW: 当前读依赖最近前序写，是真数据依赖。
      // WAR: 当前写需等待最近前序写之后的所有读，是顺序依赖。
      // WAW: 当前写需等待最近前序写，是顺序依赖。
      bool found_prev_writer = false;
      for (auto it = kernel_dag_nodes.rbegin(); it != kernel_dag_nodes.rend(); ++it) {
        DAGNode *prev_node = *it;
        bool prev_node_has_writer = false;
        bool prev_node_added = false;

        for (SyclReqData &prev_req : prev_node->req_data) {
          if (prev_req.mem_pointer != req.mem_pointer) {
            continue;
          }

          const bool prev_reads = isReadAccess(prev_req.req_accmode);
          const bool prev_writes = isWriteAccess(prev_req.req_accmode);

          if (current_reads && prev_writes) {
            addDAGDependency(prev_node, node, req, /*data_dependency=*/true);
            prev_node_added = true;
            prev_node_has_writer = true;
          }

          if (current_writes && prev_reads && !found_prev_writer) {
            addDAGDependency(prev_node, node, req, /*data_dependency=*/false);
            prev_node_added = true;
          }

          if (current_writes && prev_writes) {
            addDAGDependency(prev_node, node, req, /*data_dependency=*/false);
            prev_node_added = true;
            prev_node_has_writer = true;
          }
        }

        // if (prev_node_added) {
        //   DAEMON_TRACE_STREAM << "generateDAGs: Kernel " << node->kernel_count
        //             << " depends on Kernel " << prev_node->kernel_count
        //             << " for req " << req.req_count << std::endl;
        // }

        if (prev_node_has_writer) {
          found_prev_writer = true;
          break;
        }
      }
    }
    kernel_dag_nodes.push_back(node);
  }
}

static void rebaseCompletedOfflineDAG(
    const std::vector<DAGNode *> &kernel_dag_nodes) {
  // A new offline batch can only be submitted after the preceding user
  // queue::wait returned. Previous nodes remain data producers and residency
  // anchors, but their synthetic HEFT offsets do not belong to this batch's
  // time origin.
  for (DAGNode *node : kernel_dag_nodes) {
    if (node == nullptr) {
      continue;
    }
    node->finish_time = 0.0;
    node->rank_u = 0.0;
    // handler materializes every live partition at the user-visible wait
    // fence. Historical nodes still anchor logical data ownership, but cannot
    // advertise per-device resident partitions into the next wait window.
    node->persistent_split = false;
  }
}

static constexpr double PROFILE_NS_TO_COST = 10000.0;
static constexpr double GIB_BYTES = 1024.0 * 1024.0 * 1024.0;
static constexpr double HEFT_COMM_COST_PER_SECOND = 100000.0;
static constexpr double FALLBACK_HOST_BW_GIB = 12.0;
static constexpr double FALLBACK_H2D_BW_GIB = 20.0;
static constexpr double FALLBACK_D2H_BW_GIB = 20.0;
static constexpr double FALLBACK_D2D_BW_GIB = 12.0;
static constexpr double FALLBACK_CROSS_RANK_BW_GIB = 0.106;
static constexpr double SPLIT_EFFICIENCY = 0.85;
static constexpr double SPLIT_MIN_ELEMS = 65536.0;
// One latency-bound CUDA kernel can require several resident warps per SM to
// reach its profiled throughput.  Treat this many aggregate work-items as one
// GPU's concurrent-residency budget when comparing a co-located out-of-order
// queue with the ordinary exclusive-device HEFT plan.  The runtime override
// is useful for architecture-specific calibration; zero disables the model.
static constexpr double DEFAULT_CONCURRENT_GPU_TARGET_ITEMS = 16384.0;

static double concurrentGpuTargetItems() {
  static const double target_items = [] {
    const char *env = std::getenv("SYCL_SNMD_CONCURRENT_TARGET_ITEMS");
    if (env == nullptr || *env == '\0') {
      return DEFAULT_CONCURRENT_GPU_TARGET_ITEMS;
    }
    char *end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0) {
      return DEFAULT_CONCURRENT_GPU_TARGET_ITEMS;
    }
    return parsed;
  }();
  return target_items;
}

#ifdef SNMD_OFFLINE_COLD_SPLIT_PROBE
static double coldSplitMinSingleCost() {
  static const double min_cost = [] {
    constexpr double default_min_cost =
        SNMD_OFFLINE_COLD_SPLIT_MIN_SINGLE_COST;
    const char *env = std::getenv("SYCL_SNMD_COLD_SPLIT_MIN_SINGLE_COST");
    if (env == nullptr || *env == '\0') {
      return default_min_cost;
    }
    char *end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || *end != '\0' || !std::isfinite(parsed) ||
        parsed < 0.0) {
      return default_min_cost;
    }
    return parsed;
  }();
  return min_cost;
}
#endif

static bool decisionSummaryRuntimeEnabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("SYCL_SNMD_DECISION_SUMMARY");
    return env != nullptr && std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "FALSE") != 0;
  }();
  return enabled;
}

static bool completionQueueRuntimeRequested() {
  static const bool enabled = [] {
    const char *env = std::getenv("SYCL_SNMD_COMPLETION_QUEUE");
    if (env == nullptr) {
      return true;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "FALSE") != 0;
  }();
  return enabled;
}

static bool componentAffineStaticFastPathEnabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("SYCL_SNMD_COMPONENT_AFFINE_STATIC");
    if (env == nullptr) {
      return true;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "FALSE") != 0;
  }();
  return enabled;
}

static size_t componentAffineStaticMinNodesPerComponent() {
  static const size_t min_nodes = [] {
    constexpr size_t default_min_nodes = 64;
    const char *env =
        std::getenv("SYCL_SNMD_COMPONENT_AFFINE_STATIC_MIN_NODES");
    if (env == nullptr || *env == '\0') {
      return default_min_nodes;
    }
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max()) {
      return default_min_nodes;
    }
    return static_cast<size_t>(parsed);
  }();
  return min_nodes;
}

enum class CostEstimateSource {
  cold_model,
  exact_profile,
  persisted_profile,
  scaled_profile,
  learned_profile,
  derived_split
};

struct CostEstimate {
  double mean = std::numeric_limits<double>::infinity();
  double uncertainty = 0.0;
  int samples = 0;
  CostEstimateSource source = CostEstimateSource::cold_model;
};

static const char *costEstimateSourceName(CostEstimateSource source) {
  switch (source) {
  case CostEstimateSource::cold_model:
    return "cold";
  case CostEstimateSource::exact_profile:
    return "exact-profile";
  case CostEstimateSource::persisted_profile:
    return "persisted-profile";
  case CostEstimateSource::scaled_profile:
    return "scaled-profile";
  case CostEstimateSource::learned_profile:
    return "learned-profile";
  case CostEstimateSource::derived_split:
    return "derived-split";
  }
  return "unknown";
}

struct NodeCommProfile {
  int id = -1;
  std::string key;
  double shm_bw_gib = FALLBACK_HOST_BW_GIB;
  double same_node_staged_bw_gib = 4.0;
  std::vector<double> h2d_bw_gib; // index 0 is CPU/host, GPUs start at 1
  std::vector<double> d2h_bw_gib; // index 0 is CPU/host, GPUs start at 1
  std::map<std::pair<int, int>, double> d2d_bw_gib;
};

static const std::map<std::string, NodeCommProfile> &nodeCommProfiles() {
  static const std::map<std::string, NodeCommProfile> profiles = {
      {"4090-01", // profile map key: hostname/env/device detection result
       NodeCommProfile{
          1,         // profile id: the compact value exchanged by MPI_Allgather
          "4090-01", // profile key: used by crossNodeCommProfiles lookup
          18.5,      // shm_bw_gib: test_host_shm_memcpy_bandwidth, large-size host<->shm
          5.5,       // same_node_staged_bw_gib: mpirun -n 2 test_mpi_cuda_staged_bandwidth on 4090 node
          {0.0, 24.0, 22.0}, // h2d_bw_gib: test_cuda_h2d_d2h_bandwidth pinned_H2D, index 0 CPU, 1..N GPU
          {0.0, 25.0, 12.6}, // d2h_bw_gib: test_cuda_h2d_d2h_bandwidth pinned_D2H, index 0 CPU, 1..N GPU
          {// d2d_bw_gib: test_cuda_p2p_bandwidth; daemon proc id, so CUDA0->CUDA1 is 1->2
           {{1, 2}, 20.7}, // GPU proc 1 -> GPU proc 2
           {{2, 1}, 7.5}}}}, // GPU proc 2 -> GPU proc 1
      {"a6000-01", // profile map key: hostname/env/device detection result
       NodeCommProfile{
          2,          // profile id: the compact value exchanged by MPI_Allgather
          "a6000-01", // profile key: used by crossNodeCommProfiles lookup
          12.5,       // shm_bw_gib: test_host_shm_memcpy_bandwidth, large-size host<->shm
          4.5,        // same_node_staged_bw_gib: mpirun -n 2 test_mpi_cuda_staged_bandwidth on a6000 node
          {0.0, 25.1, 25.1, 25.1, 25.0}, // h2d_bw_gib: test_cuda_h2d_d2h_bandwidth pinned_H2D
          {0.0, 24.5, 24.5, 24.5, 23.3}, // d2h_bw_gib: test_cuda_h2d_d2h_bandwidth pinned_D2H
          {// d2d_bw_gib: test_cuda_p2p_bandwidth; daemon proc id, GPU procs are 1..4
           {{1, 2}, 24.58}, // GPU proc 1 -> GPU proc 2
           {{2, 1}, 24.58}, // GPU proc 2 -> GPU proc 1
           {{1, 3}, 21.22}, // GPU proc 1 -> GPU proc 3
           {{3, 1}, 21.22}, // GPU proc 3 -> GPU proc 1
           {{1, 4}, 20.93}, // GPU proc 1 -> GPU proc 4
           {{4, 1}, 21.20}, // GPU proc 4 -> GPU proc 1
           {{2, 3}, 21.22}, // GPU proc 2 -> GPU proc 3
           {{3, 2}, 21.22}, // GPU proc 3 -> GPU proc 2
           {{2, 4}, 20.97}, // GPU proc 2 -> GPU proc 4
           {{4, 2}, 21.15}, // GPU proc 4 -> GPU proc 2
           {{3, 4}, 24.48}, // GPU proc 3 -> GPU proc 4
           {{4, 3}, 24.58}}}}, // GPU proc 4 -> GPU proc 3
  };
  return profiles;
}

static const std::map<std::pair<std::string, std::string>, double> &
crossNodeCommProfiles() {
  static const std::map<std::pair<std::string, std::string>, double> profiles = {
      {{"4090-01", "a6000-01"}, 0.106}, // hostfile cross-node test_mpi_cuda_staged_bandwidth
      {{"a6000-01", "4090-01"}, 0.106}, // hostfile cross-node test_mpi_cuda_staged_bandwidth
  };
  return profiles;
}

struct TaskCandidate {
  struct DeviceReservation {
    int rank = -1;
    int proc = -1;
    double ready_time = 0.0;
  };

  int rank = -1;
  int proc = -1;
  int num_parts = 1;
  bool persistent_split = false;
  double start_time = 0.0;
  double finish_time = std::numeric_limits<double>::infinity();
  CostEstimate exec_estimate;
  double transfer_uncertainty = 0.0;
  double movement_bytes = 0.0;
  std::vector<int> occupied_procs;
  std::vector<DeviceReservation> transfer_reservations;
};

struct DependencyTransferPlan {
  double ready_time = 0.0;
  double estimated_cost = 0.0;
  double uncertainty = 0.0;
  double movement_bytes = 0.0;
  std::vector<TaskCandidate::DeviceReservation> reservations;
};

static double riskConfidenceMultiplier() {
  return static_cast<double>(SNMD_OFFLINE_RISK_CONFIDENCE_PERCENT) / 100.0;
}

static double riskAdjustedCost(const CostEstimate &estimate) {
  return estimate.mean +
         riskConfidenceMultiplier() * estimate.uncertainty;
}

static double candidateTotalUncertainty(const TaskCandidate &candidate) {
  return std::hypot(candidate.exec_estimate.uncertainty,
                    candidate.transfer_uncertainty);
}

static double candidateRiskScore(const TaskCandidate &candidate) {
  return candidate.finish_time +
         riskConfidenceMultiplier() * candidateTotalUncertainty(candidate);
}

static bool preferTaskCandidate(const TaskCandidate &candidate,
                                const TaskCandidate &incumbent) {
  const double candidate_score = candidateRiskScore(candidate);
  const double incumbent_score = candidateRiskScore(incumbent);
  if (!std::isfinite(candidate_score)) {
    return false;
  }
  if (!std::isfinite(incumbent_score)) {
    return true;
  }

  const double scale = std::max({1.0, std::abs(candidate_score),
                                 std::abs(incumbent_score)});
  const double epsilon = scale * 1.0e-9;
  if (candidate_score + epsilon < incumbent_score) {
    return true;
  }
  if (incumbent_score + epsilon < candidate_score) {
    return false;
  }

  // Deterministic tie-breaking keeps risk-equivalent work data-local and
  // avoids consuming a gang when one device is sufficient.
  if (candidate.movement_bytes != incumbent.movement_bytes) {
    return candidate.movement_bytes < incumbent.movement_bytes;
  }
  if (candidate.num_parts != incumbent.num_parts) {
    return candidate.num_parts < incumbent.num_parts;
  }
  if (candidate.finish_time != incumbent.finish_time) {
    return candidate.finish_time < incumbent.finish_time;
  }
  if (candidate.rank != incumbent.rank) {
    return candidate.rank < incumbent.rank;
  }
  return candidate.proc < incumbent.proc;
}

struct NodePlacementState {
  int exec_rank = -1;
  int exec_proc = -1;
  int num_parts = 1;
  bool persistent_split = false;
  double finish_time = 0.0;
  std::vector<int> split_devices;
};

static std::string profileKeyForNode(const DAGNode *node) {
  return buildKernelProfileKey(node->kernel_identity, node->req_data,
                               node->work_dim,
                               node->global_size0, node->global_size1,
                               node->global_size2);
}

static double reqBytes(const SyclReqData &req) {
  return static_cast<double>(req.elem_size) * static_cast<double>(req.buff_size);
}

static double reqAccessElems(const SyclReqData &req) {
  return static_cast<double>(req.access_range0) *
         static_cast<double>(req.access_range1) *
         static_cast<double>(req.access_range2);
}

static double reqAccessBytes(const SyclReqData &req) {
  return static_cast<double>(req.elem_size) * reqAccessElems(req);
}

static double totalReqElems(const DAGNode *node) {
  double elems = 0.0;
  for (const SyclReqData &req : node->req_data) {
    elems += reqAccessElems(req);
  }
  return elems;
}

static double totalReqBytes(const DAGNode *node) {
  double bytes = 0.0;
  for (const SyclReqData &req : node->req_data) {
    bytes += reqBytes(req);
  }
  return bytes;
}

static double totalReadElems(const DAGNode *node) {
  double elems = 0.0;
  for (const SyclReqData &req : node->req_data) {
    if (isReadAccess(req.req_accmode)) {
      elems += reqAccessElems(req);
    }
  }
  return elems;
}

static double totalReadBytesForPartitionMode(const DAGNode *node,
                                             bool partition_local) {
  double bytes = 0.0;
  for (const SyclReqData &req : node->req_data) {
    if (isReadAccess(req.req_accmode) &&
        req.partition_local == partition_local) {
      bytes += reqAccessBytes(req);
    }
  }
  return bytes;
}

static double totalWriteElems(const DAGNode *node) {
  double elems = 0.0;
  for (const SyclReqData &req : node->req_data) {
    if (isWriteAccess(req.req_accmode)) {
      elems += reqAccessElems(req);
    }
  }
  return elems;
}

static double totalWriteBytes(const DAGNode *node) {
  double bytes = 0.0;
  for (const SyclReqData &req : node->req_data) {
    if (isWriteAccess(req.req_accmode)) {
      bytes += reqAccessBytes(req);
    }
  }
  return bytes;
}

static double dependentReadBytesForPartitionMode(const DAGNode *node,
                                                  bool partition_local) {
  double bytes = 0.0;
  std::unordered_set<void *> seen_mem;
  for (const auto &dep_pair : node->depend_on_mem) {
    for (const SyclReqData &req : dep_pair.second) {
      if (!isReadAccess(req.req_accmode) ||
          req.partition_local != partition_local) {
        continue;
      }
      if (seen_mem.insert(req.mem_pointer).second) {
        bytes += reqAccessBytes(req);
      }
    }
  }
  return bytes;
}

static double coldArithmeticIntensityFactor(const DAGNode *node) {
  if (node->depend_on.empty() && node->batch_root_count > 1) {
    return 1.0;
  }

  int read_only_reqs = 0;
  int write_reqs = 0;
  double min_req_elems = std::numeric_limits<double>::infinity();
  double max_req_elems = 0.0;
  double max_write_elems = 0.0;

  for (const SyclReqData &req : node->req_data) {
    const bool reads = isReadAccess(req.req_accmode);
    const bool writes = isWriteAccess(req.req_accmode);
    if (reads && !writes) {
      ++read_only_reqs;
    }
    if (writes) {
      ++write_reqs;
      max_write_elems =
          std::max(max_write_elems, reqAccessElems(req));
    }
    if (reads || writes) {
      const double elems = reqAccessElems(req);
      min_req_elems = std::min(min_req_elems, elems);
      max_req_elems = std::max(max_req_elems, elems);
    }
  }

  if (read_only_reqs < 2 || write_reqs == 0 ||
      !std::isfinite(min_req_elems) || min_req_elems <= 0.0 ||
      max_req_elems / min_req_elems > 1.05 ||
      max_write_elems < 1024.0 * 1024.0) {
    return 1.0;
  }

  const double linear_extent = std::sqrt(max_write_elems);
  return std::max(1.0, std::min(64.0, linear_extent / 256.0));
}

static double coldWorkElems(const DAGNode *node) {
  return std::max(1.0, node->total_elem) *
         coldArithmeticIntensityFactor(node);
}

static uint64_t daemonSteadyNowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
      .count());
}

static uint64_t daemonUnixNowSec() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

static ComputeCapability fallbackCapabilityForProc(int proc);

static KernelFeatureSignature buildKernelFeature(const DAGNode *node) {
  KernelFeatureSignature feature;
  if (node == nullptr) {
    return feature;
  }

  feature.kernel_identity = node->kernel_identity;
  feature.work_dim = node->work_dim;
  feature.global_items =
      static_cast<double>(node->global_size0) *
      static_cast<double>(node->global_size1) *
      static_cast<double>(node->global_size2);
  feature.total_access_elems = 0.0;
  feature.req_count = static_cast<int>(node->req_data.size());

  std::map<int, double> elems_by_size;
  for (const SyclReqData &req : node->req_data) {
    const bool reads = isReadAccess(req.req_accmode);
    const bool writes = isWriteAccess(req.req_accmode);
    const double access_elems = reqAccessElems(req);
    const double access_bytes = reqAccessBytes(req);
    feature.total_access_elems += access_elems;
    if (reads) {
      feature.read_req_count++;
      feature.read_bytes += access_bytes;
      feature.partition_local_read =
          feature.partition_local_read || req.partition_local;
    }
    if (writes) {
      feature.write_req_count++;
      feature.write_bytes += access_bytes;
      feature.partition_local_write =
          feature.partition_local_write || req.partition_local;
    }
    const unsigned mode = static_cast<unsigned>(req.req_accmode);
    if (mode < 32) {
      feature.access_mode_mask |= (1U << mode);
    }
    if (req.elem_size > 0) {
      elems_by_size[req.elem_size] += access_elems;
    }
  }

  double dominant_elems = -1.0;
  for (const auto &entry : elems_by_size) {
    if (entry.second > dominant_elems) {
      dominant_elems = entry.second;
      feature.dominant_elem_size = entry.first;
    }
  }

  // Keep the transfer feature independent of the current DAG. The cold model
  // deliberately changes its arithmetic-intensity guard for a wide root set,
  // but a persisted sample must retain the same feature when the same kernel
  // appears in a narrow window. Identity transfer learns the kernel-specific
  // intensity from the measured service time; this proxy only scales shape.
  feature.analytical_work =
      std::max(feature.global_items, feature.total_access_elems);
  feature.global_items = std::max(1.0, feature.global_items);
  feature.total_access_elems = std::max(1.0, feature.total_access_elems);
  feature.analytical_work = std::max(1.0, feature.analytical_work);
  return feature;
}

static void registerKernelFeature(const DAGNode *node) {
  if (node == nullptr) {
    return;
  }
  const std::string key = profileKeyForNode(node);
  const KernelFeatureSignature feature = buildKernelFeature(node);
  std::lock_guard<std::mutex> lock(profile_cost_mutex);
  kernel_feature_table[key] = feature;
}

static std::pair<double, double> profileSourceCapabilities(int rank,
                                                           int device) {
  if (rank >= 0 && rank < static_cast<int>(gpu_capability.size()) &&
      device >= 0 &&
      device < static_cast<int>(gpu_capability[rank].size())) {
    return {gpu_capability[rank][device].fp32,
            gpu_capability[rank][device].fp64};
  }
  const ComputeCapability fallback = fallbackCapabilityForProc(device);
  return {fallback.fp32, fallback.fp64};
}

static bool profileLegacyOrdinalStoreEnabled() {
  static const bool enabled = [] {
    const char *env =
        std::getenv("SYCL_SNMD_PROFILE_ALLOW_LEGACY_ORDINALS");
    return env != nullptr && std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "FALSE") != 0;
  }();
  return enabled;
}

static std::string ordinalProfileDeviceIdentity(int rank, int device) {
  return "rank-" + std::to_string(rank) + "-device-" +
         std::to_string(device);
}

static std::string profileDeviceIdentity(int rank, int device) {
  if (profileLegacyOrdinalStoreEnabled() || rank != 0 || device <= 0) {
    return ordinalProfileDeviceIdentity(rank, device);
  }

  std::lock_guard<std::mutex> lock(monitor_state_mutex);
  auto identity_it = index_sycl_device_identity.find(device);
  if (identity_it != index_sycl_device_identity.end() &&
      !identity_it->second.empty()) {
    return identity_it->second;
  }
  // Monitor initialization is synchronous before profile loading and program
  // submission. Keep a deterministic fallback for non-CUDA devices.
  return ordinalProfileDeviceIdentity(rank, device);
}

static void applyProfileObservationLocked(
    const PersistedProfileObservation &observation, bool live_sample) {
  ProfileCostEntry &entry = profile_cost_table[observation.key];
  if (entry.samples == 0) {
    entry.ewma_cost = observation.sample_cost;
  } else {
    entry.ewma_cost = entry.ewma_cost * 0.7 + observation.sample_cost * 0.3;
  }
  entry.samples++;
  entry.live_samples += live_sample ? 1 : 0;
  const double delta = observation.sample_cost - entry.mean_cost;
  entry.mean_cost += delta / static_cast<double>(entry.samples);
  const double delta_after_mean = observation.sample_cost - entry.mean_cost;
  entry.m2_cost += delta * delta_after_mean;
  entry.min_cost = std::min(entry.min_cost, observation.sample_cost);
  if (observation.source_fp32_capability > 0.0) {
    entry.source_fp32_capability = observation.source_fp32_capability;
  }
  if (observation.source_fp64_capability > 0.0) {
    entry.source_fp64_capability = observation.source_fp64_capability;
  }
  entry.last_observed_unix_sec =
      std::max(entry.last_observed_unix_sec, observation.observed_unix_sec);
  if (observation.feature_valid) {
    kernel_feature_table[observation.key.kernel_key] = observation.feature;
  }
}

static std::string serializePersistedProfileObservation(
    const PersistedProfileObservation &observation) {
  const KernelFeatureSignature &feature = observation.feature;
  std::ostringstream oss;
  oss << std::setprecision(17) << "SNMD_PROFILE_OBS_V3\t"
      << observation.profile_namespace << '\t' << observation.key.rank << '\t'
      << observation.key.device << '\t'
      << observation.key.device_identity << '\t'
      << observation.key.num_parts << '\t'
      << (observation.key.persistent_split ? 1 : 0) << '\t'
      << observation.sample_cost << '\t'
      << observation.source_fp32_capability << '\t'
      << observation.source_fp64_capability << '\t'
      << observation.observed_unix_sec << '\t'
      << (observation.feature_valid ? 1 : 0) << '\t'
      << feature.kernel_identity << '\t' << feature.work_dim << '\t'
      << feature.global_items << '\t' << feature.total_access_elems << '\t'
      << feature.read_bytes << '\t' << feature.write_bytes << '\t'
      << feature.analytical_work << '\t' << feature.req_count << '\t'
      << feature.read_req_count << '\t' << feature.write_req_count << '\t'
      << feature.dominant_elem_size << '\t' << feature.access_mode_mask << '\t'
      << (feature.partition_local_read ? 1 : 0) << '\t'
      << (feature.partition_local_write ? 1 : 0) << '\t'
      << observation.key.kernel_key << '\n';
  return oss.str();
}

static bool deserializePersistedProfileObservation(
    const std::string &line, PersistedProfileObservation &observation) {
  std::istringstream iss(line);
  std::string tag;
  int persistent_split = 0;
  int feature_valid = 0;
  int partition_local_read = 0;
  int partition_local_write = 0;
  if (!(iss >> tag)) {
    return false;
  }
  const bool has_stable_device_identity = tag == "SNMD_PROFILE_OBS_V3";
  if (has_stable_device_identity) {
    if (!(iss >> observation.profile_namespace)) {
      return false;
    }
  } else if (tag == "SNMD_PROFILE_OBS_V2") {
    if (!profileLegacyOrdinalStoreEnabled() ||
        !(iss >> observation.profile_namespace)) {
      return false;
    }
  } else if (tag == "SNMD_PROFILE_OBS_V1") {
    if (!profileLegacyOrdinalStoreEnabled()) {
      return false;
    }
    // V1 predates build/application isolation. Only the default namespace
    // accepts it; a caller selecting a namespace gets strict isolation.
    observation.profile_namespace = "default";
  } else {
    return false;
  }
  if (!(iss >> observation.key.rank >> observation.key.device)) {
    return false;
  }
  if (has_stable_device_identity) {
    if (!(iss >> observation.key.device_identity)) {
      return false;
    }
  } else {
    observation.key.device_identity = ordinalProfileDeviceIdentity(
        observation.key.rank, observation.key.device);
  }
  if (!(iss >> observation.key.num_parts >> persistent_split >>
        observation.sample_cost >> observation.source_fp32_capability >>
        observation.source_fp64_capability >> observation.observed_unix_sec >>
        feature_valid >> observation.feature.kernel_identity >>
        observation.feature.work_dim >> observation.feature.global_items >>
        observation.feature.total_access_elems >>
        observation.feature.read_bytes >> observation.feature.write_bytes >>
        observation.feature.analytical_work >> observation.feature.req_count >>
        observation.feature.read_req_count >>
        observation.feature.write_req_count >>
        observation.feature.dominant_elem_size >>
        observation.feature.access_mode_mask >> partition_local_read >>
        partition_local_write >> observation.key.kernel_key)) {
    return false;
  }
  observation.key.num_parts = std::max(1, observation.key.num_parts);
  observation.key.persistent_split = persistent_split != 0;
  observation.feature_valid = feature_valid != 0;
  observation.feature.partition_local_read = partition_local_read != 0;
  observation.feature.partition_local_write = partition_local_write != 0;
  return observation.sample_cost > 0.0 &&
         std::isfinite(observation.sample_cost) &&
         !observation.key.kernel_key.empty() &&
         !observation.key.device_identity.empty();
}

static void persistentProfileStoreWriter() {
  std::ofstream store(profile_store_path, std::ios::out | std::ios::app);
  if (!store.is_open()) {
    std::lock_guard<std::mutex> lock(profile_store_mutex);
    profile_store_enabled = false;
    profile_store_queue.clear();
    DAEMON_TRACE_STREAM << "ProfileStore: cannot open " << profile_store_path
                        << std::endl;
    return;
  }

  while (true) {
    PersistedProfileObservation observation;
    {
      std::unique_lock<std::mutex> lock(profile_store_mutex);
      profile_store_cv.wait(lock, [] {
        return profile_store_stop || !profile_store_queue.empty();
      });
      if (profile_store_stop && profile_store_queue.empty()) {
        break;
      }
      observation = std::move(profile_store_queue.front());
      profile_store_queue.pop_front();
    }
    store << serializePersistedProfileObservation(observation);
    store.flush();
  }
}

static void enqueuePersistentProfileObservation(
    PersistedProfileObservation observation) {
  std::lock_guard<std::mutex> lock(profile_store_mutex);
  if (!profile_store_enabled || profile_store_stop) {
    return;
  }
  if (profile_store_queue.size() >=
      static_cast<size_t>(SNMD_OFFLINE_PROFILE_STORE_QUEUE_LIMIT)) {
    profile_store_queue.pop_front();
    ++profile_store_dropped_records;
  }
  profile_store_queue.push_back(std::move(observation));
  profile_store_cv.notify_one();
}

static void initializePersistentProfileStore() {
  if (mpi_rank != 0) {
    return;
  }
  const char *enabled_env = std::getenv("SYCL_SNMD_PROFILE_PERSIST");
  bool enabled = SNMD_OFFLINE_PROFILE_STORE_DEFAULT_ENABLED != 0;
  if (enabled_env != nullptr) {
    enabled = std::strcmp(enabled_env, "0") != 0 &&
              std::strcmp(enabled_env, "false") != 0 &&
              std::strcmp(enabled_env, "FALSE") != 0;
  }
  if (!enabled) {
    return;
  }

  const char *namespace_env = std::getenv("SYCL_SNMD_PROFILE_NAMESPACE");
  if (namespace_env != nullptr && namespace_env[0] != '\0') {
    profile_store_namespace = namespace_env;
    for (char &ch : profile_store_namespace) {
      if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' &&
          ch != '_' && ch != '.') {
        ch = '_';
      }
    }
  }

  const char *path_env = std::getenv("SYCL_SNMD_PROFILE_STORE");
  profile_store_path =
      path_env != nullptr && path_env[0] != '\0'
          ? path_env
          : "/tmp/sycl-snmd-profile-observations-v3-rank0.tsv";

  size_t loaded = 0;
  std::ifstream store(profile_store_path);
  std::string line;
  const uint64_t now_sec = daemonUnixNowSec();
  const uint64_t max_age_sec =
      static_cast<uint64_t>(SNMD_OFFLINE_PROFILE_STORE_MAX_AGE_DAYS) * 86400ULL;
  while (std::getline(store, line)) {
    PersistedProfileObservation observation;
    if (!deserializePersistedProfileObservation(line, observation)) {
      continue;
    }
    if (observation.profile_namespace != profile_store_namespace) {
      continue;
    }
    if (observation.observed_unix_sec != 0 &&
        now_sec > observation.observed_unix_sec &&
        now_sec - observation.observed_unix_sec > max_age_sec) {
      continue;
    }
    std::lock_guard<std::mutex> lock(profile_cost_mutex);
    applyProfileObservationLocked(observation, /*live_sample=*/false);
    ++loaded;
  }

  {
    std::lock_guard<std::mutex> lock(profile_store_mutex);
    profile_store_stop = false;
    profile_store_enabled = true;
  }
  profile_store_thread = std::thread(persistentProfileStoreWriter);
  DAEMON_TRACE_STREAM << "ProfileStore: loaded " << loaded << " records from "
                      << profile_store_path << " namespace "
                      << profile_store_namespace << std::endl;
}

static void shutdownPersistentProfileStore() {
  {
    std::lock_guard<std::mutex> lock(profile_store_mutex);
    if (!profile_store_thread.joinable()) {
      return;
    }
    profile_store_stop = true;
    profile_store_cv.notify_all();
  }
  profile_store_thread.join();
  DAEMON_TRACE_STREAM << "ProfileStore: stopped dropped_records "
                      << profile_store_dropped_records << std::endl;
}

static void updateProfileCostTable(const S2DKernelProfileData &profile,
                                   int sample_rank) {
  if (profile.duration_ns == 0 || profile.kernel_key.empty()) {
    return;
  }

  ProfileCostKey key{profile.kernel_key, sample_rank, profile.device_index,
                     profileDeviceIdentity(sample_rank, profile.device_index),
                     std::max(1, profile.num_parts),
                     profile.persistent_split};
  const double sample_cost =
      static_cast<double>(profile.duration_ns) / PROFILE_NS_TO_COST;

  PersistedProfileObservation observation;
  observation.profile_namespace = profile_store_namespace;
  observation.key = key;
  observation.sample_cost = sample_cost;
  const auto source_capability =
      profileSourceCapabilities(sample_rank, profile.device_index);
  observation.source_fp32_capability = source_capability.first;
  observation.source_fp64_capability = source_capability.second;
  observation.observed_unix_sec = daemonUnixNowSec();

  ProfileCostEntry entry_snapshot;
  {
    std::lock_guard<std::mutex> lock(profile_cost_mutex);
    auto feature_it = kernel_feature_table.find(profile.kernel_key);
    if (feature_it != kernel_feature_table.end()) {
      observation.feature_valid = true;
      observation.feature = feature_it->second;
    }
    applyProfileObservationLocked(observation, /*live_sample=*/true);
    profile_device_update_ns[
        {profile.pid, sample_rank, profile.device_index}] =
        daemonSteadyNowNs();
    entry_snapshot = profile_cost_table[key];
  }
  // Persistence is deliberately outside profile_cost_mutex and only enqueues
  // a bounded record. The writer thread may block on storage without delaying
  // completion acknowledgement or the next ready-queue admission.
  enqueuePersistentProfileObservation(std::move(observation));

  const double stddev =
      entry_snapshot.samples > 1
          ? std::sqrt(entry_snapshot.m2_cost /
                      static_cast<double>(entry_snapshot.samples - 1))
          : 0.0;

  DAEMON_TRACE_STREAM << "ProfileCostTable: key " << profile.kernel_key
            << " rank " << sample_rank << " device " << profile.device_index
            << " identity " << key.device_identity
            << " parts " << std::max(1, profile.num_parts)
            << " persistent " << profile.persistent_split
            << " sample_cost " << sample_cost
            << " ewma_cost " << entry_snapshot.ewma_cost
            << " mean_cost " << entry_snapshot.mean_cost
            << " stddev_cost " << stddev
            << " min_cost " << entry_snapshot.min_cost
            << " samples " << entry_snapshot.samples
            << " live_samples " << entry_snapshot.live_samples << std::endl;
}

static double profileEntryUncertainty(const ProfileCostEntry &entry) {
  if (entry.samples <= 0) {
    return 0.0;
  }

  // Predict the next service time, not only the sample mean. Runtime jitter is
  // aleatoric uncertainty and remains after repeated observations; the
  // capability prior is epistemic and decays as exact samples arrive.
  const double observed_prediction_uncertainty =
      entry.samples > 1
          ? std::sqrt(entry.m2_cost /
                      static_cast<double>(entry.samples - 1))
          : 0.0;
  const double prior_uncertainty =
      std::max(0.001, entry.ewma_cost) *
      (static_cast<double>(SNMD_OFFLINE_PROFILE_PRIOR_ERROR_PERCENT) / 100.0) /
      std::sqrt(static_cast<double>(entry.samples));
  double uncertainty =
      std::hypot(observed_prediction_uncertainty, prior_uncertainty);
  if (entry.live_samples == 0) {
    const uint64_t now_sec = daemonUnixNowSec();
    const double age_days =
        entry.last_observed_unix_sec != 0 && now_sec > entry.last_observed_unix_sec
            ? static_cast<double>(now_sec - entry.last_observed_unix_sec) /
                  86400.0
            : 0.0;
    const double persisted_floor =
        std::max(0.001, entry.ewma_cost) *
        (static_cast<double>(SNMD_OFFLINE_PERSISTED_PROFILE_ERROR_PERCENT) /
             100.0 +
         std::min(0.5, age_days * 0.01));
    uncertainty = std::hypot(uncertainty, persisted_floor);
  }
  return uncertainty;
}

static bool lookupExactProfileEstimate(const std::string &kernel_key,
                                       int rank, int device, int num_parts,
                                       bool persistent_split,
                                       CostEstimate &estimate) {
  ProfileCostKey exact{kernel_key, rank, device,
                       profileDeviceIdentity(rank, device),
                       std::max(1, num_parts), persistent_split};
  std::lock_guard<std::mutex> lock(profile_cost_mutex);
  auto exact_it = profile_cost_table.find(exact);
  if (exact_it == profile_cost_table.end() || exact_it->second.samples <= 0) {
    return false;
  }

  estimate.mean = exact_it->second.ewma_cost;
  estimate.uncertainty = profileEntryUncertainty(exact_it->second);
  estimate.samples = exact_it->second.samples;
  estimate.source = exact_it->second.live_samples > 0
                        ? CostEstimateSource::exact_profile
                        : CostEstimateSource::persisted_profile;
  return true;
}

static bool hasLiveExactProfileCost(const std::string &kernel_key, int rank,
                                    int device, int num_parts,
                                    bool persistent_split = false) {
  ProfileCostKey exact{kernel_key, rank, device,
                       profileDeviceIdentity(rank, device),
                       std::max(1, num_parts), persistent_split};
  std::lock_guard<std::mutex> lock(profile_cost_mutex);
  auto it = profile_cost_table.find(exact);
  return it != profile_cost_table.end() && it->second.live_samples > 0;
}

#if defined(SNMD_OFFLINE_COLD_SPLIT_PROBE) ||                              \
    defined(SNMD_OFFLINE_WIDE_DAG_GUARD) ||                                \
    defined(SNMD_OFFLINE_SPLIT_STATS)
static bool hasProfileCostForParts(const std::string &kernel_key,
                                   int num_parts,
                                   bool persistent_split = false) {
  const int parts = std::max(1, num_parts);
  std::lock_guard<std::mutex> lock(profile_cost_mutex);
  for (const auto &entry : profile_cost_table) {
    if (entry.first.kernel_key == kernel_key &&
        entry.first.num_parts == parts &&
        entry.first.persistent_split == persistent_split &&
        entry.second.samples > 0) {
      return true;
    }
  }
  return false;
}
#endif

static std::string toLowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

static bool containsIgnoreCase(const std::string &value,
                               const std::string &pattern) {
  return toLowerAscii(value).find(toLowerAscii(pattern)) != std::string::npos;
}

static std::string readCpuModelName() {
  std::ifstream file("/proc/cpuinfo");
  std::string line;
  while (std::getline(file, line)) {
    const std::string key = "model name";
    if (line.rfind(key, 0) != 0) {
      continue;
    }

    size_t colon = line.find(':');
    if (colon == std::string::npos || colon + 1 >= line.size()) {
      return line;
    }
    size_t start = colon + 1;
    while (start < line.size() &&
           std::isspace(static_cast<unsigned char>(line[start]))) {
      ++start;
    }
    return line.substr(start);
  }
  return "CPU";
}

static ComputeCapability inferCpuCapabilityFromName(const std::string &name) {
  if (containsIgnoreCase(name, "epyc 7763")) {
    return ComputeCapability{4.4, 2.2};
  }
  if (containsIgnoreCase(name, "xeon") &&
      containsIgnoreCase(name, "gold 6530")) {
    // Calibrated from fdtd2d_timer_32768 on a6000-docker.
    // Baseline is Xeon Gold 6530 FP32 kernel-run time: fp32=1.0,
    // fp64=319.533s/595.512s.
    return ComputeCapability{1.0, 0.537};
  }
  if (containsIgnoreCase(name, "epyc")) {
    return ComputeCapability{3.0, 1.5};
  }
  if (containsIgnoreCase(name, "xeon")) {
    return ComputeCapability{2.0, 1.0};
  }
  return ComputeCapability{1.0, 0.5};
}

static ComputeCapability inferGpuCapabilityFromName(const std::string &name) {
  if (containsIgnoreCase(name, "h100")) {
    return ComputeCapability{67.0, 33.5};
  }
  if (containsIgnoreCase(name, "rtx 6000") &&
      containsIgnoreCase(name, "ada")) {
    // Calibrated from fdtd2d_timer_32768 on a6000-docker.
    // fp32=319.533s/28.0602s, fp64=319.533s/57.4197s.
    return ComputeCapability{11.39, 5.56};
  }
  if (containsIgnoreCase(name, "rtx 4090")) {
    return ComputeCapability{82.6, 1.29};
  }
  if (containsIgnoreCase(name, "rtx 3090")) {
    return ComputeCapability{35.6, 0.56};
  }
  if (containsIgnoreCase(name, "a100")) {
    return ComputeCapability{19.5, 9.7};
  }
  if (containsIgnoreCase(name, "a40")) {
    return ComputeCapability{37.4, 0.58};
  }
  if (containsIgnoreCase(name, "rtx a6000") ||
      containsIgnoreCase(name, "a6000")) {
    return ComputeCapability{38.7, 0.60};
  }
  return ComputeCapability{10.0, 1.0};
}

static const NodeCommProfile *findNodeCommProfileByKey(
    const std::string &key) {
  const auto it = nodeCommProfiles().find(key);
  if (it != nodeCommProfiles().end()) {
    return &it->second;
  }
  return nullptr;
}

static const NodeCommProfile *findNodeCommProfileById(int id) {
  for (const auto &entry : nodeCommProfiles()) {
    const NodeCommProfile &profile = entry.second;
    if (profile.id == id) {
      return &profile;
    }
  }
  return nullptr;
}

static std::string localHostName() {
  char hostname[256] = {};
  if (gethostname(hostname, sizeof(hostname) - 1) == 0) {
    return hostname;
  }
  return "";
}

static std::string inferCommProfileKeyFromHostName(
    const std::string &hostname) {
  if (containsIgnoreCase(hostname, "4090")) {
    return "4090-01";
  }
  if (containsIgnoreCase(hostname, "a6000")) {
    return "a6000-01";
  }
  return "";
}

static std::string inferCommProfileKeyFromLocalDevices() {
  std::vector<MonitorInfo> monitor_snapshot;
  {
    std::lock_guard<std::mutex> lock(monitor_state_mutex);
    monitor_snapshot = device_monitor_info;
  }
  for (const MonitorInfo &info : monitor_snapshot) {
    if (containsIgnoreCase(info.name, "4090")) {
      return "4090-01";
    }
    if (containsIgnoreCase(info.name, "rtx 6000") &&
        containsIgnoreCase(info.name, "ada")) {
      return "a6000-01";
    }
  }
  return "";
}

static int detectLocalCommProfileId() {
  const char *override_key = std::getenv("SYCL_DAEMON_NODE_KEY");
  if (override_key != nullptr && override_key[0] != '\0') {
    const NodeCommProfile *profile = findNodeCommProfileByKey(override_key);
    if (profile != nullptr) {
      return profile->id;
    }
    DAEMON_TRACE_STREAM << "CommProfile: unknown SYCL_DAEMON_NODE_KEY=" << override_key
              << ", fallback to auto detection" << std::endl;
  }

  std::string key = inferCommProfileKeyFromHostName(localHostName());
  if (key.empty()) {
    key = inferCommProfileKeyFromLocalDevices();
  }

  const NodeCommProfile *profile = findNodeCommProfileByKey(key);
  return profile == nullptr ? -1 : profile->id;
}

static const NodeCommProfile *commProfileForRank(int rank) {
  if (rank >= 0 && rank < static_cast<int>(gpu_comm_profile_ids.size())) {
    const NodeCommProfile *profile =
        findNodeCommProfileById(gpu_comm_profile_ids[rank]);
    if (profile != nullptr) {
      return profile;
    }
  }
  return nullptr;
}

static void ensureLocalCommProfileVisible(size_t rank_count) {
  const int detected_profile_id = detectLocalCommProfileId();
  std::lock_guard<std::mutex> lock(monitor_state_mutex);
  local_comm_profile_id = detected_profile_id;
  if (cluster_comm_profile_ids.size() < rank_count) {
    cluster_comm_profile_ids.resize(rank_count, -1);
  }
  if (local_comm_profile_id < 0) {
    return;
  }

  if (mpi_rank >= 0 &&
      mpi_rank < static_cast<int>(cluster_comm_profile_ids.size())) {
    cluster_comm_profile_ids[mpi_rank] = local_comm_profile_id;
  }
  if (monitor_rank >= 0 &&
      monitor_rank < static_cast<int>(cluster_comm_profile_ids.size())) {
    cluster_comm_profile_ids[monitor_rank] = local_comm_profile_id;
  }
}

static double bandwidthAtDeviceIndex(const std::vector<double> &bandwidths,
                                     int device, double fallback) {
  if (device >= 0 && device < static_cast<int>(bandwidths.size()) &&
      bandwidths[device] > 0.0) {
    return bandwidths[device];
  }
  return fallback;
}

static double secondsForBytesAtBandwidth(double bytes, double bw_gib) {
  if (bytes <= 0.0) {
    return 0.0;
  }
  return bytes / (std::max(0.001, bw_gib) * GIB_BYTES);
}

static double heftCostFromSeconds(double seconds) {
  return seconds * HEFT_COMM_COST_PER_SECOND;
}

static double sameRankCopySeconds(int rank, int src_proc, int dst_proc,
                                  double bytes) {
  if (bytes <= 0.0 || src_proc == dst_proc) {
    return 0.0;
  }

  const NodeCommProfile *profile = commProfileForRank(rank);
  const double shm_bw = profile == nullptr ? FALLBACK_HOST_BW_GIB
                                           : profile->shm_bw_gib;

  if (src_proc == 0 && dst_proc > 0) {
    const double h2d_bw =
        profile == nullptr ? FALLBACK_H2D_BW_GIB
                           : bandwidthAtDeviceIndex(profile->h2d_bw_gib,
                                                    dst_proc,
                                                    FALLBACK_H2D_BW_GIB);
    return secondsForBytesAtBandwidth(bytes, h2d_bw);
  }
  if (src_proc > 0 && dst_proc == 0) {
    const double d2h_bw =
        profile == nullptr ? FALLBACK_D2H_BW_GIB
                           : bandwidthAtDeviceIndex(profile->d2h_bw_gib,
                                                    src_proc,
                                                    FALLBACK_D2H_BW_GIB);
    return secondsForBytesAtBandwidth(bytes, d2h_bw);
  }
  if (src_proc == 0 && dst_proc == 0) {
    return secondsForBytesAtBandwidth(bytes, shm_bw);
  }

  if (profile != nullptr) {
    auto it = profile->d2d_bw_gib.find({src_proc, dst_proc});
    if (it != profile->d2d_bw_gib.end() && it->second > 0.0) {
      return secondsForBytesAtBandwidth(bytes, it->second);
    }

    const double d2h_bw = bandwidthAtDeviceIndex(profile->d2h_bw_gib,
                                                 src_proc,
                                                 FALLBACK_D2H_BW_GIB);
    const double h2d_bw = bandwidthAtDeviceIndex(profile->h2d_bw_gib,
                                                 dst_proc,
                                                 FALLBACK_H2D_BW_GIB);
    return secondsForBytesAtBandwidth(bytes, d2h_bw) +
           secondsForBytesAtBandwidth(bytes, h2d_bw);
  }

  return secondsForBytesAtBandwidth(bytes, FALLBACK_D2D_BW_GIB);
}

static double crossRankBandwidthGiB(int src_rank, int dst_rank) {
  const NodeCommProfile *src_profile = commProfileForRank(src_rank);
  const NodeCommProfile *dst_profile = commProfileForRank(dst_rank);
  if (src_profile != nullptr && dst_profile != nullptr) {
    if (src_profile->key == dst_profile->key) {
      return std::min(src_profile->same_node_staged_bw_gib,
                      dst_profile->same_node_staged_bw_gib);
    }
    const auto it =
        crossNodeCommProfiles().find({src_profile->key, dst_profile->key});
    if (it != crossNodeCommProfiles().end()) {
      return it->second;
    }
  }
  return FALLBACK_CROSS_RANK_BW_GIB;
}

static ComputeCapability inferDeviceCapabilityFromName(const std::string &name,
                                                       bool is_cpu) {
  return is_cpu ? inferCpuCapabilityFromName(name)
                : inferGpuCapabilityFromName(name);
}

static ComputeCapability fallbackCapabilityForProc(int proc) {
  return proc == 0 ? ComputeCapability{1.0, 0.5}
                   : ComputeCapability{10.0, 1.0};
}

static ComputeCapability maxCapability(const ComputeCapability &lhs,
                                       const ComputeCapability &rhs) {
  return ComputeCapability{std::max(lhs.fp32, rhs.fp32),
                           std::max(lhs.fp64, rhs.fp64)};
}

static double minPositiveCapability(
    const std::vector<std::vector<ComputeCapability>> &capabilities) {
  double min_capability = std::numeric_limits<double>::infinity();
  for (const std::vector<ComputeCapability> &rank_capability : capabilities) {
    for (const ComputeCapability &capability : rank_capability) {
      if (capability.fp32 > 0.0) {
        min_capability = std::min(min_capability, capability.fp32);
      }
      if (capability.fp64 > 0.0) {
        min_capability = std::min(min_capability, capability.fp64);
      }
    }
  }
  return std::isfinite(min_capability) ? min_capability : 1.0;
}

static bool hasFreshProfileForDevice(pid_t program_pid, int rank, int proc) {
  // A profile is sent immediately after the user's wait fence completed.  The
  // monitor sample gathered in the same short interval still mostly reflects
  // work that this runtime already knows has finished.  Treating it as new
  // external load would count the same work twice.
  static constexpr uint64_t FRESH_PROFILE_NS =
      static_cast<uint64_t>(MONITOR_GATHER_INTERVAL) * 2ULL * 1000ULL;
  const uint64_t now_ns = daemonSteadyNowNs();
  std::lock_guard<std::mutex> lock(profile_cost_mutex);
  auto it = profile_device_update_ns.find({program_pid, rank, proc});
  return it != profile_device_update_ns.end() && now_ns >= it->second &&
         now_ns - it->second <= FRESH_PROFILE_NS;
}

static double monitorServiceTimeScale(double util, bool fresh_profile) {
  util = std::max(0.0, std::min(100.0, util));
  if (fresh_profile || util <= MONITOR_THRESHOLD) {
    return 1.0;
  }

  const double external_busy_ratio =
      (util - MONITOR_THRESHOLD) / (100.0 - MONITOR_THRESHOLD);
  return 1.0 + external_busy_ratio;
}

static size_t offlineRankCount() {
  size_t rank_count = std::max(gpu_capability.size(), cluster_monitor_info.size());
  rank_count = std::max(rank_count, cluster_device_capability.size());
  if (monitor_size > 0) {
    rank_count = std::max(rank_count, static_cast<size_t>(monitor_size));
  }
  if (mpi_size > 0) {
    rank_count = std::max(rank_count, static_cast<size_t>(mpi_size));
  }
  return std::max<size_t>(rank_count, 1);
}

static size_t offlineDeviceCountForRank(size_t rank) {
  if (rank < cluster_monitor_info.size() &&
      !cluster_monitor_info[rank].empty()) {
    return cluster_monitor_info[rank].size();
  }
  if (rank < cluster_device_capability.size() &&
      !cluster_device_capability[rank].empty()) {
    return cluster_device_capability[rank].size();
  }
  if (rank == static_cast<size_t>(monitor_rank) &&
      !device_monitor_info.empty()) {
    return device_monitor_info.size();
  }
  return 1;
}

static ComputeCapability rawCapabilityForDevice(size_t rank, size_t proc) {
  if (rank < cluster_device_capability.size() &&
      proc < cluster_device_capability[rank].size() &&
      (cluster_device_capability[rank][proc].fp32 > 0.0 ||
       cluster_device_capability[rank][proc].fp64 > 0.0)) {
    return cluster_device_capability[rank][proc];
  }
  if (rank == static_cast<size_t>(monitor_rank) &&
      proc < device_capability.size() &&
      (device_capability[proc].fp32 > 0.0 ||
       device_capability[proc].fp64 > 0.0)) {
    return device_capability[proc];
  }
  return fallbackCapabilityForProc(static_cast<int>(proc));
}

static const MonitorInfo *monitorInfoForDevice(int rank, int proc) {
  if (rank >= 0 && rank < static_cast<int>(cluster_monitor_info.size()) &&
      proc >= 0 && proc < static_cast<int>(cluster_monitor_info[rank].size())) {
    return &cluster_monitor_info[rank][proc];
  }
  if (rank == monitor_rank && proc >= 0 &&
      proc < static_cast<int>(device_monitor_info.size())) {
    return &device_monitor_info[proc];
  }
  return nullptr;
}

static void ensureOfflineDeviceModel(pid_t program_pid) {
  size_t rank_count = 0;
  {
    std::lock_guard<std::mutex> monitor_lock(monitor_state_mutex);
    rank_count = offlineRankCount();
  }
  ensureLocalCommProfileVisible(rank_count);

  // Capture one coherent monitor generation for the whole scheduling pass.
  // The algorithm never consults live NVML state again after this function.
  std::lock_guard<std::mutex> monitor_lock(monitor_state_mutex);
  rank_count = std::max(rank_count, offlineRankCount());
  gpu_comm_profile_ids = cluster_comm_profile_ids;
  gpu_comm_profile_ids.resize(rank_count, -1);

  std::vector<std::vector<ComputeCapability>> raw_capability(rank_count);
  for (size_t rank = 0; rank < rank_count; ++rank) {
    const size_t device_count = offlineDeviceCountForRank(rank);
    raw_capability[rank].resize(device_count);
    for (size_t proc = 0; proc < device_count; ++proc) {
      raw_capability[rank][proc] = rawCapabilityForDevice(rank, proc);
    }
  }

  const double min_capability = minPositiveCapability(raw_capability);
  gpu_capability.resize(rank_count);
  gpu_available_time.resize(rank_count);
  gpu_service_time_scale.resize(rank_count);
  gpu_memory_available_kib.resize(rank_count);

  for (size_t rank = 0; rank < rank_count; ++rank) {
    gpu_capability[rank].resize(raw_capability[rank].size());
    gpu_available_time[rank].resize(raw_capability[rank].size());
    gpu_service_time_scale[rank].resize(raw_capability[rank].size());
    gpu_memory_available_kib[rank].resize(raw_capability[rank].size());

    for (size_t proc = 0; proc < raw_capability[rank].size(); ++proc) {
      gpu_capability[rank][proc] = ComputeCapability{
          std::max(0.1, raw_capability[rank][proc].fp32 / min_capability),
          std::max(0.1, raw_capability[rank][proc].fp64 / min_capability)};

      const MonitorInfo *info =
          monitorInfoForDevice(static_cast<int>(rank), static_cast<int>(proc));
      const double util = info == nullptr ? 0.0 : info->util_used;
      const bool fresh_profile = hasFreshProfileForDevice(
          program_pid, static_cast<int>(rank), static_cast<int>(proc));

      // Monitor data is represented once as a contextual service-time scale,
      // rather than also inventing a second availability delay. Cross-window
      // predecessor finish times remain the conservative ordering anchor in
      // multi-rank mode until remote completion acknowledgements are added.
      gpu_available_time[rank][proc] = 0.0;
      gpu_service_time_scale[rank][proc] =
          monitorServiceTimeScale(util, fresh_profile);
      gpu_memory_available_kib[rank][proc] =
          info == nullptr ? 0.0 : static_cast<double>(info->mem_available);

      DAEMON_TRACE_STREAM << "ensureOfflineDeviceModel: Rank " << rank
                << " Proc " << proc
                << " FP32Capability " << gpu_capability[rank][proc].fp32
                << " FP64Capability " << gpu_capability[rank][proc].fp64
                << " Util " << util
                << " FreshProfile " << (fresh_profile ? 1 : 0)
                << " ServiceTimeScale "
                << gpu_service_time_scale[rank][proc]
                << " AvailableTime " << gpu_available_time[rank][proc]
                << std::endl;
    }
  }
}

enum class KernelPrecision {
  unknown,
  fp32,
  fp64
};

static KernelPrecision inferKernelPrecisionFromReqs(
    const std::vector<SyclReqData> &reqs) {
  double fp32_elems = 0.0;
  double fp64_elems = 0.0;
  for (const SyclReqData &req : reqs) {
    if (req.elem_size == 4) {
      fp32_elems += reqAccessElems(req);
    } else if (req.elem_size == 8) {
      fp64_elems += reqAccessElems(req);
    }
  }

  if (fp64_elems == 0.0 && fp32_elems == 0.0) {
    return KernelPrecision::unknown;
  }
  return fp64_elems > fp32_elems ? KernelPrecision::fp64
                                 : KernelPrecision::fp32;
}

static const char *precisionName(KernelPrecision precision) {
  switch (precision) {
  case KernelPrecision::fp32:
    return "fp32";
  case KernelPrecision::fp64:
    return "fp64";
  case KernelPrecision::unknown:
    return "unknown";
  }
  return "unknown";
}

static double deviceCapability(int rank, int proc, KernelPrecision precision) {
  if (rank >= 0 && rank < static_cast<int>(gpu_capability.size()) &&
      proc >= 0 && proc < static_cast<int>(gpu_capability[rank].size()) &&
      (gpu_capability[rank][proc].fp32 > 0.0 ||
       gpu_capability[rank][proc].fp64 > 0.0)) {
    if (precision == KernelPrecision::fp64) {
      return gpu_capability[rank][proc].fp64;
    }
    return gpu_capability[rank][proc].fp32;
  }
  const ComputeCapability fallback = fallbackCapabilityForProc(proc);
  return precision == KernelPrecision::fp64 ? fallback.fp64 : fallback.fp32;
}

static bool rankHasGpuProc(int rank) {
  return rank >= 0 && rank < static_cast<int>(gpu_available_time.size()) &&
         gpu_available_time[rank].size() > 1;
}

static bool isKernelPlacementProc(int rank, int proc) {
  if (rankHasGpuProc(rank)) {
    return proc > 0;
  }
  return proc == 0;
}

static double profileEntrySourceCapability(const ProfileCostKey &key,
                                           const ProfileCostEntry &entry,
                                           KernelPrecision precision);

static bool lookupScaledProfileEstimate(const std::string &kernel_key,
                                        int rank, int device, int num_parts,
                                        bool persistent_split,
                                        KernelPrecision precision,
                                        CostEstimate &estimate) {
  if (lookupExactProfileEstimate(kernel_key, rank, device, num_parts,
                                 persistent_split, estimate)) {
    return true;
  }

  const int parts = std::max(1, num_parts);
  const double target_capability =
      std::max(0.1, deviceCapability(rank, device, precision));
  double weighted_sum = 0.0;
  double weighted_uncertainty = 0.0;
  int samples = 0;

  std::lock_guard<std::mutex> lock(profile_cost_mutex);
  for (const auto &entry : profile_cost_table) {
    const ProfileCostKey &sample_key = entry.first;
    if (sample_key.kernel_key != kernel_key || sample_key.num_parts != parts) {
      continue;
    }
    if (sample_key.persistent_split != persistent_split) {
      continue;
    }

    const double source_capability = profileEntrySourceCapability(
        sample_key, entry.second, precision);
    const double scaled_cost =
        entry.second.ewma_cost * source_capability / target_capability;
    const double scaled_uncertainty =
        profileEntryUncertainty(entry.second) * source_capability /
        target_capability;
    weighted_sum += scaled_cost * entry.second.samples;
    weighted_uncertainty += scaled_uncertainty * entry.second.samples;
    samples += entry.second.samples;
  }

  if (samples == 0) {
    return false;
  }

  estimate.mean = weighted_sum / samples;
  // Residuals from source devices alone cannot describe capability-model
  // error on the target. Preserve both terms instead of treating a scaled
  // profile as if it were an exact observation.
  estimate.uncertainty = std::hypot(
      weighted_uncertainty / samples,
      estimate.mean *
          (static_cast<double>(SNMD_OFFLINE_SCALED_PROFILE_ERROR_PERCENT) /
           100.0));
  estimate.samples = samples;
  estimate.source = CostEstimateSource::scaled_profile;
  return true;
}

static bool kernelFeaturesShareStructuralCohort(
    const KernelFeatureSignature &lhs, const KernelFeatureSignature &rhs) {
  return lhs.work_dim == rhs.work_dim && lhs.req_count == rhs.req_count &&
         lhs.read_req_count == rhs.read_req_count &&
         lhs.write_req_count == rhs.write_req_count &&
         lhs.dominant_elem_size == rhs.dominant_elem_size &&
         lhs.access_mode_mask == rhs.access_mode_mask &&
         lhs.partition_local_read == rhs.partition_local_read &&
         lhs.partition_local_write == rhs.partition_local_write;
}

static double logFeatureRatioDistance(double lhs, double rhs) {
  lhs = std::max(1.0, lhs);
  rhs = std::max(1.0, rhs);
  return std::abs(std::log(lhs / rhs));
}

static double kernelFeatureDistance(const KernelFeatureSignature &target,
                                    const KernelFeatureSignature &sample) {
  const double global_distance =
      logFeatureRatioDistance(target.global_items, sample.global_items);
  const double access_distance = logFeatureRatioDistance(
      target.total_access_elems, sample.total_access_elems);
  const double read_distance =
      logFeatureRatioDistance(target.read_bytes + 1.0,
                              sample.read_bytes + 1.0);
  const double write_distance =
      logFeatureRatioDistance(target.write_bytes + 1.0,
                              sample.write_bytes + 1.0);
  return std::sqrt(global_distance * global_distance +
                   access_distance * access_distance +
                   0.5 * read_distance * read_distance +
                   0.5 * write_distance * write_distance);
}

static double profileEntrySourceCapability(const ProfileCostKey &key,
                                           const ProfileCostEntry &entry,
                                           KernelPrecision precision) {
  const double stored = precision == KernelPrecision::fp64
                            ? entry.source_fp64_capability
                            : entry.source_fp32_capability;
  return stored > 0.0
             ? stored
             : std::max(0.1,
                        deviceCapability(key.rank, key.device, precision));
}

static bool lookupLearnedProfileEstimate(
    DAGNode *node, int rank, int device, int num_parts,
    bool persistent_split, KernelPrecision precision, CostEstimate &estimate) {
  if (node == nullptr) {
    return false;
  }
  const KernelFeatureSignature target = buildKernelFeature(node);
  const int parts = std::max(1, num_parts);
  const double target_capability =
      std::max(0.1, deviceCapability(rank, device, precision));

  struct Neighbor {
    const ProfileCostKey *key = nullptr;
    const ProfileCostEntry *entry = nullptr;
    const KernelFeatureSignature *feature = nullptr;
    double distance = 0.0;
    bool same_identity = false;
  };
  std::vector<Neighbor> neighbors;
  bool has_identity_neighbor = false;

  std::lock_guard<std::mutex> lock(profile_cost_mutex);
  for (const auto &profile_entry : profile_cost_table) {
    const ProfileCostKey &sample_key = profile_entry.first;
    const ProfileCostEntry &sample_entry = profile_entry.second;
    if (sample_entry.samples <= 0 || sample_key.num_parts != parts ||
        sample_key.persistent_split != persistent_split) {
      continue;
    }
    auto feature_it = kernel_feature_table.find(sample_key.kernel_key);
    if (feature_it == kernel_feature_table.end()) {
      continue;
    }
    const KernelFeatureSignature &sample_feature = feature_it->second;
    const bool same_identity = target.kernel_identity != 0 &&
                               target.kernel_identity ==
                                   sample_feature.kernel_identity;
    if (!same_identity &&
        !kernelFeaturesShareStructuralCohort(target, sample_feature)) {
      continue;
    }
    const double distance = kernelFeatureDistance(target, sample_feature);
    // Same-kernel strong scaling may span a wide range of input sizes.
    // Cross-kernel transfer is intentionally local because equal access modes
    // alone do not prove equal arithmetic intensity.
    if ((!same_identity && distance > 1.5) ||
        (same_identity && distance > 6.0)) {
      continue;
    }
    neighbors.push_back(
        {&sample_key, &sample_entry, &sample_feature, distance, same_identity});
    has_identity_neighbor = has_identity_neighbor || same_identity;
  }

  double total_weight = 0.0;
  double weighted_mean = 0.0;
  double weighted_second_moment = 0.0;
  int total_samples = 0;
  int used_neighbors = 0;
  for (const Neighbor &neighbor : neighbors) {
    if (has_identity_neighbor && !neighbor.same_identity) {
      continue;
    }
    const double source_capability = profileEntrySourceCapability(
        *neighbor.key, *neighbor.entry, precision);
    const double work_scale = std::max(
        0.05, std::min(20.0, target.analytical_work /
                                  std::max(1.0,
                                           neighbor.feature->analytical_work)));
    const double predicted_cost =
        neighbor.entry->ewma_cost * source_capability / target_capability *
        work_scale;
    const double scaled_profile_uncertainty =
        profileEntryUncertainty(*neighbor.entry) * source_capability /
        target_capability * work_scale;
    const double base_model_error =
        static_cast<double>(neighbor.same_identity
                                ? SNMD_OFFLINE_LEARNED_IDENTITY_ERROR_PERCENT
                                : SNMD_OFFLINE_LEARNED_STRUCTURAL_ERROR_PERCENT) /
        100.0;
    const double model_uncertainty =
        predicted_cost *
        std::min(1.5, base_model_error + 0.10 * neighbor.distance);
    const double neighbor_uncertainty =
        std::hypot(scaled_profile_uncertainty, model_uncertainty);
    const double weight =
        std::sqrt(static_cast<double>(neighbor.entry->samples)) *
        (neighbor.same_identity ? 4.0 : 1.0) /
        (1.0 + neighbor.distance * neighbor.distance);
    total_weight += weight;
    weighted_mean += weight * predicted_cost;
    weighted_second_moment +=
        weight * (predicted_cost * predicted_cost +
                  neighbor_uncertainty * neighbor_uncertainty);
    total_samples += neighbor.entry->samples;
    ++used_neighbors;
  }

  if (total_weight <= 0.0 || used_neighbors == 0) {
    return false;
  }
  estimate.mean = weighted_mean / total_weight;
  const double second_moment = weighted_second_moment / total_weight;
  estimate.uncertainty =
      std::sqrt(std::max(0.0, second_moment - estimate.mean * estimate.mean));
  estimate.samples = total_samples;
  estimate.source = CostEstimateSource::learned_profile;
  DAEMON_TRACE_STREAM
      << "LearnedProfile: kernel " << node->kernel_count << " parts " << parts
      << " persistent " << persistent_split << " identity_neighbors "
      << (has_identity_neighbor ? 1 : 0) << " neighbors " << used_neighbors
      << " samples " << total_samples << " mean " << estimate.mean
      << " uncertainty " << estimate.uncertainty << std::endl;
  return true;
}

static double deviceServiceTimeScale(int rank, int proc) {
  if (rank < 0 || rank >= static_cast<int>(gpu_service_time_scale.size()) ||
      proc < 0 ||
      proc >= static_cast<int>(gpu_service_time_scale[rank].size())) {
    return 1.0;
  }
  return std::max(1.0, gpu_service_time_scale[rank][proc]);
}

static bool monitorMemoryFits(const DAGNode *node, int rank, int proc,
                              int num_parts) {
  // An exact successful profile proves that this already-materialized mode fit
  // on the device. Until residency-aware allocation deltas are tracked, do not
  // double-count its existing buffers against current free memory.
  if (proc > 0 &&
      (hasLiveExactProfileCost(profileKeyForNode(node), rank, proc, num_parts,
                               false) ||
       hasLiveExactProfileCost(profileKeyForNode(node), rank, proc, num_parts,
                               true))) {
    return true;
  }

  if (rank < 0 || rank >= static_cast<int>(gpu_memory_available_kib.size()) ||
      proc < 0 ||
      proc >= static_cast<int>(gpu_memory_available_kib[rank].size()) ||
      gpu_memory_available_kib[rank][proc] <= 0.0) {
    return true;
  }

  // Split currently keeps a full virtual buffer allocation on every part
  // device so an unchanged kernel can continue to index by global id. Only
  // data validity and transfers are partitioned; counting writable storage as
  // bytes/parts would admit a gang that can OOM during allocation.
  const double required_bytes = totalReqBytes(node);
  const double available_bytes =
      gpu_memory_available_kib[rank][proc] * 1024.0;
  return required_bytes < available_bytes * 0.85;
}

static bool monitorCoLocatedBatchMemoryFits(
    const std::vector<DAGNode *> &nodes, int rank, int proc) {
  if (rank < 0 || rank >= static_cast<int>(gpu_memory_available_kib.size()) ||
      proc < 0 ||
      proc >= static_cast<int>(gpu_memory_available_kib[rank].size()) ||
      gpu_memory_available_kib[rank][proc] <= 0.0) {
    return true;
  }

  // Co-location keeps every distinct buffer used by the batch in one private
  // device context. Per-kernel fit is insufficient when independent chains
  // own disjoint allocations, so conservatively count each buffer once.
  std::map<uintptr_t, double> unique_buffer_bytes;
  for (const DAGNode *node : nodes) {
    for (const SyclReqData &req : node->req_data) {
      const uintptr_t key = reinterpret_cast<uintptr_t>(req.mem_pointer);
      unique_buffer_bytes[key] =
          std::max(unique_buffer_bytes[key], reqBytes(req));
    }
  }

  double required_bytes = 0.0;
  for (const auto &entry : unique_buffer_bytes) {
    required_bytes += entry.second;
  }
  const double available_bytes =
      gpu_memory_available_kib[rank][proc] * 1024.0;
  return required_bytes < available_bytes * 0.85;
}

static CostEstimate estimateSingleExecCost(DAGNode *node, int rank, int proc) {
  CostEstimate estimate;
  const std::string key = profileKeyForNode(node);
  const KernelPrecision precision = inferKernelPrecisionFromReqs(node->req_data);
  if (lookupScaledProfileEstimate(key, rank, proc, 1, false, precision,
                                  estimate)) {
    const double scale = deviceServiceTimeScale(rank, proc);
    estimate.mean = std::max(0.001, estimate.mean) * scale;
    estimate.uncertainty *= scale;
    return estimate;
  }
  if (lookupLearnedProfileEstimate(node, rank, proc, 1, false, precision,
                                   estimate)) {
    const double scale = deviceServiceTimeScale(rank, proc);
    estimate.mean = std::max(0.001, estimate.mean) * scale;
    estimate.uncertainty *= scale;
    return estimate;
  }

  const double cold_cost = coldWorkElems(node) /
                           std::max(0.1, deviceCapability(rank, proc,
                                                          precision));
  estimate.mean = cold_cost * deviceServiceTimeScale(rank, proc);
  estimate.uncertainty =
      std::max(0.001, estimate.mean) *
      (static_cast<double>(SNMD_OFFLINE_COLD_MODEL_ERROR_PERCENT) / 100.0);
  estimate.source = CostEstimateSource::cold_model;
  return estimate;
}

static std::vector<int> producerSourceProcs(const DAGNode *pre_node) {
  std::vector<int> source_procs;
  if (pre_node->num_parts > 1 && !pre_node->split_devices.empty()) {
    if (pre_node->persistent_split) {
      return pre_node->split_devices;
    }
#ifdef SNMD_OFFLINE_CANONICAL_MERGE
    const bool exec_proc_is_part =
        std::find(pre_node->split_devices.begin(),
                  pre_node->split_devices.end(), pre_node->exec_proc) !=
        pre_node->split_devices.end();
    source_procs.push_back(exec_proc_is_part
                               ? pre_node->exec_proc
                               : pre_node->split_devices.front());
#else
    source_procs = pre_node->split_devices;
#endif
  } else if (pre_node->exec_proc >= 0) {
    source_procs.push_back(pre_node->exec_proc);
  }
  return source_procs;
}

static double estimateCommCostForDevices(DAGNode *node, DAGNode *pre_node,
                                         int rank,
                                         const std::vector<int> &target_procs) {
  if (pre_node->exec_rank < 0) {
    return 0.0;
  }

  const double comm_bytes = getCommBytes(node, pre_node);
  if (comm_bytes == 0.0 || target_procs.empty()) {
    return 0.0;
  }

  if (pre_node->exec_rank != rank) {
    return heftCostFromSeconds(secondsForBytesAtBandwidth(
        comm_bytes, crossRankBandwidthGiB(pre_node->exec_rank, rank)));
  }

  if (pre_node->persistent_split && pre_node->num_parts > 1 &&
      !pre_node->split_devices.empty()) {
    // The handler always gathers to the producer's first ordered Split
    // device, then serves an incompatible consumer from that complete
    // canonical version. Keep the estimator and endpoint reservations aligned
    // with that physical path even when the consumer's first device differs.
    const int canonical_proc = pre_node->split_devices.front();
    const double part_bytes =
        comm_bytes / static_cast<double>(pre_node->num_parts);
    double seconds = 0.0;
    for (int src_proc : pre_node->split_devices) {
      if (src_proc != canonical_proc) {
        seconds += sameRankCopySeconds(rank, src_proc, canonical_proc,
                                       part_bytes);
      }
    }
    for (int target_proc : target_procs) {
      if (target_proc != canonical_proc) {
        seconds += sameRankCopySeconds(rank, canonical_proc, target_proc,
                                       comm_bytes);
      }
    }
    return heftCostFromSeconds(seconds);
  }

  // The handler exposes one complete version on the canonical merge device;
  // other Split devices contain only their owned partition.
  const std::vector<int> source_procs = producerSourceProcs(pre_node);

  double total_seconds = 0.0;
  for (int dst_proc : target_procs) {
    bool already_local = false;
    for (int src_proc : source_procs) {
      if (src_proc == dst_proc) {
        already_local = true;
        break;
      }
    }
    if (already_local) {
      continue;
    }

    double best_seconds = std::numeric_limits<double>::infinity();
    for (int src_proc : source_procs) {
      best_seconds =
          std::min(best_seconds,
                   sameRankCopySeconds(rank, src_proc, dst_proc, comm_bytes));
    }
    if (std::isfinite(best_seconds)) {
      total_seconds += best_seconds;
    }
  }

  return heftCostFromSeconds(total_seconds);
}

static double calendarReadyTime(
    const std::vector<std::vector<double>> &calendar, int rank, int proc) {
  if (rank < 0 || rank >= static_cast<int>(calendar.size()) || proc < 0 ||
      proc >= static_cast<int>(calendar[rank].size())) {
    return 0.0;
  }
  return calendar[rank][proc];
}

static void reserveCalendar(std::vector<std::vector<double>> &calendar,
                            int rank, int proc, double ready_time) {
  if (rank < 0 || rank >= static_cast<int>(calendar.size()) || proc < 0 ||
      proc >= static_cast<int>(calendar[rank].size())) {
    return;
  }
  calendar[rank][proc] = std::max(calendar[rank][proc], ready_time);
}

static void addTransferEstimate(DependencyTransferPlan &plan, double cost) {
  if (!std::isfinite(cost) || cost <= 0.0) {
    return;
  }
  plan.estimated_cost += cost;
  const double uncertainty =
      cost *
      (static_cast<double>(SNMD_OFFLINE_TRANSFER_ERROR_PERCENT) / 100.0);
  plan.uncertainty = std::hypot(plan.uncertainty, uncertainty);
}

static bool persistentEdgeCompatible(
    const DAGNode *node, const DAGNode *pre_node, int target_rank,
    const std::vector<int> &target_procs);

static DependencyTransferPlan buildDependencyTransferPlan(
    DAGNode *node, int target_rank, const std::vector<int> &target_procs) {
  DependencyTransferPlan plan;
  if (target_procs.empty()) {
    return plan;
  }

  std::vector<std::vector<double>> transfer_calendar = gpu_available_time;
  for (DAGNode *pre_node : node->depend_on) {
    if (pre_node == nullptr || pre_node->exec_rank < 0) {
      continue;
    }

    const double comm_bytes = getCommBytes(node, pre_node);
    plan.ready_time = std::max(plan.ready_time, pre_node->finish_time);
    if (comm_bytes <= 0.0) {
      continue;
    }

    if (persistentEdgeCompatible(node, pre_node, target_rank, target_procs)) {
      DAEMON_TRACE_STREAM
          << "buildDependencyTransferPlan: Kernel " << node->kernel_count
          << " consumes resident partitions from Kernel "
          << pre_node->kernel_count << " without materialization" << std::endl;
      continue;
    }

    if (pre_node->persistent_split && pre_node->num_parts > 1) {
      // An incompatible consumer forces the handler's canonical fallback.
      // Model the gather plus any full replicas as one conservative transfer
      // and reserve every hidden materialization endpoint for this wave.
      double transfer_start = pre_node->finish_time;
      for (int src_proc : pre_node->split_devices) {
        transfer_start = std::max(
            transfer_start,
            calendarReadyTime(transfer_calendar, pre_node->exec_rank,
                              src_proc));
      }
      for (int dst_proc : target_procs) {
        transfer_start = std::max(
            transfer_start,
            calendarReadyTime(transfer_calendar, target_rank, dst_proc));
      }
      if (!std::isfinite(transfer_start)) {
        // A compatible successor selected earlier in this dispatch wave can
        // already own the resident source queues. Delay this incompatible
        // gather until that successor completes instead of admitting two
        // commands whose hidden materialization endpoints conflict.
        plan.ready_time = std::numeric_limits<double>::infinity();
        continue;
      }
      const double transfer_cost = estimateCommCostForDevices(
          node, pre_node, target_rank, target_procs);
      const double transfer_end = transfer_start + transfer_cost;
      plan.ready_time = std::max(plan.ready_time, transfer_end);
      addTransferEstimate(plan, transfer_cost);
      for (int src_proc : pre_node->split_devices) {
        reserveCalendar(transfer_calendar, pre_node->exec_rank, src_proc,
                        transfer_end);
      }
      for (int dst_proc : target_procs) {
        reserveCalendar(transfer_calendar, target_rank, dst_proc,
                        transfer_end);
      }
      const double gather_bytes =
          comm_bytes * static_cast<double>(pre_node->num_parts - 1) /
          static_cast<double>(pre_node->num_parts);
      size_t replica_count = 0;
      const int canonical_proc = pre_node->split_devices.empty()
                                     ? -1
                                     : pre_node->split_devices.front();
      for (int dst_proc : target_procs) {
        replica_count += dst_proc == canonical_proc ? 0 : 1;
      }
      const double replica_bytes =
          comm_bytes * static_cast<double>(replica_count);
      plan.movement_bytes += gather_bytes + replica_bytes;
      continue;
    }

    const std::vector<int> source_procs = producerSourceProcs(pre_node);
    if (source_procs.empty()) {
      const double transfer_cost = estimateCommCostForDevices(
          node, pre_node, target_rank, target_procs);
      plan.ready_time = std::max(
          plan.ready_time, pre_node->finish_time + transfer_cost);
      addTransferEstimate(plan, transfer_cost);
      plan.movement_bytes += comm_bytes;
      continue;
    }

    if (pre_node->exec_rank != target_rank) {
      // Cross-rank data is materialized once on the primary target. Split's
      // additional local replication is accounted for by its internal-copy
      // model.
      const int dst_proc = target_procs.front();
      double best_end = std::numeric_limits<double>::infinity();
      double best_transfer_cost = 0.0;
      int best_src_proc = -1;
      for (int src_proc : source_procs) {
        const double transfer_start = std::max(
            {pre_node->finish_time,
             calendarReadyTime(transfer_calendar, pre_node->exec_rank,
                               src_proc),
             calendarReadyTime(transfer_calendar, target_rank, dst_proc)});
        const double transfer_cost =
            heftCostFromSeconds(secondsForBytesAtBandwidth(
                comm_bytes,
                crossRankBandwidthGiB(pre_node->exec_rank, target_rank)));
        const double transfer_end = transfer_start + transfer_cost;
        if (transfer_end < best_end) {
          best_end = transfer_end;
          best_transfer_cost = transfer_cost;
          best_src_proc = src_proc;
        }
      }
      if (std::isfinite(best_end)) {
        reserveCalendar(transfer_calendar, pre_node->exec_rank, best_src_proc,
                        best_end);
        reserveCalendar(transfer_calendar, target_rank, dst_proc, best_end);
        plan.ready_time = std::max(plan.ready_time, best_end);
        addTransferEstimate(plan, best_transfer_cost);
        plan.movement_bytes += comm_bytes;
      } else {
        // No reachable source/target endpoint pair exists for this placement.
        // Do not silently drop the transfer and dispatch against a different
        // target; keep the candidate infeasible for this admission pass.
        plan.ready_time = std::numeric_limits<double>::infinity();
      }
      continue;
    }

    for (int dst_proc : target_procs) {
      if (std::find(source_procs.begin(), source_procs.end(), dst_proc) !=
          source_procs.end()) {
        continue;
      }

      double best_end = std::numeric_limits<double>::infinity();
      double best_transfer_cost = 0.0;
      int best_src_proc = -1;
      for (int src_proc : source_procs) {
        const double transfer_start = std::max(
            {pre_node->finish_time,
             calendarReadyTime(transfer_calendar, target_rank, src_proc),
             calendarReadyTime(transfer_calendar, target_rank, dst_proc)});
        const double transfer_cost = heftCostFromSeconds(sameRankCopySeconds(
            target_rank, src_proc, dst_proc, comm_bytes));
        const double transfer_end = transfer_start + transfer_cost;
        if (transfer_end < best_end) {
          best_end = transfer_end;
          best_transfer_cost = transfer_cost;
          best_src_proc = src_proc;
        }
      }
      if (std::isfinite(best_end)) {
        reserveCalendar(transfer_calendar, target_rank, best_src_proc,
                        best_end);
        reserveCalendar(transfer_calendar, target_rank, dst_proc, best_end);
        plan.ready_time = std::max(plan.ready_time, best_end);
        addTransferEstimate(plan, best_transfer_cost);
        plan.movement_bytes += comm_bytes;
      } else {
        plan.ready_time = std::numeric_limits<double>::infinity();
      }
    }
  }

  for (int rank = 0; rank < static_cast<int>(transfer_calendar.size()); ++rank) {
    for (int proc = 0;
         proc < static_cast<int>(transfer_calendar[rank].size()); ++proc) {
      const double original_ready =
          calendarReadyTime(gpu_available_time, rank, proc);
      if (transfer_calendar[rank][proc] > original_ready) {
        plan.reservations.push_back(
            TaskCandidate::DeviceReservation{rank, proc,
                                             transfer_calendar[rank][proc]});
      }
    }
  }
  return plan;
}

static bool splitWriteRangesMatchDim0(const DAGNode *node, int num_parts) {
  if (node->global_size0 < static_cast<size_t>(num_parts) ||
      node->global_size0 % static_cast<size_t>(num_parts) != 0) {
    return false;
  }

  const bool kernel_splits_only_dim0 =
      node->global_size0 > 1 && node->global_size1 <= 1 &&
      node->global_size2 <= 1;

  for (const SyclReqData &req : node->req_data) {
    if (!isWriteAccess(req.req_accmode)) {
      continue;
    }

    if (req.is_sub_buffer || req.access_range0 != node->global_size0 ||
        req.access_range0 < static_cast<size_t>(num_parts) ||
        req.access_range0 % static_cast<size_t>(num_parts) != 0 ||
        req.offset1 != 0 || req.offset2 != 0 ||
        req.access_range1 != req.range1 || req.access_range2 != req.range2) {
      return false;
    }

    if (kernel_splits_only_dim0 &&
        (req.access_range1 > 1 || req.access_range2 > 1)) {
      DAEMON_TRACE_STREAM
          << "algorithmHEFT: Kernel " << node->kernel_count
          << " split rejected: dim0-only kernel writes non-contiguous range "
          << req.access_range0 << "x" << req.access_range1 << "x"
          << req.access_range2
          << std::endl;
      return false;
    }
  }

  return true;
}

static bool worthConsideringSplit(DAGNode *node, int num_parts) {
#ifdef SNMD_OFFLINE_TEST_DISABLE_SPLIT
  (void)node;
  (void)num_parts;
  return false;
#else
  static const bool split_enabled = [] {
    const char *env = std::getenv("SYCL_SNMD_ENABLE_SPLIT");
    if (env == nullptr) {
      return SNMD_OFFLINE_SPLIT_DEFAULT_ENABLED != 0;
    }
    return std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
           std::strcmp(env, "TRUE") == 0;
  }();
  if (!split_enabled) {
    return false;
  }
  if (num_parts <= 1) {
    return false;
  }
  if (num_parts % 2 != 0) {
    return false;
  }
  if (totalWriteElems(node) == 0.0) {
    return false;
  }
  if (totalReqElems(node) < SPLIT_MIN_ELEMS) {
    return false;
  }
  if (!splitWriteRangesMatchDim0(node, num_parts)) {
    return false;
  }
  return true;
#endif
}

static bool supportsPersistentSplit(const DAGNode *node, int num_parts) {
  if (node == nullptr || num_parts <= 1 ||
      node->global_size0 < static_cast<size_t>(num_parts) ||
      node->global_size0 % static_cast<size_t>(num_parts) != 0) {
    return false;
  }

  bool has_write = false;
  for (const SyclReqData &req : node->req_data) {
    if (!isWriteAccess(req.req_accmode)) {
      if (!req.partition_local) {
        continue;
      }
    } else {
      has_write = true;
      // Cross-partition atomics require a reduction/ownership protocol that the
      // dim-0 resident mode intentionally does not claim to provide.
      if (!req.partition_local || req.req_accmode == acc_mode::atomic) {
        return false;
      }
    }
    if (req.partition_local &&
        (req.is_sub_buffer || req.offset0 != 0 || req.offset1 != 0 ||
         req.offset2 != 0 || req.access_range0 != node->global_size0 ||
         req.access_range0 < static_cast<size_t>(num_parts) ||
         req.access_range0 % static_cast<size_t>(num_parts) != 0 ||
         req.access_range1 != req.range1 ||
         req.access_range2 != req.range2)) {
      return false;
    }
  }
  return has_write;
}

static bool hasPartitionLocalSuccessor(const DAGNode *node, int num_parts) {
  if (node == nullptr) {
    return false;
  }
  for (const DAGNode *successor : node->depend_by) {
    if (!supportsPersistentSplit(successor, num_parts) ||
        successor->global_size0 != node->global_size0) {
      continue;
    }
    const auto dep_it = successor->depend_on_mem.find(
        const_cast<DAGNode *>(node));
    if (dep_it == successor->depend_on_mem.end() || dep_it->second.empty()) {
      continue;
    }
    const bool all_partition_local =
        std::all_of(dep_it->second.begin(), dep_it->second.end(),
                    [](const SyclReqData &req) {
                      return req.partition_local;
                    });
    if (all_partition_local) {
      return true;
    }
  }
  return false;
}

static bool persistentEdgeCompatible(
    const DAGNode *node, const DAGNode *pre_node, int target_rank,
    const std::vector<int> &target_procs) {
  if (node == nullptr || pre_node == nullptr || !pre_node->persistent_split ||
      !supportsPersistentSplit(node, pre_node->num_parts) ||
      pre_node->exec_rank != target_rank ||
      pre_node->num_parts <= 1 ||
      pre_node->global_size0 != node->global_size0 ||
      static_cast<int>(target_procs.size()) != pre_node->num_parts ||
      pre_node->split_devices != target_procs) {
    return false;
  }

  const auto dep_it = node->depend_on_mem.find(
      const_cast<DAGNode *>(pre_node));
  if (dep_it == node->depend_on_mem.end() || dep_it->second.empty()) {
    return false;
  }
  for (const SyclReqData &req : dep_it->second) {
    if (!req.partition_local) {
      return false;
    }
  }
  return true;
}

static int countMaskBits(uint64_t mask) {
  int count = 0;
  while (mask != 0) {
    count += static_cast<int>(mask & 1ULL);
    mask >>= 1;
  }
  return count;
}

static double estimateSplitInternalCopyCost(
    DAGNode *node, int rank, const std::vector<int> &split_devices,
    bool persistent_split) {
  if (split_devices.size() <= 1) {
    return 0.0;
  }

  const int main_proc = split_devices.front();
  const double replicated_read_bytes =
      std::max(0.0, totalReadBytesForPartitionMode(node, false) -
                        dependentReadBytesForPartitionMode(node, false));
  const double partition_local_read_bytes =
      std::max(0.0, totalReadBytesForPartitionMode(node, true) -
                        dependentReadBytesForPartitionMode(node, true));
  const double read_bytes_per_extra_device =
      replicated_read_bytes +
      partition_local_read_bytes / static_cast<double>(split_devices.size());
  const double write_part_bytes =
      persistent_split
          ? 0.0
          : totalWriteBytes(node) /
                static_cast<double>(split_devices.size());

  double seconds = 0.0;
  for (size_t i = 1; i < split_devices.size(); ++i) {
    const int proc = split_devices[i];
    seconds += sameRankCopySeconds(rank, main_proc, proc,
                                   read_bytes_per_extra_device);
    seconds += sameRankCopySeconds(rank, proc, main_proc, write_part_bytes);
  }

  return heftCostFromSeconds(seconds);
}

static CostEstimate
estimateSplitExecCost(DAGNode *node, int rank,
                      const std::vector<int> &split_devices,
                      bool persistent_split) {
  const int num_parts = static_cast<int>(split_devices.size());
  if (num_parts <= 1) {
    return estimateSingleExecCost(
        node, rank, split_devices.empty() ? 1 : split_devices.front());
  }

  CostEstimate estimate;
  const std::string key = profileKeyForNode(node);
  const KernelPrecision precision = inferKernelPrecisionFromReqs(node->req_data);
  if (lookupScaledProfileEstimate(key, rank, split_devices.front(), num_parts,
                                  persistent_split, precision, estimate)) {
    double penalty = 1.0;
    for (int proc : split_devices) {
      penalty = std::max(penalty, deviceServiceTimeScale(rank, proc));
    }
    estimate.mean = std::max(0.001, estimate.mean) * penalty;
    estimate.uncertainty *= penalty;
    return estimate;
  }
  if (lookupLearnedProfileEstimate(node, rank, split_devices.front(),
                                   num_parts, persistent_split, precision,
                                   estimate)) {
    double penalty = 1.0;
    for (int proc : split_devices) {
      penalty = std::max(penalty, deviceServiceTimeScale(rank, proc));
    }
    estimate.mean = std::max(0.001, estimate.mean) * penalty;
    estimate.uncertainty *= penalty;
    return estimate;
  }

  CostEstimate best_single;
  for (int proc : split_devices) {
    CostEstimate single = estimateSingleExecCost(node, rank, proc);
    if (riskAdjustedCost(single) < riskAdjustedCost(best_single)) {
      best_single = single;
    }
  }

  if (!std::isfinite(best_single.mean)) {
    best_single = estimateSingleExecCost(node, rank, 1);
  }

  const double copy_overhead =
      estimateSplitInternalCopyCost(node, rank, split_devices,
                                    persistent_split);
  const double launch_overhead = 0.2 * num_parts;
  estimate.mean = best_single.mean / (num_parts * SPLIT_EFFICIENCY) +
                  copy_overhead + launch_overhead;
  const double propagated_single_uncertainty =
      best_single.uncertainty / (num_parts * SPLIT_EFFICIENCY);
  const double split_model_uncertainty =
      estimate.mean *
      (static_cast<double>(SNMD_OFFLINE_DERIVED_SPLIT_ERROR_PERCENT) / 100.0);
  estimate.uncertainty =
      std::hypot(propagated_single_uncertainty, split_model_uncertainty);
  estimate.samples = best_single.samples;
  estimate.source = CostEstimateSource::derived_split;
  return estimate;
}

static TaskCandidate makeSingleCandidate(DAGNode *node, int rank, int proc) {
  TaskCandidate candidate;
  candidate.rank = rank;
  candidate.proc = proc;
  candidate.num_parts = 1;
  candidate.occupied_procs.push_back(proc);

  if (!isKernelPlacementProc(rank, proc)) {
    return candidate;
  }
  if (!monitorMemoryFits(node, rank, proc, 1)) {
    return candidate;
  }

  const double device_ready = gpu_available_time[rank][proc];
  const DependencyTransferPlan transfer_plan =
      buildDependencyTransferPlan(node, rank, std::vector<int>{proc});
  candidate.start_time = std::max(device_ready, transfer_plan.ready_time);
  candidate.exec_estimate = estimateSingleExecCost(node, rank, proc);
  candidate.transfer_uncertainty = transfer_plan.uncertainty;
  candidate.movement_bytes = transfer_plan.movement_bytes;
  candidate.transfer_reservations = transfer_plan.reservations;
  candidate.finish_time = candidate.start_time + candidate.exec_estimate.mean;
  return candidate;
}

static TaskCandidate makeSingleCandidateNoMemoryFilter(DAGNode *node, int rank,
                                                       int proc) {
  TaskCandidate candidate;
  candidate.rank = rank;
  candidate.proc = proc;
  candidate.num_parts = 1;
  candidate.occupied_procs.push_back(proc);

  if (!isKernelPlacementProc(rank, proc)) {
    return candidate;
  }

  const double device_ready = gpu_available_time[rank][proc];
  const DependencyTransferPlan transfer_plan =
      buildDependencyTransferPlan(node, rank, std::vector<int>{proc});
  candidate.start_time = std::max(device_ready, transfer_plan.ready_time);
  candidate.exec_estimate = estimateSingleExecCost(node, rank, proc);
  candidate.transfer_uncertainty = transfer_plan.uncertainty;
  candidate.movement_bytes = transfer_plan.movement_bytes;
  candidate.transfer_reservations = transfer_plan.reservations;
  candidate.finish_time = candidate.start_time + candidate.exec_estimate.mean;
  return candidate;
}

// A co-located batch is submitted to one out-of-order device queue.  Its
// independent kernels must therefore not inherit the exclusive device
// calendar used by ordinary HEFT.  Data predecessors and transfer endpoints
// are still modeled exactly as for a regular Single candidate.
static TaskCandidate makeConcurrentSingleCandidate(DAGNode *node, int rank,
                                                   int proc) {
  TaskCandidate candidate;
  candidate.rank = rank;
  candidate.proc = proc;
  candidate.num_parts = 1;
  candidate.occupied_procs.push_back(proc);

  if (!isKernelPlacementProc(rank, proc)) {
    return candidate;
  }
  if (!monitorMemoryFits(node, rank, proc, 1)) {
    return candidate;
  }

  const DependencyTransferPlan transfer_plan =
      buildDependencyTransferPlan(node, rank, std::vector<int>{proc});
  candidate.start_time = transfer_plan.ready_time;
  candidate.exec_estimate = estimateSingleExecCost(node, rank, proc);
  candidate.transfer_uncertainty = transfer_plan.uncertainty;
  candidate.movement_bytes = transfer_plan.movement_bytes;
  candidate.transfer_reservations = transfer_plan.reservations;
  candidate.finish_time = candidate.start_time + candidate.exec_estimate.mean;
  return candidate;
}

static void commitCandidateReservations(const TaskCandidate &candidate) {
  for (const TaskCandidate::DeviceReservation &reservation :
       candidate.transfer_reservations) {
    if (reservation.rank < 0 ||
        reservation.rank >= static_cast<int>(gpu_available_time.size()) ||
        reservation.proc < 0 ||
        reservation.proc >=
            static_cast<int>(gpu_available_time[reservation.rank].size())) {
      continue;
    }
    gpu_available_time[reservation.rank][reservation.proc] =
        std::max(gpu_available_time[reservation.rank][reservation.proc],
                 reservation.ready_time);
  }
}

static std::vector<NodePlacementState>
saveNodePlacementStates(const std::vector<DAGNode *> &nodes) {
  std::vector<NodePlacementState> states;
  states.reserve(nodes.size());
  for (const DAGNode *node : nodes) {
    states.push_back(NodePlacementState{
        node->exec_rank, node->exec_proc, node->num_parts,
        node->persistent_split, node->finish_time, node->split_devices});
  }
  return states;
}

static void restoreNodePlacementStates(
    const std::vector<DAGNode *> &nodes,
    const std::vector<NodePlacementState> &states) {
  for (size_t i = 0; i < nodes.size() && i < states.size(); ++i) {
    DAGNode *node = nodes[i];
    const NodePlacementState &state = states[i];
    node->exec_rank = state.exec_rank;
    node->exec_proc = state.exec_proc;
    node->num_parts = state.num_parts;
    node->persistent_split = state.persistent_split;
    node->finish_time = state.finish_time;
    node->split_devices = state.split_devices;
  }
}

static double batchFinishTime(const std::vector<DAGNode *> &nodes) {
  double finish = 0.0;
  for (const DAGNode *node : nodes) {
    finish = std::max(finish, node->finish_time);
  }
  return finish;
}

static std::vector<DAGNode *>
topologicalOrderForCurrentBatch(const std::vector<DAGNode *> &nodes) {
  std::vector<DAGNode *> order = reverseTopologicalOrder(nodes);
  std::reverse(order.begin(), order.end());
  return order;
}

// Whole-batch co-location is not a safe replacement when HEFT has already
// placed independent components (for example, observation tiles) across
// multiple GPUs. Collapsing those components to one GPU discards physical
// device parallelism based only on the approximate stream-concurrency model.
static bool heftAlreadyParallelizesIndependentComponents(
    const std::vector<DAGNode *> &topo_order,
    const std::unordered_set<DAGNode *> &current_nodes,
    size_t &component_count, size_t &device_count) {
  std::unordered_map<DAGNode *, size_t> component_ids;
  component_count = 0;

  for (DAGNode *seed : topo_order) {
    if (component_ids.count(seed)) {
      continue;
    }

    const size_t component_id = component_count++;
    std::vector<DAGNode *> pending{seed};
    component_ids.emplace(seed, component_id);
    while (!pending.empty()) {
      DAGNode *node = pending.back();
      pending.pop_back();

      auto add_neighbor = [&](DAGNode *neighbor) {
        if (neighbor != nullptr && current_nodes.count(neighbor) &&
            component_ids.emplace(neighbor, component_id).second) {
          pending.push_back(neighbor);
        }
      };
      for (DAGNode *predecessor : node->depend_on) {
        add_neighbor(predecessor);
      }
      for (DAGNode *successor : node->depend_by) {
        add_neighbor(successor);
      }
    }
  }

  if (component_count < 2) {
    device_count = 0;
    return false;
  }

  std::vector<std::pair<int, int>> component_devices(
      component_count, std::make_pair(-1, -1));

  for (DAGNode *node : topo_order) {
    const size_t component_id = component_ids.at(node);
    // A split or a component that HEFT moves between devices is not already a
    // one-device component and may still benefit from another placement shape.
    if (node->num_parts != 1 || node->exec_rank < 0 || node->exec_proc <= 0) {
      device_count = 0;
      return false;
    }
    const std::pair<int, int> placement{node->exec_rank, node->exec_proc};
    if (component_devices[component_id].first < 0) {
      component_devices[component_id] = placement;
    } else if (component_devices[component_id] != placement) {
      device_count = 0;
      return false;
    }
  }

  const std::set<std::pair<int, int>> distinct_devices(
      component_devices.begin(), component_devices.end());
  device_count = distinct_devices.size();
  return device_count > 1;
}

static bool independentComponentsShareOneGpu(
    const std::vector<DAGNode *> &nodes, size_t &component_count) {
  const std::vector<DAGNode *> topo_order =
      topologicalOrderForCurrentBatch(nodes);
  const std::unordered_set<DAGNode *> current_nodes(nodes.begin(), nodes.end());
  size_t device_count = 0;
  (void)heftAlreadyParallelizesIndependentComponents(
      topo_order, current_nodes, component_count, device_count);
  return component_count > 1 && device_count == 1;
}

static double coLocatedOccupancyDemand(const DAGNode *node,
                                       double target_items) {
  const double global_items =
      std::max(1.0, static_cast<double>(node->global_size0) *
                        static_cast<double>(node->global_size1) *
                        static_cast<double>(node->global_size2));

  // Accessor footprint is not a residency measure. Row correlation and classic
  // MGS touch large buffers through comparatively few work-items, and
  // independent tiles can therefore co-reside on one GPU. Keep the demand
  // proxy tied to schedulable work-items; the independent-component guard
  // separately protects physical parallelism already established by HEFT.
  return std::min(1.0, global_items / target_items);
}

static void rebuildKernelSchedInfos(
    const std::vector<DAGNode *> &order,
    std::vector<D2DKernelSchedInfo> &kernel_sched_order_infos) {
  kernel_sched_order_infos.clear();
  int exec_order = 1;
  for (DAGNode *node : order) {
    D2DKernelSchedInfo kernel_sched_info;
    kernel_sched_info.kernel_count = node->kernel_count;
    kernel_sched_info.exec_order = exec_order++;
    kernel_sched_info.exec_rank = node->exec_rank;
    kernel_sched_info.exec_device = node->exec_proc;
    kernel_sched_info.num_parts = node->num_parts;
    kernel_sched_info.persistent_split = node->persistent_split;
    kernel_sched_info.split_devices = node->split_devices;
    kernel_sched_order_infos.push_back(kernel_sched_info);
  }
}

static bool applyCoLocatedGpuScheduleIfBetter(
    const std::vector<DAGNode *> &nodes,
    const std::vector<NodePlacementState> &initial_node_states,
    const std::vector<std::vector<double>> &initial_available_time,
    double heft_finish_time, double heft_risk_finish_time,
    const std::vector<DAGNode *> &topo_order,
    const std::unordered_set<DAGNode *> &current_nodes,
    bool heft_parallelizes_independent_components,
    size_t independent_component_count,
    size_t independent_component_device_count,
    std::vector<D2DKernelSchedInfo> &kernel_sched_order_infos) {
  const double target_items = concurrentGpuTargetItems();
  if (nodes.size() < 2 || target_items <= 0.0) {
    return false;
  }

  const std::vector<NodePlacementState> heft_states =
      saveNodePlacementStates(nodes);
  const std::vector<std::vector<double>> heft_available_time =
      gpu_available_time;
  std::map<DAGNode *, int> batch_depth;
  std::map<int, std::vector<DAGNode *>> depth_nodes;
  for (DAGNode *node : topo_order) {
    int depth = 0;
    for (DAGNode *predecessor : node->depend_on) {
      if (current_nodes.count(predecessor)) {
        depth = std::max(depth, batch_depth[predecessor] + 1);
      }
    }
    batch_depth[node] = depth;
    depth_nodes[depth].push_back(node);
  }

  bool has_concurrent_level = false;
  for (const auto &entry : depth_nodes) {
    if (entry.second.size() > 1) {
      has_concurrent_level = true;
      break;
    }
  }
  if (!has_concurrent_level) {
    return false;
  }

  if (heft_parallelizes_independent_components) {
    DAEMON_TRACE_STREAM
        << "algorithmHEFT: whole-batch co-location skipped; ordinary HEFT "
           "already maps "
        << independent_component_count << " independent components across "
        << independent_component_device_count << " distinct GPUs" << std::endl;
    return false;
  }

  std::vector<NodePlacementState> best_states;
  std::vector<std::vector<double>> best_available_time;
  double best_finish_time = std::numeric_limits<double>::infinity();
  double best_risk_finish_time = std::numeric_limits<double>::infinity();

  for (int rank = 0; rank < static_cast<int>(initial_available_time.size());
       ++rank) {
    for (int proc = 1;
         proc < static_cast<int>(initial_available_time[rank].size());
         ++proc) {
      if (!isKernelPlacementProc(rank, proc)) {
        continue;
      }
      if (!monitorCoLocatedBatchMemoryFits(nodes, rank, proc)) {
        continue;
      }

      restoreNodePlacementStates(nodes, initial_node_states);
      gpu_available_time = initial_available_time;

      bool valid = true;
      double finish_time = 0.0;
      double risk_finish_time = 0.0;
      for (const auto &entry : depth_nodes) {
        struct ConcurrentNodePlan {
          DAGNode *node = nullptr;
          TaskCandidate candidate;
        };
        std::vector<ConcurrentNodePlan> level_plans;
        level_plans.reserve(entry.second.size());
        double level_ready = finish_time;
        double level_risk_ready = risk_finish_time;
        double occupancy_weighted_cost = 0.0;
        double occupancy_weighted_risk_cost = 0.0;
        double span_cost = 0.0;
        double span_risk_cost = 0.0;

        for (DAGNode *node : entry.second) {
          TaskCandidate candidate =
              makeConcurrentSingleCandidate(node, rank, proc);
          if (!std::isfinite(candidate.finish_time)) {
            valid = false;
            break;
          }

          level_ready = std::max(level_ready, candidate.start_time);
          level_risk_ready = std::max(
              level_risk_ready,
              candidate.start_time +
                  riskConfidenceMultiplier() * candidate.transfer_uncertainty);

          const double occupancy_demand =
              coLocatedOccupancyDemand(node, target_items);
          const double exec_risk = riskAdjustedCost(candidate.exec_estimate);
          occupancy_weighted_cost +=
              candidate.exec_estimate.mean * occupancy_demand;
          occupancy_weighted_risk_cost += exec_risk * occupancy_demand;
          span_cost = std::max(span_cost, candidate.exec_estimate.mean);
          span_risk_cost = std::max(span_risk_cost, exec_risk);
          level_plans.push_back({node, std::move(candidate)});
        }
        if (!valid) {
          break;
        }

        // Below the residency target, additional independent kernels mainly
        // hide instruction/memory latency, so the layer cannot be faster than
        // its longest kernel.  Once aggregate demand exceeds one GPU, the
        // occupancy-weighted work term creates the required extra waves.
        const double level_cost =
            std::max(span_cost, occupancy_weighted_cost);
        const double level_risk_cost =
            std::max(span_risk_cost, occupancy_weighted_risk_cost);
        finish_time = level_ready + level_cost;
        risk_finish_time = level_risk_ready + level_risk_cost;

        for (ConcurrentNodePlan &plan : level_plans) {
          DAGNode *node = plan.node;
          const TaskCandidate &candidate = plan.candidate;
          node->exec_rank = candidate.rank;
          node->exec_proc = candidate.proc;
          node->num_parts = candidate.num_parts;
          node->persistent_split = candidate.persistent_split;
          node->split_devices = candidate.occupied_procs;
          // A depth barrier is conservative for irregular DAGs and ensures a
          // successor never assumes that its co-resident producer completed
          // before the layer's shared occupancy budget became available.
          node->finish_time = finish_time;
          commitCandidateReservations(candidate);
        }
      }

      if (!valid) {
        continue;
      }

      gpu_available_time[rank][proc] =
          std::max(gpu_available_time[rank][proc], finish_time);
      if (risk_finish_time < best_risk_finish_time ||
          (risk_finish_time == best_risk_finish_time &&
           finish_time < best_finish_time)) {
        best_finish_time = finish_time;
        best_risk_finish_time = risk_finish_time;
        best_states = saveNodePlacementStates(nodes);
        best_available_time = gpu_available_time;
      }
    }
  }

  restoreNodePlacementStates(nodes, initial_node_states);
  gpu_available_time = initial_available_time;

  const bool mean_improves =
      std::isfinite(best_finish_time) && best_finish_time < heft_finish_time;
  const bool risk_improves = std::isfinite(best_risk_finish_time) &&
                             best_risk_finish_time < heft_risk_finish_time;
  if (!mean_improves || !risk_improves) {
    DAEMON_TRACE_STREAM
        << "algorithmHEFT: whole-batch co-location rejected"
        << " mean_improves " << (mean_improves ? 1 : 0)
        << " risk_improves " << (risk_improves ? 1 : 0)
        << " candidate_finish_time " << best_finish_time
        << " candidate_risk_finish_time " << best_risk_finish_time
        << " heft_finish_time " << heft_finish_time
        << " heft_risk_finish_time " << heft_risk_finish_time << std::endl;
    restoreNodePlacementStates(nodes, heft_states);
    gpu_available_time = heft_available_time;
    return false;
  }

  restoreNodePlacementStates(nodes, best_states);
  gpu_available_time = best_available_time;
  rebuildKernelSchedInfos(topo_order, kernel_sched_order_infos);

  DAEMON_TRACE_STREAM << "algorithmHEFT: co-located GPU batch schedule selected"
            << " finish_time " << best_finish_time
            << " risk_finish_time " << best_risk_finish_time
            << " previous_heft_finish_time " << heft_finish_time
            << " previous_heft_risk_finish_time " << heft_risk_finish_time
            << " concurrent_target_items " << target_items
            << std::endl;
  for (DAGNode *node : topo_order) {
    DAEMON_TRACE_STREAM << "algorithmHEFT: Kernel " << node->kernel_count
              << " co-located to Rank " << node->exec_rank
              << " Proc " << node->exec_proc
              << " NumParts " << node->num_parts
              << " finish_time " << node->finish_time << std::endl;
  }
  return true;
}

static TaskCandidate makeSplitCandidate(DAGNode *node, int rank,
                                        int num_parts) {
  TaskCandidate candidate;
  candidate.rank = rank;
  candidate.num_parts = num_parts;

  if (!worthConsideringSplit(node, num_parts)) {
    return candidate;
  }
  const bool persistent_split =
      gpu_available_time.size() == 1 &&
      supportsPersistentSplit(node, num_parts) &&
      hasPartitionLocalSuccessor(node, num_parts);
  if (rank < 0 || rank >= static_cast<int>(gpu_available_time.size()) ||
      static_cast<int>(gpu_available_time[rank].size()) <= num_parts) {
    return candidate;
  }

  std::vector<int> gpu_procs;
  for (int proc = 1; proc < static_cast<int>(gpu_available_time[rank].size());
       ++proc) {
    if (!monitorMemoryFits(node, rank, proc, num_parts)) {
      continue;
    }
    gpu_procs.push_back(proc);
  }
  if (static_cast<int>(gpu_procs.size()) < num_parts ||
      gpu_procs.size() >= 63) {
    return candidate;
  }

#if defined(SNMD_OFFLINE_WIDE_DAG_GUARD) ||                                \
    defined(SNMD_OFFLINE_COLD_SPLIT_PROBE)
  const bool has_split_profile =
      hasProfileCostForParts(profileKeyForNode(node), num_parts,
                             persistent_split);
  std::map<int, CostEstimate> single_estimates;
  for (int proc : gpu_procs) {
    single_estimates.emplace(proc, estimateSingleExecCost(node, rank, proc));
  }
#endif
#ifdef SNMD_OFFLINE_COLD_SPLIT_PROBE
  if (!has_split_profile &&
      std::all_of(single_estimates.begin(), single_estimates.end(),
                  [](const auto &Entry) {
                    return std::isfinite(Entry.second.mean) &&
                           Entry.second.mean <
                               coldSplitMinSingleCost();
                  })) {
    return candidate;
  }
#endif

  const uint64_t mask_limit = 1ULL << gpu_procs.size();
  for (uint64_t mask = 0; mask < mask_limit; ++mask) {
    if (countMaskBits(mask) != num_parts) {
      continue;
    }

    std::vector<int> split_devices;
    split_devices.reserve(num_parts);
    double device_ready = 0.0;
    for (size_t i = 0; i < gpu_procs.size(); ++i) {
      if ((mask & (1ULL << i)) == 0) {
        continue;
      }
      const int proc = gpu_procs[i];
      split_devices.push_back(proc);
      device_ready = std::max(device_ready, gpu_available_time[rank][proc]);
    }

#if defined(SNMD_OFFLINE_WIDE_DAG_GUARD) ||                                \
    defined(SNMD_OFFLINE_COLD_SPLIT_PROBE)
    CostEstimate best_single_estimate;
    for (int proc : split_devices) {
      const CostEstimate &single_estimate = single_estimates.at(proc);
      if (riskAdjustedCost(single_estimate) <
          riskAdjustedCost(best_single_estimate)) {
        best_single_estimate = single_estimate;
      }
    }
#endif
#ifdef SNMD_OFFLINE_COLD_SPLIT_PROBE
    // The minimum single-task cost is a hard gate. Apply it before building a
    // transfer plan or estimating Split execution: completion-driven
    // scheduling revisits every ready task, so doing the expensive work first
    // creates dispatch bubbles even though the candidate cannot be selected.
    if (!has_split_profile && std::isfinite(best_single_estimate.mean) &&
        best_single_estimate.mean <
            coldSplitMinSingleCost()) {
      continue;
    }
#endif

    const DependencyTransferPlan transfer_plan =
        buildDependencyTransferPlan(node, rank, split_devices);
    const double start_time =
        std::max(device_ready, transfer_plan.ready_time);
    const CostEstimate exec_estimate =
        estimateSplitExecCost(node, rank, split_devices, persistent_split);

#ifdef SNMD_OFFLINE_COLD_SPLIT_PROBE
    if (!has_split_profile && std::isfinite(best_single_estimate.mean)) {
      const double max_cold_split_cost =
          best_single_estimate.mean *
          (100.0 - SNMD_OFFLINE_COLD_SPLIT_MIN_GAIN_PERCENT) / 100.0;
      if (exec_estimate.mean > max_cold_split_cost) {
        DAEMON_TRACE_STREAM
            << "algorithmHEFT: Kernel " << node->kernel_count
            << " cold split probe rejected: split " << exec_estimate.mean
            << " single " << best_single_estimate.mean << " min_single "
            << coldSplitMinSingleCost() << std::endl;
        continue;
      }
    }
#endif

#ifdef SNMD_OFFLINE_WIDE_DAG_GUARD
    const int usable_gpu_count = static_cast<int>(gpu_procs.size());
    if (usable_gpu_count > 1 &&
        node->batch_parallel_width >= usable_gpu_count) {
      bool measured_throughput_win = false;
      if (has_split_profile && std::isfinite(best_single_estimate.mean)) {
        const double max_throughput_split_risk =
            riskAdjustedCost(best_single_estimate) *
            (100.0 - SNMD_OFFLINE_SPLIT_THROUGHPUT_MARGIN_PERCENT) /
            (100.0 * static_cast<double>(num_parts));
        measured_throughput_win =
            riskAdjustedCost(exec_estimate) <= max_throughput_split_risk;
      }
      if (!measured_throughput_win) {
        DAEMON_TRACE_STREAM
            << "algorithmHEFT: Kernel " << node->kernel_count
            << " split rejected by wide-DAG guard: level_width "
            << node->batch_parallel_width << " usable_gpus "
            << usable_gpu_count << std::endl;
        continue;
      }
    }
#endif

    TaskCandidate split_candidate;
    split_candidate.rank = rank;
    split_candidate.proc = split_devices.front();
    split_candidate.num_parts = num_parts;
    split_candidate.persistent_split = persistent_split;
    split_candidate.occupied_procs = split_devices;
    split_candidate.start_time = start_time;
    split_candidate.exec_estimate = exec_estimate;
    split_candidate.transfer_uncertainty = transfer_plan.uncertainty;
    split_candidate.movement_bytes = transfer_plan.movement_bytes;
    split_candidate.transfer_reservations = transfer_plan.reservations;
    split_candidate.finish_time = start_time + exec_estimate.mean;
    if (preferTaskCandidate(split_candidate, candidate)) {
      candidate = std::move(split_candidate);
    }
  }
  return candidate;
}

// Single, Split, monitor scale, profile uncertainty, data movement, and
// device calendars all meet at this one admission point. Both batch HEFT and
// the completion-driven dispatcher use it, preventing a second queue-local
// policy from silently overriding the global scheduler.
static TaskCandidate selectUnifiedTaskCandidate(DAGNode *node) {
  TaskCandidate best_candidate;
  TaskCandidate best_mean_candidate;

  for (int rank = 0; rank < static_cast<int>(gpu_available_time.size());
       ++rank) {
    for (int proc = 0;
         proc < static_cast<int>(gpu_available_time[rank].size()); ++proc) {
      TaskCandidate candidate = makeSingleCandidate(node, rank, proc);
      if (preferTaskCandidate(candidate, best_candidate)) {
        best_candidate = candidate;
      }
      if (candidate.finish_time < best_mean_candidate.finish_time) {
        best_mean_candidate = candidate;
      }
    }

    const int max_split_parts =
        std::min<int>(4,
                      static_cast<int>(gpu_available_time[rank].size()) - 1);
    for (int num_parts = 2; num_parts <= max_split_parts; ++num_parts) {
      if (num_parts % 2 != 0) {
        continue;
      }
      TaskCandidate candidate = makeSplitCandidate(node, rank, num_parts);
      if (preferTaskCandidate(candidate, best_candidate)) {
        best_candidate = candidate;
      }
      if (candidate.finish_time < best_mean_candidate.finish_time) {
        best_mean_candidate = candidate;
      }
    }
  }

#ifdef SNMD_OFFLINE_UNIFIED_RISK_OBJECTIVE
  if (std::isfinite(best_candidate.finish_time) &&
      std::isfinite(best_mean_candidate.finish_time) &&
      (best_candidate.rank != best_mean_candidate.rank ||
       best_candidate.proc != best_mean_candidate.proc ||
       best_candidate.num_parts != best_mean_candidate.num_parts)) {
    DAEMON_TRACE_STREAM
        << "selectUnifiedTaskCandidate: Kernel " << node->kernel_count
        << " risk objective selected Rank " << best_candidate.rank
        << " Proc " << best_candidate.proc << " NumParts "
        << best_candidate.num_parts << " finish " << best_candidate.finish_time
        << " uncertainty " << candidateTotalUncertainty(best_candidate)
        << " risk " << candidateRiskScore(best_candidate)
        << " over mean-best Rank " << best_mean_candidate.rank << " Proc "
        << best_mean_candidate.proc << " NumParts "
        << best_mean_candidate.num_parts << " finish "
        << best_mean_candidate.finish_time << " uncertainty "
        << candidateTotalUncertainty(best_mean_candidate) << " risk "
        << candidateRiskScore(best_mean_candidate) << std::endl;
  }
#endif

  if (best_candidate.rank < 0 || best_candidate.proc < 0) {
    for (int rank = 0;
         rank < static_cast<int>(gpu_available_time.size()) &&
         (best_candidate.rank < 0 || best_candidate.proc < 0);
         ++rank) {
      for (int proc = 0;
           proc < static_cast<int>(gpu_available_time[rank].size()); ++proc) {
        if (!isKernelPlacementProc(rank, proc)) {
          continue;
        }
        best_candidate = makeSingleCandidateNoMemoryFilter(node, rank, proc);
        break;
      }
    }
  }

  if (best_candidate.rank < 0 || best_candidate.proc < 0) {
    best_candidate = makeSingleCandidateNoMemoryFilter(node, 0, 0);
  }
  return best_candidate;
}

static double estimateAverageRankCost(DAGNode *node) {
  double sum = 0.0;
  int count = 0;
  for (int rank = 0; rank < static_cast<int>(gpu_available_time.size()); ++rank) {
    for (int proc = 0; proc < static_cast<int>(gpu_available_time[rank].size());
         ++proc) {
      if (isKernelPlacementProc(rank, proc) &&
          monitorMemoryFits(node, rank, proc, 1)) {
        sum += riskAdjustedCost(estimateSingleExecCost(node, rank, proc));
        count++;
      }
    }
  }
  if (count == 0) {
    return std::max(1.0, node->total_elem);
  }
  return sum / count;
}

static double estimateAverageCommCost(DAGNode *node, DAGNode *pre_node) {
  const double comm_bytes = getCommBytes(node, pre_node);
  if (comm_bytes <= 0.0) {
    return 0.0;
  }

  double total_cost = 0.0;
  int count = 0;
  for (int src_rank = 0;
       src_rank < static_cast<int>(gpu_available_time.size()); ++src_rank) {
    for (int src_proc = 0;
         src_proc < static_cast<int>(gpu_available_time[src_rank].size());
         ++src_proc) {
      if (!isKernelPlacementProc(src_rank, src_proc)) {
        continue;
      }
      for (int dst_rank = 0;
           dst_rank < static_cast<int>(gpu_available_time.size());
           ++dst_rank) {
        for (int dst_proc = 0;
             dst_proc < static_cast<int>(gpu_available_time[dst_rank].size());
             ++dst_proc) {
          if (!isKernelPlacementProc(dst_rank, dst_proc)) {
            continue;
          }
          double seconds = 0.0;
          if (src_rank == dst_rank) {
            seconds =
                sameRankCopySeconds(src_rank, src_proc, dst_proc, comm_bytes);
          } else {
            seconds = secondsForBytesAtBandwidth(
                comm_bytes, crossRankBandwidthGiB(src_rank, dst_rank));
          }
          CostEstimate transfer_estimate;
          transfer_estimate.mean = heftCostFromSeconds(seconds);
          transfer_estimate.uncertainty =
              transfer_estimate.mean *
              (static_cast<double>(SNMD_OFFLINE_TRANSFER_ERROR_PERCENT) /
               100.0);
          total_cost += riskAdjustedCost(transfer_estimate);
          count++;
        }
      }
    }
  }

  return count == 0 ? 0.0 : total_cost / static_cast<double>(count);
}

// nodes: 这批要调度的所有kernel
bool algorithmHEFT(
    std::vector<DAGNode *> &nodes,
    std::vector<D2DKernelSchedInfo> &kernel_sched_order_infos) {
  // Build one scheduling context for this wait-delimited batch. A confirmed
  // completion frontier resets runtime-owned calendars; monitor load appears
  // once as a service-time scale. Device numbering remains aligned with
  // handler's globalDevices (proc 0 is the CPU).
  const pid_t program_pid =
      nodes.empty() || nodes.front() == nullptr ? 0 : nodes.front()->program_pid;
  ensureOfflineDeviceModel(program_pid);

  int batch_root_count = 0;
  for (DAGNode *node : nodes) {
    if (node->depend_on.empty()) {
      ++batch_root_count;
    }
  }
  for (DAGNode *node : nodes) {
    node->batch_root_count = batch_root_count;
  }

#if defined(SNMD_OFFLINE_WIDE_DAG_GUARD) ||                                \
    defined(SNMD_OFFLINE_SPLIT_STATS)
  // Compute depth inside this wait batch only. Cross-window predecessors still
  // contribute placement/communication cost, but must not make otherwise
  // parallel roots appear at unrelated absolute depths.
  const std::unordered_set<DAGNode *> current_batch_nodes(nodes.begin(),
                                                          nodes.end());
  std::vector<DAGNode *> batch_topo = reverseTopologicalOrder(nodes);
  std::reverse(batch_topo.begin(), batch_topo.end());
  std::map<DAGNode *, int> batch_depths;
  std::map<int, int> batch_depth_widths;
  for (DAGNode *node : batch_topo) {
    int batch_depth = 0;
    for (DAGNode *pre_node : node->depend_on) {
      if (!current_batch_nodes.count(pre_node)) {
        continue;
      }
      batch_depth = std::max(batch_depth, batch_depths[pre_node] + 1);
    }
    batch_depths[node] = batch_depth;
    batch_depth_widths[batch_depth]++;
  }
  for (DAGNode *node : nodes) {
    node->batch_parallel_width = batch_depth_widths[batch_depths[node]];
  }
#endif

  // 1.根据(规模+monitor)生成执行时间表 对于每个任务t_i计算w(i) 以及任务在不同proc上时各个pre的传输代价
  for (DAGNode *node : nodes) {
    // 1.1. 计算每个任务的平均计算时间
    double total_elem = 0;
    for (const SyclReqData &req : node->req_data) {
      total_elem += reqAccessElems(req);
    }
    node->total_elem = total_elem / 1000; // TODO 归一化
    registerKernelFeature(node);
    DAEMON_TRACE_STREAM << "algorithmHEFT: Kernel " << node->kernel_count
              << " identity " << std::hex << node->kernel_identity << std::dec
              << " total_elem: " << total_elem
              << " precision: "
              << precisionName(inferKernelPrecisionFromReqs(node->req_data))
              << std::endl;
    const double cold_intensity = coldArithmeticIntensityFactor(node);
    if (cold_intensity > 1.0) {
      DAEMON_TRACE_STREAM << "algorithmHEFT: Kernel " << node->kernel_count
                << " cold_arithmetic_intensity_factor: "
                << cold_intensity << std::endl;
    }

    // 1.2. 计算每个任务需要从各个前序接收多少数据
    for (std::pair<DAGNode *, std::set<SyclReqData>> dep_pair : node->depend_on_mem) {
      DAGNode *pre_node = dep_pair.first;
      int pre_comm_elem = 0;
      for (const SyclReqData &req : dep_pair.second) {
        pre_comm_elem += req.buff_size;
      }
      node->comm_elem[pre_node] = pre_comm_elem / 1000;
      DAEMON_TRACE_STREAM << "algorithmHEFT: Kernel " << node->kernel_count << " depends on Kernel " << pre_node->kernel_count << " pre_comm_elem: " << pre_comm_elem << std::endl;
    }
  }

  // 2.从后向前 对于每个任务u->v 计算rank(u)=w_mean(u)+max_v[c(u,v)+rank(v)] 并从大到小排序
  // 必须使用逆拓扑序，保证计算一个节点时同批次所有后继已经计算过rank_u。
  std::unordered_set<DAGNode *> current_nodes(nodes.begin(), nodes.end());
  std::vector<DAGNode *> visited = reverseTopologicalOrder(nodes);
  const std::vector<DAGNode *> topo_order(visited.rbegin(), visited.rend());
  for (DAGNode *node : visited) {
    double max_succ = 0;
    for (DAGNode *succ_node : node->depend_by) {
      if (!current_nodes.count(succ_node)) {
        continue;
      }
      const double succ_cost =
          estimateAverageCommCost(succ_node, node) + succ_node->rank_u;
      if (succ_cost > max_succ) {
        max_succ = succ_cost;
      }
    }
    node->rank_u = estimateAverageRankCost(node) + max_succ;
    if (max_succ == 0) {
      DAEMON_TRACE_STREAM << "algorithmHEFT: Kernel " << node->kernel_count
                << " is exit node rank_u: " << node->rank_u << std::endl;
    }
  }
  // 以rank_u从大到小排序
  std::sort(visited.begin(), visited.end(), [](DAGNode *a, DAGNode *b) {
    return a->rank_u > b->rank_u;
  });

  const std::vector<NodePlacementState> initial_node_states =
      saveNodePlacementStates(nodes);
  const std::vector<std::vector<double>> initial_available_time =
      gpu_available_time;

  // 3.每个任务计算 对于每个proc 计算start_v(p)=max_[last_finish(p),finish(u_1)+comm(u_1,v),...]
  // 和finish_v(p)=start_v(p)+w(v,p)
  // 同时把SNMD split作为候选放置方式，选择统一风险目标最小者。
  double heft_risk_finish_time = 0.0;
  for (int order = 0; order < visited.size(); order++) {
    DAGNode *node = visited[order];
    TaskCandidate best_candidate = selectUnifiedTaskCandidate(node);

    node->exec_rank = best_candidate.rank;
    node->exec_proc = best_candidate.proc;
    node->num_parts = best_candidate.num_parts;
    node->persistent_split = best_candidate.persistent_split;
    node->split_devices = best_candidate.occupied_procs;
    node->finish_time = best_candidate.finish_time;
    heft_risk_finish_time =
        std::max(heft_risk_finish_time, candidateRiskScore(best_candidate));
    commitCandidateReservations(best_candidate);
    for (int proc : best_candidate.occupied_procs) {
      gpu_available_time[node->exec_rank][proc] = node->finish_time;
    }

    D2DKernelSchedInfo kernel_sched_info;
    kernel_sched_info.kernel_count = node->kernel_count;
    kernel_sched_info.exec_order = order + 1;
    kernel_sched_info.exec_rank = node->exec_rank;
    kernel_sched_info.exec_device = node->exec_proc;
    kernel_sched_info.num_parts = node->num_parts;
    kernel_sched_info.persistent_split = node->persistent_split;
    kernel_sched_info.split_devices = node->split_devices;
    kernel_sched_order_infos.push_back(kernel_sched_info);

    DAEMON_TRACE_STREAM << "algorithmHEFT: Kernel " << node->kernel_count
              << " assigned to Rank " << node->exec_rank
              << " Proc " << node->exec_proc
              << " NumParts " << node->num_parts
              << " PersistentSplit " << node->persistent_split
              << " SplitDevices";
    for (int split_device : node->split_devices) {
      DAEMON_TRACE_STREAM << " " << split_device;
    }
    DAEMON_TRACE_STREAM
              << " start_time " << best_candidate.start_time
              << " exec_cost " << best_candidate.exec_estimate.mean
              << " exec_uncertainty "
              << best_candidate.exec_estimate.uncertainty
              << " transfer_uncertainty "
              << best_candidate.transfer_uncertainty
              << " cost_source "
              << costEstimateSourceName(best_candidate.exec_estimate.source)
              << " profile_samples " << best_candidate.exec_estimate.samples
              << " risk_score " << candidateRiskScore(best_candidate)
              << " movement_bytes " << best_candidate.movement_bytes
              << " transfer_reservations "
              << best_candidate.transfer_reservations.size()
              << " finish_time " << node->finish_time << std::endl;
  }

  const double heft_finish_time = batchFinishTime(nodes);
  size_t independent_component_count = 0;
  size_t independent_component_device_count = 0;
  const bool heft_parallelizes_independent_components =
      heftAlreadyParallelizesIndependentComponents(
          topo_order, current_nodes, independent_component_count,
          independent_component_device_count);
  const bool co_located_batch = applyCoLocatedGpuScheduleIfBetter(
      nodes, initial_node_states, initial_available_time, heft_finish_time,
      heft_risk_finish_time, topo_order, current_nodes,
      heft_parallelizes_independent_components, independent_component_count,
      independent_component_device_count, kernel_sched_order_infos);
  bool use_static_batch = co_located_batch;
  bool component_affine_static_batch = false;
  bool single_gpu_static_batch = false;
  const size_t component_static_min_nodes =
      componentAffineStaticMinNodesPerComponent();
  const bool component_graph_is_deep =
      component_static_min_nodes == 0 ||
      independent_component_count <= nodes.size() / component_static_min_nodes;
  if (!use_static_batch && componentAffineStaticFastPathEnabled() &&
      heft_parallelizes_independent_components && component_graph_is_deep) {
    // HEFT has already produced a complete, migration-free placement for
    // every independent component. Preserve that placement and submit the
    // full DAG once: per-kernel completion feedback cannot improve affinity
    // here, but it creates a host round trip between dependent microkernels.
    use_static_batch = true;
    component_affine_static_batch = true;
    DAEMON_TRACE_STREAM
        << "algorithmHEFT: multi-GPU component-affine batch uses static "
           "execution for "
        << independent_component_count << " independent components across "
        << independent_component_device_count << " distinct GPUs"
        << std::endl;
  } else if (!use_static_batch && componentAffineStaticFastPathEnabled() &&
             heft_parallelizes_independent_components &&
             !component_graph_is_deep) {
    DAEMON_TRACE_STREAM
        << "algorithmHEFT: component-affine static execution skipped; "
           "average component size "
        << nodes.size() / independent_component_count
        << " is below threshold " << component_static_min_nodes << std::endl;
  }
  if (!use_static_batch && concurrentGpuTargetItems() > 0.0) {
    size_t independent_component_count = 0;
    if (independentComponentsShareOneGpu(nodes,
                                         independent_component_count)) {
      // There is no placement trade-off left on one GPU. Completion-driven
      // admission would serialize the independent components before queue
      // ordering can help, so submit the HEFT placement as one static OOO
      // batch. The handler independently verifies one device and num_parts=1
      // before selecting its concurrent queue.
      use_static_batch = true;
      single_gpu_static_batch = true;
      DAEMON_TRACE_STREAM
          << "algorithmHEFT: single-GPU non-split batch uses static "
             "out-of-order execution for "
          << independent_component_count << " independent components"
          << std::endl;
    }
  }

#ifdef SNMD_OFFLINE_SPLIT_STATS
  uint64_t selected_single_kernels = 0;
  uint64_t selected_split_kernels = 0;
  long double estimated_split_extra_input_bytes = 0.0;
  long double estimated_split_merge_bytes = 0.0;
  for (DAGNode *node : nodes) {
    const CostEstimate single_exec_estimate =
        estimateSingleExecCost(node, node->exec_rank, node->exec_proc);
    CostEstimate selected_exec_estimate = single_exec_estimate;
    const bool has_single_profile =
        hasProfileCostForParts(profileKeyForNode(node), 1);
    bool has_selected_profile = has_single_profile;
    if (node->num_parts > 1) {
      selected_split_kernels++;
      selected_exec_estimate =
          estimateSplitExecCost(node, node->exec_rank, node->split_devices,
                                node->persistent_split);
      has_selected_profile = hasProfileCostForParts(
          profileKeyForNode(node), node->num_parts,
          node->persistent_split);
      const long double replicated_read_bytes =
          totalReadBytesForPartitionMode(node, false);
      const long double partition_local_read_bytes =
          totalReadBytesForPartitionMode(node, true);
      estimated_split_extra_input_bytes +=
          replicated_read_bytes *
              static_cast<long double>(node->num_parts - 1) +
          partition_local_read_bytes *
              static_cast<long double>(node->num_parts - 1) /
              static_cast<long double>(node->num_parts);
      if (!node->persistent_split) {
        estimated_split_merge_bytes +=
            static_cast<long double>(totalWriteBytes(node)) *
            static_cast<long double>(node->num_parts - 1) /
            static_cast<long double>(node->num_parts);
      }
    } else {
      selected_single_kernels++;
    }

    std::cout << "SNMD_SCHED_DECISION kernel=" << node->kernel_count
              << " depth_width=" << node->batch_parallel_width
              << " rank=" << node->exec_rank
              << " device=" << node->exec_proc
              << " parts=" << node->num_parts
              << " persistent=" << (node->persistent_split ? 1 : 0)
              << " single_exec_cost=" << single_exec_estimate.mean
              << " single_exec_uncertainty="
              << single_exec_estimate.uncertainty
              << " selected_exec_cost=" << selected_exec_estimate.mean
              << " selected_exec_uncertainty="
              << selected_exec_estimate.uncertainty
              << " single_profile=" << (has_single_profile ? 1 : 0)
              << " selected_profile=" << (has_selected_profile ? 1 : 0)
              << std::endl;
  }
  std::cout << "SNMD_SCHED_STATS kernels=" << nodes.size()
            << " single_kernels=" << selected_single_kernels
            << " split_kernels=" << selected_split_kernels
            << " estimated_split_extra_input_bytes="
            << static_cast<double>(estimated_split_extra_input_bytes)
            << " estimated_split_merge_bytes="
            << static_cast<double>(estimated_split_merge_bytes) << std::endl;
#endif

  if (decisionSummaryRuntimeEnabled()) {
    uint64_t selected_single_kernels = 0;
    uint64_t selected_split_kernels = 0;
    std::array<uint64_t, 6> source_counts{};
    uint64_t profile_sampled_kernels = 0;
    uint64_t profile_samples_total = 0;
    for (DAGNode *node : nodes) {
      CostEstimate estimate;
      if (node->num_parts > 1) {
        ++selected_split_kernels;
        estimate = estimateSplitExecCost(
            node, node->exec_rank, node->split_devices,
            node->persistent_split);
      } else {
        ++selected_single_kernels;
        estimate =
            estimateSingleExecCost(node, node->exec_rank, node->exec_proc);
      }
      const size_t source_index = static_cast<size_t>(estimate.source);
      if (source_index < source_counts.size()) {
        ++source_counts[source_index];
      }
      if (estimate.samples > 0) {
        ++profile_sampled_kernels;
        profile_samples_total += static_cast<uint64_t>(estimate.samples);
      }
    }
    const char *path =
        co_located_batch
            ? "co_located_static"
            : (component_affine_static_batch
                   ? "component_affine_static"
                   : (single_gpu_static_batch ? "single_gpu_ooo_static"
                                              : (completionQueueRuntimeRequested()
                                                     ? "completion_candidate"
                                                     : "heft_static_fallback")));
    std::cout
        << "SNMD_WINDOW_DECISION pid=" << program_pid
        << " kernels=" << nodes.size()
        << " roots=" << batch_root_count
        << " components=" << independent_component_count
        << " component_devices=" << independent_component_device_count
        << " path=" << path
        << " completion_queue="
        << (completionQueueRuntimeRequested() ? 1 : 0)
        << " single_kernels=" << selected_single_kernels
        << " split_kernels=" << selected_split_kernels
        << " source_cold="
        << source_counts[static_cast<size_t>(CostEstimateSource::cold_model)]
        << " source_exact="
        << source_counts[
               static_cast<size_t>(CostEstimateSource::exact_profile)]
        << " source_persisted="
        << source_counts[
               static_cast<size_t>(CostEstimateSource::persisted_profile)]
        << " source_scaled="
        << source_counts[
               static_cast<size_t>(CostEstimateSource::scaled_profile)]
        << " source_learned="
        << source_counts[
               static_cast<size_t>(CostEstimateSource::learned_profile)]
        << " source_derived_split="
        << source_counts[
               static_cast<size_t>(CostEstimateSource::derived_split)]
        << " profile_sampled_kernels=" << profile_sampled_kernels
        << " profile_samples_total=" << profile_samples_total
#ifdef SNMD_OFFLINE_COLD_SPLIT_PROBE
        << " cold_split_min_single_cost=" << coldSplitMinSingleCost()
#endif
        << std::endl;
  }

  regenerateReqRanksAfterHEFT(nodes, kernel_sched_order_infos);

  // TODO 最合适用几个节点去跑
  // 通信代价和贪心避免了扩张代价大于运行代价

  return use_static_batch;
}

#ifdef SNMD_OFFLINE_COMPLETION_DRIVEN_QUEUE
enum class CompletionNodePhase { Pending, Dispatched, Complete };

struct CompletionNodeRuntimeState {
  CompletionNodePhase phase = CompletionNodePhase::Pending;
  std::vector<std::pair<int, int>> reserved_devices;
  // Used to compare immediate execution on an idle slow device with waiting
  // for a faster in-flight device. Real completion acknowledgements remain
  // the only mechanism that releases the reservation.
  uint64_t dispatch_started_ns = 0;
  uint64_t predicted_release_ns = 0;
};

enum class CompletionWindowResult {
  Completed,
  ExitRequested,
  FailedBeforeStart,
  FailedAfterStart
};

static bool completionDrivenQueueRuntimeEnabled() {
  return completionQueueRuntimeRequested();
}

static void resetCompletionWindowCalendar(const std::vector<DAGNode *> &nodes) {
  for (std::vector<double> &rank_calendar : gpu_available_time) {
    std::fill(rank_calendar.begin(), rank_calendar.end(), 0.0);
  }
  for (DAGNode *node : nodes) {
    node->exec_rank = -1;
    node->exec_proc = -1;
    node->num_parts = 1;
    node->persistent_split = false;
    node->split_devices.clear();
    node->finish_time = 0.0;
  }
}

static std::vector<std::pair<int, int>>
completionCandidateReservations(const TaskCandidate &candidate) {
  std::set<std::pair<int, int>> unique_devices;
  for (int proc : candidate.occupied_procs) {
    unique_devices.insert({candidate.rank, proc});
  }
  return std::vector<std::pair<int, int>>(unique_devices.begin(),
                                          unique_devices.end());
}

static void clearCompletionEphemeralTransferCalendar() {
  for (std::vector<double> &rank_calendar : gpu_available_time) {
    for (double &ready_time : rank_calendar) {
      // Completion compute predictions are reconstructed from runtime state
      // before every admission pass. Everything left here after a dispatch
      // wave is ephemeral transfer-planning state.
      ready_time = 0.0;
    }
  }
}

static double completionRemainingCost(
    const CompletionNodeRuntimeState &state, uint64_t now_ns) {
  if (state.predicted_release_ns > now_ns) {
    return static_cast<double>(state.predicted_release_ns - now_ns) /
           PROFILE_NS_TO_COST;
  }

  // A task that outlives its upper-confidence prediction is evidence that the
  // old prediction was too optimistic, not that the device will become free
  // "immediately". Use elapsed service time as a conservative residual until
  // the real completion updates the profile.
  if (state.dispatch_started_ns < now_ns) {
    return std::max(
        0.001, static_cast<double>(now_ns - state.dispatch_started_ns) /
                   PROFILE_NS_TO_COST);
  }
  return 0.001;
}

static uint64_t completionPredictedReleaseNs(
    const TaskCandidate &candidate) {
  const double release_cost = candidateRiskScore(candidate);
  if (!std::isfinite(release_cost) || release_cost <= 0.0) {
    return daemonSteadyNowNs();
  }

  const long double release_ns =
      static_cast<long double>(release_cost) * PROFILE_NS_TO_COST;
  const uint64_t now_ns = daemonSteadyNowNs();
  if (release_ns >=
      static_cast<long double>(std::numeric_limits<uint64_t>::max() -
                               now_ns)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return now_ns + static_cast<uint64_t>(std::ceil(release_ns));
}

static void setCompletionDeviceCalendar(
    const std::vector<std::pair<int, int>> &devices, double ready_time) {
  for (const std::pair<int, int> &device : devices) {
    const int rank = device.first;
    const int proc = device.second;
    if (rank < 0 || rank >= static_cast<int>(gpu_available_time.size()) ||
        proc < 0 ||
        proc >= static_cast<int>(gpu_available_time[rank].size())) {
      continue;
    }
    gpu_available_time[rank][proc] =
        std::max(gpu_available_time[rank][proc], ready_time);
  }
}

using CompletionDeviceSet = std::set<std::pair<int, int>>;

static CompletionDeviceSet refreshCompletionDeviceCalendar(
    const std::unordered_map<DAGNode *, CompletionNodeRuntimeState> &states,
    const std::unordered_set<DAGNode *> &in_flight_nodes) {
  for (std::vector<double> &rank_calendar : gpu_available_time) {
    std::fill(rank_calendar.begin(), rank_calendar.end(), 0.0);
  }

  CompletionDeviceSet in_flight_devices;
  const uint64_t now_ns = daemonSteadyNowNs();
  for (DAGNode *node : in_flight_nodes) {
    const CompletionNodeRuntimeState &state = states.at(node);
    if (state.phase != CompletionNodePhase::Dispatched) {
      continue;
    }
    setCompletionDeviceCalendar(
        state.reserved_devices, completionRemainingCost(state, now_ns));
    for (const std::pair<int, int> &device : state.reserved_devices) {
      in_flight_devices.insert(device);
    }
  }
  return in_flight_devices;
}

static bool completionCandidateTouchesInFlightDevice(
    const TaskCandidate &candidate,
    const CompletionDeviceSet &in_flight_devices) {
  for (int proc : candidate.occupied_procs) {
    if (in_flight_devices.count({candidate.rank, proc}) != 0) {
      return true;
    }
  }
  for (const TaskCandidate::DeviceReservation &reservation :
       candidate.transfer_reservations) {
    if (in_flight_devices.count({reservation.rank, reservation.proc}) != 0) {
      return true;
    }
  }
  return false;
}

static std::vector<D2SKernelExecInfo> dispatchCompletionReadyNodes(
    const std::vector<DAGNode *> &priority_order,
    std::set<size_t> &ready_indices,
    std::unordered_map<DAGNode *, CompletionNodeRuntimeState> &states,
    std::unordered_set<DAGNode *> &in_flight_nodes,
    int &dispatch_order) {
  struct PlannedDispatch {
    D2SKernelExecInfo exec_info;
    double movement_bytes = 0.0;
    size_t compute_reservations = 0;
  };
  std::vector<PlannedDispatch> planned_dispatches;
  CompletionDeviceSet in_flight_devices =
      refreshCompletionDeviceCalendar(states, in_flight_nodes);
  while (true) {
    DAGNode *selected_node = nullptr;
    auto selected_ready_it = ready_indices.end();
    TaskCandidate selected_candidate;
    DAGNode *deferred_node = nullptr;
    TaskCandidate deferred_candidate;

    for (auto ready_it = ready_indices.begin();
         ready_it != ready_indices.end(); ++ready_it) {
      DAGNode *node = priority_order[*ready_it];
      CompletionNodeRuntimeState &state = states.at(node);
      if (state.phase != CompletionNodePhase::Pending) {
        continue;
      }

      TaskCandidate candidate = selectUnifiedTaskCandidate(node);
      if (!std::isfinite(candidate.finish_time)) {
        continue;
      }
      // The completion-driven path is deliberately single-rank until remote
      // completion acknowledgements and failure recovery are implemented.
      if (candidate.rank != 0) {
        continue;
      }
      // Waiting for an in-flight fast device can be cheaper than immediate
      // execution on an idle slow device. Do not enqueue that future choice:
      // it would discard completion feedback and a cross-context migration
      // can synchronously wait behind the source device's compute queue,
      // preventing the handler from reporting unrelated completions. Keep
      // scanning so another READY node can still use an idle device when that
      // is genuinely its best candidate.
      if (completionCandidateTouchesInFlightDevice(
              candidate, in_flight_devices)) {
        if (deferred_node == nullptr) {
          deferred_node = node;
          deferred_candidate = candidate;
        }
        continue;
      }
      selected_node = node;
      selected_ready_it = ready_it;
      selected_candidate = std::move(candidate);
      break;
    }

    if (selected_node == nullptr) {
      if (deferred_node != nullptr) {
        DAEMON_TRACE_STREAM
            << "CompletionQueue: defer kernel "
            << deferred_node->kernel_count
            << " for predicted-better in-flight candidate rank "
            << deferred_candidate.rank << " proc "
            << deferred_candidate.proc << " start_time "
            << deferred_candidate.start_time << " finish_time "
            << deferred_candidate.finish_time << " risk "
            << candidateRiskScore(deferred_candidate) << std::endl;
      }
      break;
    }

    selected_node->exec_rank = selected_candidate.rank;
    selected_node->exec_proc = selected_candidate.proc;
    selected_node->num_parts = selected_candidate.num_parts;
    selected_node->persistent_split = selected_candidate.persistent_split;
    selected_node->split_devices = selected_candidate.occupied_procs;
    selected_node->finish_time = selected_candidate.finish_time;

    CompletionNodeRuntimeState &selected_state = states.at(selected_node);
    selected_state.phase = CompletionNodePhase::Dispatched;
    selected_state.reserved_devices =
        completionCandidateReservations(selected_candidate);
    selected_state.dispatch_started_ns = daemonSteadyNowNs();
    selected_state.predicted_release_ns =
        completionPredictedReleaseNs(selected_candidate);
    // Reserve copy endpoints while selecting the remainder of this dispatch
    // wave, but do not hold a source GPU for the target kernel's full compute
    // duration. That old lifetime turns a millisecond migration into a
    // multi-second false occupancy and recreates the skipped-GPU symptom.
    commitCandidateReservations(selected_candidate);
    setCompletionDeviceCalendar(
        selected_state.reserved_devices,
        completionRemainingCost(selected_state, daemonSteadyNowNs()));
    ready_indices.erase(selected_ready_it);
    in_flight_nodes.insert(selected_node);
    in_flight_devices.insert(selected_state.reserved_devices.begin(),
                             selected_state.reserved_devices.end());

    D2SKernelExecInfo exec_info;
    exec_info.kernel_count = selected_node->kernel_count;
    exec_info.exec = true;
    exec_info.device_index = selected_node->exec_proc;
    exec_info.num_parts = selected_node->num_parts;
    exec_info.persistent_split = selected_node->persistent_split;
    exec_info.split_devices = selected_node->split_devices;
    planned_dispatches.push_back(
        {std::move(exec_info), selected_candidate.movement_bytes,
         selected_state.reserved_devices.size()});
  }
  clearCompletionEphemeralTransferCalendar();

  // Preserve admission order. Transfer reservations are committed while this
  // list is built; moving zero-copy work ahead of an earlier migration can
  // put a long in-order compute on the migration's source device and recreate
  // the handler-wide synchronous wait that endpoint-safe admission prevents.
  // Independent zero-copy work is still admitted before a blocked migration
  // because the READY scan skips candidates that touch in-flight endpoints.

  std::vector<D2SKernelExecInfo> dispatches;
  dispatches.reserve(planned_dispatches.size());
  for (PlannedDispatch &dispatch : planned_dispatches) {
    DAEMON_TRACE_STREAM
        << "CompletionQueue: dispatch_order " << ++dispatch_order
        << " kernel " << dispatch.exec_info.kernel_count << " rank 0 proc "
        << dispatch.exec_info.device_index << " parts "
        << dispatch.exec_info.num_parts << " persistent "
        << dispatch.exec_info.persistent_split << " movement_bytes "
        << dispatch.movement_bytes << " reserved_devices "
        << dispatch.compute_reservations << std::endl;
    dispatches.push_back(std::move(dispatch.exec_info));
  }
  return dispatches;
}

static bool sendCompletionDispatchBatch(
    mqd_t mq_id_program, const std::vector<D2SKernelExecInfo> &dispatches,
    bool window_complete, bool window_failed = false) {
  D2SDispatchBatchData batch;
  batch.completion_driven = true;
  batch.window_complete = window_complete;
  batch.window_failed = window_failed;
  batch.kernel_exec_infos = dispatches;
  const std::string payload = batch.serialize();
  sendOfflineMqPayload(mq_id_program, payload, MAX_MSG_PROGRAM_SIZE,
                       "completion dispatch batch");
  return true;
}

static CompletionWindowResult runCompletionDrivenWindow(
    const std::vector<DAGNode *> &nodes, int daemon_wait_count, int local_pid,
    mqd_t mq_id_daemon) {
  if (nodes.empty()) {
    return CompletionWindowResult::FailedBeforeStart;
  }

  char queue_name[MESSAGE_QUEUE_PROGRAM_NAME_MAX];
  std::snprintf(queue_name, sizeof(queue_name), MESSAGE_QUEUE_PROGRAM_PATTERN,
                local_pid);
  mqd_t mq_id_program = mq_open(queue_name, O_WRONLY);
  if (mq_id_program == static_cast<mqd_t>(-1)) {
    perror("completion queue mq_id_program open failed");
    return CompletionWindowResult::FailedBeforeStart;
  }

  std::vector<DAGNode *> priority_order = nodes;
  std::stable_sort(priority_order.begin(), priority_order.end(),
                   [](const DAGNode *lhs, const DAGNode *rhs) {
                     return lhs->rank_u > rhs->rank_u;
                   });
  std::unordered_set<DAGNode *> window_nodes(nodes.begin(), nodes.end());
  std::unordered_map<DAGNode *, CompletionNodeRuntimeState> states;
  std::unordered_map<DAGNode *, size_t> remaining_predecessors;
  std::unordered_map<DAGNode *, size_t> priority_indices;
  std::unordered_map<int, DAGNode *> nodes_by_kernel_count;
  std::unordered_set<DAGNode *> in_flight_nodes;
  states.reserve(nodes.size());
  remaining_predecessors.reserve(nodes.size());
  priority_indices.reserve(nodes.size());
  nodes_by_kernel_count.reserve(nodes.size());
  in_flight_nodes.reserve(nodes.size());
  for (size_t index = 0; index < priority_order.size(); ++index) {
    priority_indices.emplace(priority_order[index], index);
  }
  std::set<size_t> ready_indices;
  for (DAGNode *node : nodes) {
    states.emplace(node, CompletionNodeRuntimeState{});
    if (!nodes_by_kernel_count.emplace(node->kernel_count, node).second) {
      std::cerr << "CompletionQueue: duplicate kernel_count "
                << node->kernel_count << " in wait " << daemon_wait_count
                << std::endl;
      mq_close(mq_id_program);
      return CompletionWindowResult::FailedBeforeStart;
    }
    size_t predecessor_count = 0;
    for (DAGNode *predecessor : node->depend_on) {
      predecessor_count += window_nodes.count(predecessor) != 0 ? 1 : 0;
    }
    remaining_predecessors.emplace(node, predecessor_count);
    if (predecessor_count == 0) {
      ready_indices.insert(priority_indices.at(node));
    }
  }
  // algorithmHEFT has already produced the batch-static fallback before this
  // function is entered. Preserve its DAG placement as well as its serialized
  // exec infos until the first dynamic dispatch commits the new protocol.
  const std::vector<NodePlacementState> static_fallback_states =
      saveNodePlacementStates(nodes);
  const std::vector<std::vector<double>> static_fallback_calendar =
      gpu_available_time;
  resetCompletionWindowCalendar(nodes);

  int dispatch_order = 0;
  bool protocol_started = false;
  size_t complete_count = 0;
  while (true) {
    std::vector<D2SKernelExecInfo> dispatches =
        dispatchCompletionReadyNodes(priority_order, ready_indices, states,
                                     in_flight_nodes, dispatch_order);
    const bool window_complete = complete_count == nodes.size();

    if (!window_complete && dispatches.empty() && in_flight_nodes.empty()) {
      std::cerr << "CompletionQueue: no ready or in-flight kernel in wait "
                << daemon_wait_count << std::endl;
      if (protocol_started) {
        sendCompletionDispatchBatch(mq_id_program, {}, false, true);
      } else {
        restoreNodePlacementStates(nodes, static_fallback_states);
        gpu_available_time = static_fallback_calendar;
      }
      mq_close(mq_id_program);
      return protocol_started ? CompletionWindowResult::FailedAfterStart
                              : CompletionWindowResult::FailedBeforeStart;
    }

    sendCompletionDispatchBatch(mq_id_program, dispatches, window_complete);
    protocol_started = true;
    if (window_complete) {
      DAEMON_TRACE_STREAM << "CompletionQueue: wait " << daemon_wait_count
                          << " complete kernels " << complete_count
                          << std::endl;
      mq_close(mq_id_program);
      return CompletionWindowResult::Completed;
    }

    const std::string payload = receiveOfflineMqPayload(
        mq_id_daemon, MAX_MSG_DAEMON_SIZE, "completion acknowledgement");
    if (payload == "EXIT") {
      mq_close(mq_id_program);
      return CompletionWindowResult::ExitRequested;
    }
    if (!S2DCompletionBatchData::isCompletionBatch(payload)) {
      std::cerr << "CompletionQueue: expected completion batch for wait "
                << daemon_wait_count << std::endl;
      sendCompletionDispatchBatch(mq_id_program, {}, false, true);
      mq_close(mq_id_program);
      return CompletionWindowResult::FailedAfterStart;
    }

    S2DCompletionBatchData completion_batch =
        S2DCompletionBatchData::deserialize(payload);
    if (completion_batch.wait_count != daemon_wait_count) {
      std::cerr << "CompletionQueue: invalid completion batch wait "
                << completion_batch.wait_count << " expected "
                << daemon_wait_count << std::endl;
      sendCompletionDispatchBatch(mq_id_program, {}, false, true);
      mq_close(mq_id_program);
      return CompletionWindowResult::FailedAfterStart;
    }
    if (completion_batch.window_failed) {
      std::cerr << "CompletionQueue: handler reported failure for wait "
                << daemon_wait_count << std::endl;
      mq_close(mq_id_program);
      return CompletionWindowResult::FailedAfterStart;
    }
    if (completion_batch.completions.empty()) {
      std::cerr << "CompletionQueue: invalid completion batch wait "
                << completion_batch.wait_count << " expected "
                << daemon_wait_count << std::endl;
      sendCompletionDispatchBatch(mq_id_program, {}, false, true);
      mq_close(mq_id_program);
      return CompletionWindowResult::FailedAfterStart;
    }

    std::unordered_set<int> completion_kernel_counts;
    std::vector<DAGNode *> completed_nodes;
    completed_nodes.reserve(completion_batch.completions.size());
    for (const S2DKernelProfileData &completion :
         completion_batch.completions) {
      auto node_it = nodes_by_kernel_count.find(completion.kernel_count);
      if (node_it == nodes_by_kernel_count.end()) {
        std::cerr << "CompletionQueue: unknown completed kernel "
                  << completion.kernel_count << std::endl;
        sendCompletionDispatchBatch(mq_id_program, {}, false, true);
        mq_close(mq_id_program);
        return CompletionWindowResult::FailedAfterStart;
      }
      DAGNode *node = node_it->second;
      CompletionNodeRuntimeState &state = states.at(node);
      if (state.phase != CompletionNodePhase::Dispatched ||
          !completion_kernel_counts.insert(completion.kernel_count).second) {
        std::cerr << "CompletionQueue: duplicate or non-dispatched completion "
                  << completion.kernel_count << std::endl;
        sendCompletionDispatchBatch(mq_id_program, {}, false, true);
        mq_close(mq_id_program);
        return CompletionWindowResult::FailedAfterStart;
      }
      completed_nodes.push_back(node);
    }

    for (size_t completion_index = 0;
         completion_index < completion_batch.completions.size();
         ++completion_index) {
      const S2DKernelProfileData &completion =
          completion_batch.completions[completion_index];
      DAGNode *node = completed_nodes[completion_index];
      CompletionNodeRuntimeState &state = states.at(node);
      state.reserved_devices.clear();
      state.dispatch_started_ns = 0;
      state.predicted_release_ns = 0;
      state.phase = CompletionNodePhase::Complete;
      in_flight_nodes.erase(node);
      ++complete_count;
      for (DAGNode *successor : node->depend_by) {
        if (window_nodes.count(successor) == 0) {
          continue;
        }
        size_t &remaining = remaining_predecessors.at(successor);
        if (remaining == 0) {
          std::cerr << "CompletionQueue: predecessor underflow for kernel "
                    << successor->kernel_count << " in wait "
                    << daemon_wait_count << std::endl;
          sendCompletionDispatchBatch(mq_id_program, {}, false, true);
          mq_close(mq_id_program);
          return CompletionWindowResult::FailedAfterStart;
        }
        --remaining;
        if (remaining == 0) {
          ready_indices.insert(priority_indices.at(successor));
        }
      }
      const int actual_parts = std::max(1, completion.num_parts);
      if (node->exec_proc != completion.device_index ||
          node->num_parts != actual_parts ||
          node->persistent_split != completion.persistent_split) {
        DAEMON_TRACE_STREAM
            << "CompletionQueue: handler adjusted kernel "
            << completion.kernel_count << " from proc " << node->exec_proc
            << " parts " << node->num_parts << " to proc "
            << completion.device_index << " parts " << actual_parts
            << " persistent " << completion.persistent_split
            << std::endl;
      }
      node->exec_rank = 0;
      node->exec_proc = completion.device_index;
      node->num_parts = actual_parts;
      node->persistent_split =
          actual_parts > 1 && completion.persistent_split;
      if (actual_parts <= 1) {
        node->split_devices.clear();
      } else if (node->split_devices.size() >
                 static_cast<size_t>(actual_parts)) {
        node->split_devices.resize(actual_parts);
      }
      node->finish_time = 0.0;
      updateProfileCostTable(completion, 0);
      DAEMON_TRACE_STREAM << "CompletionQueue: complete kernel "
                          << completion.kernel_count << " device "
                          << completion.device_index << " parts "
                          << completion.num_parts << " persistent "
                          << completion.persistent_split << " duration_ns "
                          << completion.duration_ns << std::endl;
    }
  }
}
#endif

// ========【Offline End】

void CPUMonitor() {
  std::ifstream file;
  std::string line;

  static CpuTimes prev, curr;
  file.open("/proc/stat");
  if(!file.is_open()) {
    std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " /proc/stat open failed";
    perror(errorMsg.c_str());
    return;
  }

  while (std::getline(file, line)) {
      if (line.substr(0, 3) == "cpu") {
          std::sscanf(line.c_str(), "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
                      &curr.user, &curr.nice, &curr.system, &curr.idle,
                      &curr.iowait, &curr.irq, &curr.softirq, &curr.steal);
          break;
      }
  }
  file.close();

  unsigned long long prevIdle = prev.idle + prev.iowait;
  unsigned long long currIdle = curr.idle + curr.iowait;

  unsigned long long prevTotal = prevIdle + prev.user + prev.nice + prev.system + prev.irq + prev.softirq + prev.steal;
  unsigned long long currTotal = currIdle + curr.user + curr.nice + curr.system + curr.irq + curr.softirq + curr.steal;

  unsigned long long totalDiff = currTotal - prevTotal;
  unsigned long long idleDiff = currIdle - prevIdle;

  double utilization = totalDiff ? (1.0 - (double)idleDiff / totalDiff) * 100.0 : 0.0;
  // std::cout << "CPU Utilization: " << utilization << "%" << std::endl;
  prev = curr;

  size_t mem_available = 0;
  file.open("/proc/meminfo");
  if(!file.is_open()) {
    std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " /proc/meminfo open failed";
    perror(errorMsg.c_str());
    return;
  }

  while (std::getline(file, line)) {
      if (line.find("MemAvailable:") == 0) {
          std::sscanf(line.c_str(), "MemAvailable: %zu kB", &mem_available);
          break;
      }
  }
  file.close();
  // std::cout << "Memory available: " << mem_available << " kB" << std::endl;

  {
    std::lock_guard<std::mutex> lock(monitor_state_mutex);
    device_monitor_info[0] = MonitorInfo{"CPU", utilization, mem_available};
  }
}

int getCudaPciBusId(const sycl::device &device) {
  if (device.get_backend() != sycl::backend::ext_oneapi_cuda) {
      return -1;
  }
  // cudaGetDevice() reports the calling thread's current context rather than
  // the device being iterated. Query the SYCL device's native CUDA handle so
  // monitor load and capability data are attached to the correct processor.
  const int cudaDevice =
      sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
  int busId;
  const cudaError_t err =
      cudaDeviceGetAttribute(&busId, cudaDevAttrPciBusId, cudaDevice);
  if (err != cudaSuccess) {
      throw std::runtime_error("Failed to get PCI Bus ID for the CUDA device.");
  }
  return busId;
}

int MonitorInit() {
  std::lock_guard<std::mutex> lock(monitor_state_mutex);
  index_sycl_nvml.clear();
  index_nvml_sycl.clear();
  index_sycl_device_identity.clear();
  device_capability.resize(1);
  device_capability[0] = inferCpuCapabilityFromName(readCpuModelName());

  nvmlReturn_t result;
  result = nvmlInit();
  if (result != NVML_SUCCESS) {
    std::string errorMsg = "Failed to initialize NVML: " + std::string(nvmlErrorString(result));
    perror(errorMsg.c_str());
    return -1;
  }
  
  unsigned int device_count = 0;
  result = nvmlDeviceGetCount(&device_count);
  if (result != NVML_SUCCESS) {
      std::string errorMsg = "Failed to get device count: " + std::string(nvmlErrorString(result));
      perror(errorMsg.c_str());
      nvmlShutdown();
      return -1;
  }
  DAEMON_TRACE_STREAM << "Number of GPUs: " << device_count << std::endl;
  
  std::vector<int> nvmlBusIds;
  std::vector<std::string> nvmlDeviceIdentities;
  for (int i = 0; i < device_count; ++i) {
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result != NVML_SUCCESS) {
      std::string errorMsg = "Failed to get device handle for device " + std::to_string(i) + ": " + std::string(nvmlErrorString(result));
      perror(errorMsg.c_str());
      continue;
    }

    nvmlPciInfo_t pciInfo;
    result = nvmlDeviceGetPciInfo(device, &pciInfo);
    if (result != NVML_SUCCESS) {
      std::string errorMsg = "Failed to get PCI info for device " + std::to_string(i) + ": " + std::string(nvmlErrorString(result));
      perror(errorMsg.c_str());
      continue;
    }

    const std::string pci_bus_id = pciInfo.busId;
    int nvmlBusId =
        std::stoi(pci_bus_id.substr(9, 2), nullptr, 16);
    DAEMON_TRACE_STREAM << "GPU " << i << ": PCI Bus ID: " << pciInfo.busId << " int: " << nvmlBusId << std::endl;
    nvmlBusIds.push_back(nvmlBusId);
    nvmlDeviceIdentities.push_back("pci-" + pci_bus_id);
  }

  std::vector<sycl::device> globalDevices = sycl::device::get_devices();
  globalDevices.erase(
    std::remove_if(
      globalDevices.begin(), 
      globalDevices.end(),
      [](const sycl::device& d) { return d.is_accelerator(); }
    ),
    globalDevices.end()
  );
  for (int i = 0; i < globalDevices.size(); i++) {
    sycl::device device = globalDevices[i];
    if (i >= static_cast<int>(device_capability.size())) {
      device_capability.resize(i + 1, fallbackCapabilityForProc(i));
    }
    // **注意** 获取device::name必不可少 不然无法切换cuda上下文
    std::string sycl_device_name =
        device.get_info<sycl::info::device::name>();
    if (device.is_cpu()) {
      device_capability[0] =
          maxCapability(device_capability[0],
                        inferDeviceCapabilityFromName(sycl_device_name, true));
    }
    int busId = getCudaPciBusId(device);
    DAEMON_TRACE_STREAM << "SYCL Device " << i << " (" << sycl_device_name
              << "): PCI Bus ID: " << busId << std::endl;

    if (busId != -1) {
      auto it = std::find(nvmlBusIds.begin(), nvmlBusIds.end(), busId);
      if (it != nvmlBusIds.end()) {
        index_sycl_nvml[i] = std::distance(nvmlBusIds.begin(), it) + 1;
        index_nvml_sycl[index_sycl_nvml[i]] = i;
        index_sycl_device_identity[i] =
            nvmlDeviceIdentities[std::distance(nvmlBusIds.begin(), it)];
        device_capability[i] =
            inferDeviceCapabilityFromName(sycl_device_name, false);
      }
    }
  }
  for (auto pair : index_sycl_nvml) {
    DAEMON_TRACE_STREAM << "SYCL Device " << pair.first << " mapped to GPU "
              << pair.second << " fp32 capability "
              << device_capability[pair.first].fp32
              << " fp64 capability "
              << device_capability[pair.first].fp64 << " identity "
              << index_sycl_device_identity[pair.first] << std::endl;
  }

  return device_count;
  // return 2; // scale测试用
}

void CudaMonitor(int device_count) {
  nvmlReturn_t result;

  for (unsigned int i = 0; i < device_count; ++i) {
      nvmlDevice_t device;
      char name[NVML_DEVICE_NAME_BUFFER_SIZE];
      nvmlUtilization_t utilization;
      nvmlMemory_t memoryInfo;

      result = nvmlDeviceGetHandleByIndex(i, &device);
      if (result != NVML_SUCCESS) {
        std::string errorMsg = "Failed to get handle for GPU " + std::to_string(i) + ": " + nvmlErrorString(result);
        perror(errorMsg.c_str());
        continue;
      }

      result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
      if (result != NVML_SUCCESS) {
        std::string errorMsg = "Failed to get name for GPU " + std::to_string(i) + ": " + nvmlErrorString(result);
        perror(errorMsg.c_str());
        continue;
      }

      result = nvmlDeviceGetUtilizationRates(device, &utilization);
      if (result != NVML_SUCCESS) {
        std::string errorMsg = "Failed to get utilization for GPU " + std::to_string(i) + ": " + nvmlErrorString(result);
        perror(errorMsg.c_str());
        continue;
      }

      // std::cout << "GPU " << i << " (" << name << "):" << std::endl;
      // std::cout << "  GPU Utilization: " << utilization.gpu << "%" << std::endl;
      // std::cout << "  Memory Utilization: " << utilization.memory << "%" << std::endl;

      result = nvmlDeviceGetMemoryInfo(device, &memoryInfo);
      if (result != NVML_SUCCESS) {
        std::string errorMsg = "Failed to get memory info for GPU " + std::to_string(i) + ": " + nvmlErrorString(result);
        perror(errorMsg.c_str());
        continue;
      }

      // std::cout << "GPU " << i << " (" << name << "):" << std::endl;
      // std::cout << "  Memory Total: " << memoryInfo.total / 1024.0 << " kB" << std::endl;
      // std::cout << "  Memory Used: " << memoryInfo.used / 1024.0 << " kB" << std::endl;
      // std::cout << "  Memory Free: " << memoryInfo.free / 1024.0 << " kB" << std::endl;

      const int nvml_device_index = i + 1;
      auto sycl_it = index_nvml_sycl.find(nvml_device_index);
      if (sycl_it == index_nvml_sycl.end() && !index_nvml_sycl.empty()) {
        continue;
      }
      int device_index = nvml_device_index;
      if (sycl_it != index_nvml_sycl.end()) {
        device_index = sycl_it->second;
      }
      {
        std::lock_guard<std::mutex> lock(monitor_state_mutex);
        if (device_index >= static_cast<int>(device_monitor_info.size())) {
          device_monitor_info.resize(device_index + 1);
        }
        if (device_index >= static_cast<int>(device_capability.size())) {
          device_capability.resize(device_index + 1,
                                   fallbackCapabilityForProc(device_index));
        }
        device_monitor_info[device_index] =
            MonitorInfo{name, utilization.gpu, memoryInfo.free / 1024.0};
        device_capability[device_index] =
            inferDeviceCapabilityFromName(name, false);
      }
  }
}

void *SystemMonitor(void *arg) {
  const int device_count =
      arg != nullptr ? *static_cast<const int *>(arg) : MonitorInit();
  
  while(1) {
    CPUMonitor();
    
    if(device_count != -1) {
      CudaMonitor(device_count);
    }

    // for(int i = 0; i < device_monitor_info.size(); i++) {
    //   std::cout << "Device " << i << " (" << device_monitor_info[i].name << "):" << std::endl;
    //   std::cout << "  Utilization: " << device_monitor_info[i].util_used << "%" << std::endl;
    //   std::cout << "  Memory Available: " << device_monitor_info[i].mem_available << " kB" << std::endl;
    // }

    usleep(MONITOR_INTERVAL);
  }
}

void *SystemSchedulerMonitor(void *arg) {
  MPI_Comm_rank(comm_monitor, &monitor_rank);
  MPI_Comm_size(comm_monitor, &monitor_size);
  DAEMON_TRACE_STREAM << "SystemSchedulerMonitor: MONITOR_Rank " << monitor_rank << " started." << std::endl;

  pthread_t monitor_tid;
  pthread_create(&monitor_tid, NULL, (void *(*)(void *))SystemMonitor, arg);
  pthread_detach(monitor_tid);

  {
    std::lock_guard<std::mutex> lock(monitor_state_mutex);
    ranks_idle.resize(monitor_size, false);
    cluster_monitor_info.resize(monitor_size);
    cluster_device_capability.resize(monitor_size);
  }
  {
    std::lock_guard<std::mutex> lock(monitor_state_mutex);
    cluster_comm_profile_ids.resize(monitor_size, -1);
  }

  // 每个rank彼此感知是否有空闲即可 无需传递所有状态？
  while (1) {
    const int detected_comm_profile_id = detectLocalCommProfileId();

    std::vector<MonitorInfo> local_monitor_snapshot;
    std::vector<ComputeCapability> local_capability_snapshot;
    {
      std::lock_guard<std::mutex> lock(monitor_state_mutex);
      local_monitor_snapshot = device_monitor_info;
      local_capability_snapshot = device_capability;
    }

    int is_idle = 0;
    for (int i = 1; i < local_monitor_snapshot.size(); i++) {
      if (local_monitor_snapshot[i].util_used < MONITOR_THRESHOLD) {
        is_idle = 1;
        break;
      }
    }

    std::vector<int> gathered_ranks_idle(monitor_size, false);
    MPI_Allgather(&is_idle, 1, MPI_INT, gathered_ranks_idle.data(), 1,
                  MPI_INT, comm_monitor);
    {
      std::lock_guard<std::mutex> lock(monitor_state_mutex);
      ranks_idle = std::move(gathered_ranks_idle);
    }
    std::vector<int> gathered_comm_profile_ids(monitor_size, -1);
    MPI_Allgather(&detected_comm_profile_id, 1, MPI_INT,
                  gathered_comm_profile_ids.data(), 1, MPI_INT,
                  comm_monitor);
    {
      std::lock_guard<std::mutex> lock(monitor_state_mutex);
      local_comm_profile_id = detected_comm_profile_id;
      cluster_comm_profile_ids = std::move(gathered_comm_profile_ids);
    }

    std::array<double, MAX_MONITOR_DEVICES * MONITOR_PACKED_FIELDS> local_monitor{};
    const int local_device_count =
        std::min<int>(local_monitor_snapshot.size(), MAX_MONITOR_DEVICES);
    for (int i = 0; i < local_device_count; ++i) {
      const int offset = i * MONITOR_PACKED_FIELDS;
      local_monitor[offset + 0] = 1.0;
      local_monitor[offset + 1] = local_monitor_snapshot[i].util_used;
      local_monitor[offset + 2] =
          static_cast<double>(local_monitor_snapshot[i].mem_available);
      const ComputeCapability capability =
          i < static_cast<int>(local_capability_snapshot.size())
              ? local_capability_snapshot[i]
              : fallbackCapabilityForProc(i);
      local_monitor[offset + 3] = capability.fp32;
      local_monitor[offset + 4] = capability.fp64;
    }

    std::vector<double> packed_monitor(
        monitor_size * MAX_MONITOR_DEVICES * MONITOR_PACKED_FIELDS, 0.0);
    MPI_Allgather(local_monitor.data(),
                  MAX_MONITOR_DEVICES * MONITOR_PACKED_FIELDS, MPI_DOUBLE,
                  packed_monitor.data(),
                  MAX_MONITOR_DEVICES * MONITOR_PACKED_FIELDS, MPI_DOUBLE,
                  comm_monitor);

    {
      std::lock_guard<std::mutex> lock(monitor_state_mutex);
      for (int rank = 0; rank < monitor_size; ++rank) {
        std::vector<MonitorInfo> rank_info;
        std::vector<ComputeCapability> rank_capability;
        for (int device = 0; device < MAX_MONITOR_DEVICES; ++device) {
          const int offset =
              rank * MAX_MONITOR_DEVICES * MONITOR_PACKED_FIELDS +
              device * MONITOR_PACKED_FIELDS;
          if (packed_monitor[offset] == 0.0) {
            continue;
          }
          rank_info.push_back(MonitorInfo{
              device == 0 ? "CPU" : "GPU",
              packed_monitor[offset + 1],
              static_cast<size_t>(packed_monitor[offset + 2])});
          rank_capability.push_back(
              ComputeCapability{packed_monitor[offset + 3],
                                packed_monitor[offset + 4]});
        }
        cluster_monitor_info[rank] = std::move(rank_info);
        cluster_device_capability[rank] = std::move(rank_capability);
      }
    }

    // for (int i = 0; i < monitor_size; i++) {
    //   std::cout << "SystemSchedulerMonitor: MONITOR_Rank " << monitor_rank << " ranks_idle[" << i << "]: " << ranks_idle[i] << std::endl;
    // }

    usleep(MONITOR_GATHER_INTERVAL);
  }
}

void *SystemSchedulerDaemon(void *arg) {
  int syclapp_count = *(int *)arg;
  int local_pid = globalcount_to_pid[syclapp_count];
  
  ProgramInfo &program_info = pid_to_program[local_pid];
  mqd_t mq_id_daemon = EstablishDaemon(local_pid);
  MPI_Comm &comm_daemon = program_info.comm_daemon;
  int &daemon_rank = program_info.daemon_rank;
  int &daemon_size = program_info.daemon_size;
  int &master_rank = program_info.master_rank;

  std::map<int, int> exec_kernel_device; // kernel_count -> device_index in monitor_info

  //【scale】在handler第一次时告知 handler在包括scalecount之前都不通信
  {
    if (globalcount_to_scalecount.find(syclapp_count) == globalcount_to_scalecount.end()) {
      std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " globalcount_to_scalecount not found";
      perror(errorMsg.c_str());
      exit(1);
    }
    int scale_count = globalcount_to_scalecount[syclapp_count];
    
    // ====【scale 第一个kernel】
    if (scale_count > 1) {
      DAEMON_TRACE_STREAM << "SystemSchedulerDaemon: Rank " << daemon_rank << " for PID " << local_pid << " with ScaleCount " << scale_count << " started." << std::endl;

      // 接收
      char buffer[MAX_MSG_DAEMON_SIZE];
      ssize_t bytes_received = mq_receive(mq_id_daemon, buffer, MAX_MSG_DAEMON_SIZE, nullptr);
      if (bytes_received > 0) {
        // std::string received_data(buffer, bytes_received);
        // S2DKernelReqData kernel_req_data = S2DKernelReqData::deserialize(received_data);
        DAEMON_TRACE_STREAM << "SystemSchedulerDaemon: Rank " << daemon_rank << " for PID " << local_pid << " received first kernel" << std::endl;
      } else {
        std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " DAEMON mq_receive failed";
        perror(errorMsg.c_str());
        exit(1);
      }

      // 返回
      mqd_t mq_id_program;
      D2SKernelExecInfo kernel_exec_info;
      {
        char MESSAGE_QUEUE_PROGRAM_NAME[MESSAGE_QUEUE_PROGRAM_NAME_MAX];
        sprintf(MESSAGE_QUEUE_PROGRAM_NAME, MESSAGE_QUEUE_PROGRAM_PATTERN, local_pid);
        mq_id_program = mq_open(MESSAGE_QUEUE_PROGRAM_NAME, O_WRONLY);
        if (mq_id_program == -1) {
          std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " mq_id_program open failed";
          perror(errorMsg.c_str());
          exit(1);
        }
        kernel_exec_info.scale_count = scale_count;
        const std::vector<MonitorInfo> monitor_snapshot =
            snapshotLocalMonitorInfo();
        int min_util = 100;
        int min_util_index = -1;
        for (int i = 1; i < monitor_snapshot.size(); i++) {
          if (monitor_snapshot[i].util_used < min_util) {
            min_util = monitor_snapshot[i].util_used;
            min_util_index = i;
          }
        }
        kernel_exec_info.device_index = min_util_index;
        exec_kernel_device[scale_count] = kernel_exec_info.device_index;
        std::string serialized_data = kernel_exec_info.serialize();
        size_t message_size = serialized_data.size();
        mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
        DAEMON_TRACE_STREAM << "SystemSchedulerDaemon: Rank " << daemon_rank << " for PID " << local_pid << " sent first kernel" << std::endl;
      }
    } else {
      DAEMON_TRACE_STREAM << "SystemSchedulerDaemon: Rank " << daemon_rank << " for PID " << local_pid << " started." << std::endl;
    }

    // ====【scale daemon处理依赖】
    // 下一次接收kernel必然是scalecount
    if (scale_count > 1) {
      S2DKernelReqData kernel_req_data;
      {
        char buffer[MAX_MSG_DAEMON_SIZE];
        ssize_t bytes_received = mq_receive(mq_id_daemon, buffer, MAX_MSG_DAEMON_SIZE, nullptr);
        if (bytes_received > 0) {
          std::string received_data(buffer, bytes_received);
          kernel_req_data = S2DKernelReqData::deserialize(received_data);
          for (SyclReqData &req : kernel_req_data.reqs) {
            DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Scale mq_receive kernel_req_data pid: " << kernel_req_data.pid << " count: " << kernel_req_data.kernel_count << " req_count: " << req.req_count << " pointer: " << req.mem_pointer << std::endl;
          }
        } else {
          std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " DAEMON mq_receive failed";
          perror(errorMsg.c_str());
          exit(1);
        }
        DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Scale mq_receive kernel_req_data pid: " << kernel_req_data.pid << " count: " << kernel_req_data.kernel_count << std::endl;
      }

      {
        for (SyclReqData &req : kernel_req_data.reqs) {
          if (req.req_accmode == acc_mode::read || req.req_accmode == acc_mode::read_write || req.req_accmode == acc_mode::atomic) {
            int elem_size = req.elem_size;
            int buff_size = req.buff_size;
            int data_rank = master_rank;

            std::vector<DATA_TYPE> host_data(elem_size * buff_size);
            MPI_Recv(host_data.data(), elem_size * buff_size, MPI_BYTE, data_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
            DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Scale received data from rank " << data_rank << std::endl;

            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_req_data.kernel_count, req.req_count, elem_size * buff_size);
            writeToSharedMemory(handle, host_data.data(), elem_size * buff_size);
            DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Scale write to shared" << std::endl;
            waitForReadCompletion(handle);
            cleanupSharedMemory(handle, elem_size * buff_size);
            DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Scale waitForReadCompletion" << std::endl;
          }
        }
      }
    }
  }

  // 维护一个SYCLAPP的所有kernel的依赖关系 DAG相关
  std::vector<DAGNode *> kernel_dag_nodes; // 所有kernel对应的DAG

  //【通用情况】
  while (1) {
    // ====【接收program通信】
    S2DKernelReqData kernel_req_data;
    {
      DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": waiting req from handler" << std::endl;
      char buffer[MAX_MSG_DAEMON_SIZE];
      ssize_t bytes_received = mq_receive(mq_id_daemon, buffer, MAX_MSG_DAEMON_SIZE, nullptr);
      if (bytes_received > 0) {
        if (std::string(buffer, bytes_received) == "EXIT") {
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": SYCLAPP finish" << std::endl;
          break;
        }
        std::string received_data(buffer, bytes_received);
        kernel_req_data = S2DKernelReqData::deserialize(received_data);
        for (SyclReqData &req : kernel_req_data.reqs) {
          // DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": mq_receive kernel_req_data pid: " << kernel_req_data.pid << " count: " << kernel_req_data.kernel_count << " req_count: " << req.req_count << " pointer: " << req.mem_pointer << std::endl;
        }
      } else {
        std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " DAEMON mq_receive failed";
        perror(errorMsg.c_str());
        exit(1);
      }
      DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": mq_receive kernel_req_data pid: " << kernel_req_data.pid << " count: " << kernel_req_data.kernel_count << std::endl;
    }

    const std::vector<MonitorInfo> monitor_snapshot =
        snapshotLocalMonitorInfo();
    const std::vector<int> ranks_idle_snapshot = snapshotRankIdleState();

    // ====【调度决策并发给其他rank】
    D2DKernelSchedInfo kernel_sched_info;
    bool scale = false;
    {
      if (daemon_rank == master_rank) {
        // [1] [rank0] 构建DAG 确定依赖的kernel 查找依赖的kernel在哪个rank执行
        DAGNode *node = new DAGNode(kernel_req_data);
        std::map<SyclReqData, std::set<int>> req_ranks = generateDAG(kernel_dag_nodes, node);
        for (auto pair : req_ranks) {
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << " req_rank: " << pair.first.kernel_count << "-" << pair.first.req_count << " pointer: " << pair.first.mem_pointer << " rank: ";
          for (int rank : pair.second) {
            DAEMON_TRACE_STREAM << rank << " ";
          }
          DAEMON_TRACE_STREAM << std::endl;
        }
        kernel_sched_info.kernel_count = kernel_req_data.kernel_count;
        // OPTI 2 [allrank] 调度决策(设备负载/性能预测/通信开销/计算开销)
        // 在master上只需要确认rank是否还有device空闲 具体的device应由各daemon监控和选择
        int most_rank = mostDepdRank(req_ranks);
        // 没依赖 master有空闲就执行 无空闲应该扩容
        // 不太可能没依赖 必然会依赖初始化的write 无空闲扩容的情况很少
        if (most_rank == -1) { 
          bool idle = false;
          for (int i = 1; i < monitor_snapshot.size(); i++) {
            if (monitor_snapshot[i].util_used < MONITOR_THRESHOLD) {
              idle = true;
              break;
            }
          }
          if (idle) {
            kernel_sched_info.exec_rank = daemon_rank;
            kernel_sched_info.exec_device = -1;
          } else {
            for (int i = 0; i < ranks_idle_snapshot.size(); i++) {
              if (i == monitor_rank) {
                continue;
              }
              if (ranks_idle_snapshot[i]) {
                kernel_sched_info.exec_rank = i;

                std::lock_guard<std::mutex> lock(*pid_to_scalecount_mutex[local_pid]);
                pid_to_scalecount_queue[local_pid]->push(std::make_pair(kernel_req_data.kernel_count, 0));
                pid_to_scalecount_cv[local_pid]->notify_one();

                scale = true;
                DAEMON_TRACE_STREAM << "Rank " << daemon_rank << " NO DEPD NOTIFY SCALE rank: " << i << std::endl;
                break;
              }
            }
          }

        }
        // 有依赖 有最多依赖的rank 如果rank没空也应该扩容
        // 依赖在master上就扩容 在其他rank就继续执行 OPTI 未实现其他rank满足依赖
        else {
          int kernel_depth = node->depth;
          int kernel_count = node->kernel_count;
          std::vector<DAGNode *> nearest_nodes; // depth-1的kernel
          for (DAGNode *node : node->depend_on) {
            if (node->depth == kernel_depth - 1) {
              nearest_nodes.push_back(node);
            }
          }
          int count_diff = kernel_depth;
          DAGNode *closest_node = nullptr; // count最近的kernel
          for (DAGNode *node : nearest_nodes) {
            if (kernel_count - node->kernel_count < count_diff) {
              count_diff = kernel_count - node->kernel_count;
              closest_node = node;
            }
          }

          if (count_diff == 1) {
            kernel_sched_info.exec_rank = closest_node->exec_rank; // OPTI 还需要指定device
            kernel_sched_info.exec_device = closest_node->kernel_count;
          } else {
            if (most_rank == daemon_rank) { // 依赖在master
              bool idle = false;
              for (int i = 1; i < monitor_snapshot.size(); i++) {
                if (monitor_snapshot[i].util_used < MONITOR_THRESHOLD) {
                  idle = true;
                  break;
                }
              }
              if (idle) { // master空闲
                kernel_sched_info.exec_rank = daemon_rank;
                kernel_sched_info.exec_device = -1;
              } else { // master不空闲 找空闲rank扩容
                for (int i = 0; i < ranks_idle_snapshot.size(); i++) {
                  if (i == monitor_rank) {
                    continue;
                  }
                  if (ranks_idle_snapshot[i]) {
                    kernel_sched_info.exec_rank = i;

                    std::lock_guard<std::mutex> lock(*pid_to_scalecount_mutex[local_pid]);
                    pid_to_scalecount_queue[local_pid]->push(std::make_pair(kernel_req_data.kernel_count, 0));
                    pid_to_scalecount_cv[local_pid]->notify_one();

                    scale = true;
                    DAEMON_TRACE_STREAM << "Rank " << daemon_rank << " YES DEPD NOTIFY SCALE rank: " << i << std::endl;
                    break;
                  }
                }
              }
            } else { // 依赖在其他rank
              kernel_sched_info.exec_rank = most_rank;
              kernel_sched_info.exec_device = -1;
            }
          }
        }
        kernel_sched_info.req_rank = chooseReqRank(req_ranks, daemon_rank);
        node->exec_rank = kernel_sched_info.exec_rank;
      }
      // BUG online的扩容逻辑是错的 没考虑不空闲 也没考虑onrun

      // 对于master 告知scale需要扩充 此时scale新线程都未建立 不会向scale发
      // 对于scale_rank 等建立到到这里不应该接收
      // 对于其他rank 仍需接收
      // [4] [rank0][MPI] bcast (这个kernel 由哪个rank执行 依赖于哪些数据 这些数据在哪些rank上)
      //     [rank!0][MPI] bcast 接收并记录
      SendD2DKernelSchedInfo(comm_daemon, master_rank, daemon_rank, globalcount_to_onrun[syclapp_count], kernel_sched_info);
      // BcastD2DKernelSchedInfo(comm_daemon, master_rank, daemon_rank, kernel_sched_info);
      DAEMON_TRACE_STREAM << "Rank " << daemon_rank << " kernel_sched_info.exec_rank: " << kernel_sched_info.exec_rank << std::endl;
    }

    // ====【向program发送执行决策】
    mqd_t mq_id_program;
    D2SKernelExecInfo kernel_exec_info;
    std::vector<SyclReqData> req_for_rank;
    {
      char MESSAGE_QUEUE_PROGRAM_NAME[MESSAGE_QUEUE_PROGRAM_NAME_MAX];
      sprintf(MESSAGE_QUEUE_PROGRAM_NAME, MESSAGE_QUEUE_PROGRAM_PATTERN, kernel_req_data.pid);
      mq_id_program = mq_open(MESSAGE_QUEUE_PROGRAM_NAME, O_WRONLY);
      if (mq_id_program == -1) {
        std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " mq_id_program open failed";
        perror(errorMsg.c_str());
        exit(1);
      }

      if (scale) {
        kernel_exec_info.scale_count = -1;
      }
      kernel_exec_info.kernel_count = kernel_sched_info.kernel_count;
      if (daemon_rank == kernel_sched_info.exec_rank) {
        kernel_exec_info.exec = true;
        req_for_rank = kernel_sched_info.get_req_for_exec_rank(daemon_rank);
        // 可能无依赖或有依赖 可能是master也可能是接收通信的其他rank
        // 空闲状态可能有变化 但一定执行 选择最空闲的device执行
        if (kernel_sched_info.exec_device == -1) {
          std::vector<int> idle_devices;
          for (int i = 1; i < monitor_snapshot.size(); i++) {
            if (monitor_snapshot[i].util_used < MONITOR_THRESHOLD) {
              idle_devices.push_back(i);
            }
          }
          if (idle_devices.size() > 0) { // 随机选择
            int rand_index = rand() % idle_devices.size();
            kernel_exec_info.device_index = idle_devices[rand_index];
          } else { // 找利用率最低的
            int min_util = 100;
            int min_util_index = -1;
            for (int i = 1; i < monitor_snapshot.size(); i++) {
              if (monitor_snapshot[i].util_used < min_util) {
                min_util = monitor_snapshot[i].util_used;
                min_util_index = i;
              }
            }
            kernel_exec_info.device_index = min_util_index;
          }
        }
        else if (kernel_sched_info.exec_device >= 1) {
          kernel_exec_info.device_index = exec_kernel_device[kernel_sched_info.exec_device];
        }
        exec_kernel_device[kernel_exec_info.kernel_count] = kernel_exec_info.device_index;
      }
      else {
        req_for_rank = kernel_sched_info.get_req_for_rank(daemon_rank);
      }
      kernel_exec_info.req_counts.resize(req_for_rank.size());
      for (int i = 0; i < req_for_rank.size(); ++i) {
        kernel_exec_info.req_counts[i] = req_for_rank[i].req_count;
      }

      // 向负责的进程mq发送 需要device->host的data
      std::string serialized_data = kernel_exec_info.serialize();
      size_t message_size = serialized_data.size();
      mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
      DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": mq_send kernel_exec_info exec: " << kernel_exec_info.exec << " req_counts.size: " << kernel_exec_info.req_counts.size() << " device_index: " << kernel_exec_info.device_index << std::endl;
    }

    // ====【scale 向daemon发送依赖数据】
    // OPTI master发送 若其他rank发送还需等待master通知 默认scale所需master都有最新？
    if (scale) {
      for (SyclReqData &req : kernel_req_data.reqs) {
        if (req.req_accmode == acc_mode::read || req.req_accmode == acc_mode::read_write || req.req_accmode == acc_mode::atomic) {
          int elem_size = req.elem_size;
          int buff_size = req.buff_size;

          std::vector<DATA_TYPE> host_data(elem_size * buff_size);
          SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_req_data.kernel_count, req.req_count, elem_size * buff_size);
          readFromSharedMemory(handle, host_data.data(), elem_size * buff_size);
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Scale data read successfully." << std::endl;
          cleanupSharedMemory(handle, elem_size * buff_size);

          MPI_Send(host_data.data(), elem_size * buff_size, MPI_BYTE, kernel_sched_info.exec_rank, 0, comm_daemon);
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Scale sent data to rank " << kernel_sched_info.exec_rank << std::endl;
        }
      }
    }

    // ====【为执行的rank满足依赖】scale完就不需要走这段流程
    else {
      // [5] [单rank] 被依赖的kernel的数据device->host
      // 如果执行 要检查是否要从其他rank获取数据
      // 如果不执行 要检查是否需要host->device 给其他rank发数据
      // 说明有需要从其他rank获取的数据
      if (kernel_sched_info.req_rank.size() != kernel_sched_info.get_req_for_rank(kernel_sched_info.exec_rank).size()) {
        DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Exec Rank: " << kernel_sched_info.exec_rank << " need data from other" << std::endl;
        // 此rank不执行kernel 且kernel有依赖此rank的数据
        if (daemon_rank != kernel_sched_info.exec_rank && req_for_rank.size() > 0) {
          for (SyclReqData &req : req_for_rank) {
            // std::cout << "Rank " << daemon_rank << ": Req " << i << std::endl;
            // [6] 从SYCL进程接受host的data
            // 因为是写读共享内存是阻塞的 不需要等待SYCL进程的通知
            int elem_size = req.elem_size;
            int buff_size = req.buff_size;

            std::vector<DATA_TYPE> host_data(elem_size * buff_size);
            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_sched_info.kernel_count, req.req_count, elem_size * buff_size);
            readFromSharedMemory(handle, host_data.data(), elem_size * buff_size);
            DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Data read successfully." << std::endl;
            cleanupSharedMemory(handle, elem_size * buff_size);

            // [7] [双rank][MPI] isend:host->buffer
            MPI_Send(host_data.data(), elem_size * buff_size, MPI_BYTE, kernel_sched_info.exec_rank, 0, comm_daemon);
            DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Sent data to rank " << kernel_sched_info.exec_rank << std::endl;
          }
        }

        // 此rank执行kernel 且必然需要从其他rank拿数据
        if (daemon_rank == kernel_sched_info.exec_rank) {
          for (SyclReqData &req : req_for_rank) {
            int elem_size = req.elem_size;
            int buff_size = req.buff_size;
            int data_rank = kernel_sched_info.req_rank[req];

            std::vector<DATA_TYPE> host_data(elem_size * buff_size);
            // [7] [双rank][MPI] irecv:buffer->host
            MPI_Recv(host_data.data(), elem_size * buff_size, MPI_BYTE, data_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
            DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Received data from rank " << data_rank << std::endl;

            // [8] 把从其他rank接受的data发给SYCL进程
            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_sched_info.kernel_count, req.req_count, elem_size * buff_size);
            writeToSharedMemory(handle, host_data.data(), elem_size * buff_size);
            DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Write to shared" << std::endl;
            waitForReadCompletion(handle);
            cleanupSharedMemory(handle, elem_size * buff_size);
            DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": waitForReadCompletion" << std::endl;
          }
        }
      }
    }
    mq_close(mq_id_program);
  }

  mq_close(mq_id_daemon);
  return NULL;
}

void commExecInfo(std::vector<D2DKernelSchedInfo> &kernel_sched_order_infos, int &local_pid, int &daemon_rank, MPI_Comm &comm_daemon) {
  // ====【向program发送执行决策】
  // 每个kernel都会给hanler发送
  mqd_t mq_id_program;
  std::vector<D2SKernelExecInfo> kernel_exec_infos;
  std::vector<std::vector<SyclReqData>> req_for_ranks;
  {
    char MESSAGE_QUEUE_PROGRAM_NAME[MESSAGE_QUEUE_PROGRAM_NAME_MAX];
    sprintf(MESSAGE_QUEUE_PROGRAM_NAME, MESSAGE_QUEUE_PROGRAM_PATTERN, local_pid);
    mq_id_program = mq_open(MESSAGE_QUEUE_PROGRAM_NAME, O_WRONLY);
    if (mq_id_program == -1) {
      std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " mq_id_program open failed";
      perror(errorMsg.c_str());
      exit(1);
    }

    // 填充D2SKernelExecInfo
    // **注意** req_rank是在DAG的决策中确定的
    for (int order = 0; order < kernel_sched_order_infos.size(); ++order) {
      D2DKernelSchedInfo &kernel_sched_info = kernel_sched_order_infos[order];
      D2SKernelExecInfo kernel_exec_info;
      std::vector<SyclReqData> req_for_rank;

      kernel_exec_info.kernel_count = kernel_sched_info.kernel_count;
      if (daemon_rank == kernel_sched_info.exec_rank) {
        kernel_exec_info.exec = true;
        req_for_rank = kernel_sched_info.get_req_for_exec_rank(daemon_rank);

        kernel_exec_info.device_index = kernel_sched_info.exec_device;
        kernel_exec_info.num_parts = kernel_sched_info.num_parts;
        kernel_exec_info.persistent_split =
            kernel_sched_info.persistent_split;
        kernel_exec_info.split_devices = kernel_sched_info.split_devices;
      }
      else {
        req_for_rank = kernel_sched_info.get_req_for_rank(daemon_rank);
      }
      kernel_exec_info.req_counts.resize(req_for_rank.size());
      for (int i = 0; i < req_for_rank.size(); ++i) {
        kernel_exec_info.req_counts[i] = req_for_rank[i].req_count;
      }

      kernel_exec_infos.push_back(kernel_exec_info);
      req_for_ranks.push_back(req_for_rank);
    }

    // 发送给handler
    std::string serialized_data;
    for (const auto &kernel_exec_info : kernel_exec_infos) {
        serialized_data += kernel_exec_info.serialize();
    }
    size_t message_size = serialized_data.size();
    sendOfflineMqPayload(mq_id_program, serialized_data, MAX_MSG_PROGRAM_SIZE,
                         "commExecInfo kernel_exec_infos");
    DAEMON_TRACE_STREAM << "commExecInfo === Rank " << daemon_rank
              << ": mq_send kernel_exec_infos size: "
              << kernel_exec_infos.size() << " mqsize: " << message_size
              << std::endl;
    for (const D2SKernelExecInfo &kernel_exec_info : kernel_exec_infos) {
      DAEMON_TRACE_STREAM << "commExecInfo === Rank " << daemon_rank
                << ": kernel_count " << kernel_exec_info.kernel_count
                << " exec " << kernel_exec_info.exec
                << " device_index " << kernel_exec_info.device_index
                << " num_parts " << kernel_exec_info.num_parts
                << " split_devices";
      for (int split_device : kernel_exec_info.split_devices) {
        DAEMON_TRACE_STREAM << " " << split_device;
      }
      DAEMON_TRACE_STREAM
                << " req_counts " << kernel_exec_info.req_counts.size()
                << std::endl;
    }
  }

  // ====【为执行的rank满足依赖】
  // **注意** daemon对所有依赖传递知情 对每一个kernel都应该在此函数中准备接收和传输
  for (int order = 0; order < kernel_sched_order_infos.size(); ++order) {
    D2DKernelSchedInfo &kernel_sched_info = kernel_sched_order_infos[order];
    D2SKernelExecInfo &kernel_exec_info = kernel_exec_infos[order];
    std::vector<SyclReqData> &req_for_rank = req_for_ranks[order];
    // 不只是此rank执行的kernel相关 可能其他rank需要此rank的数据
    // 此判断说明此kernel有需要从其他rank获取的数据
    // DAEMON_TRACE_STREAM << "OfflineCommExecInfo: Rank " << daemon_rank << ": Kernel_order: " << order << " : Kernel_count: " << kernel_sched_info.kernel_count << std::endl;

    if (kernel_sched_info.req_rank.size() != kernel_sched_info.get_req_for_rank(kernel_sched_info.exec_rank).size()) {
      DAEMON_TRACE_STREAM << "OfflineCommExecInfo: Rank " << daemon_rank << ": Exec Rank: " << kernel_sched_info.exec_rank << " need data from other" << std::endl;

      // 此rank不执行kernel 且kernel有依赖此rank的数据
      if (daemon_rank != kernel_sched_info.exec_rank && req_for_rank.size() > 0) {
        DAEMON_TRACE_STREAM << "OfflineCommExecInfo: Rank " << daemon_rank << ": NO exec, provide" << std::endl;
        for (SyclReqData &req : req_for_rank) {
          // [6] 从SYCL进程接受host的data
          // 因为是写读共享内存是阻塞的 不需要等待SYCL进程的通知
          int elem_size = req.elem_size;
          int buff_size = req.buff_size;
          std::vector<DATA_TYPE> host_data(elem_size * buff_size);

          DAEMON_TRACE_STREAM << "===SEND Rank " << daemon_rank << ": Req hostdata: " << elem_size << "*" << buff_size << std::endl;

          SharedMemoryHandle handle = initSharedMemory(local_pid, kernel_sched_info.kernel_count, req.req_count, elem_size * buff_size);
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": initSharedMemory" << std::endl;

          readFromSharedMemory(handle, host_data.data(), elem_size * buff_size);
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Data read successfully." << std::endl;

          cleanupSharedMemory(handle, elem_size * buff_size);

          // [7] [双rank][MPI] isend:host->buffer
          MPI_Send(host_data.data(), elem_size * buff_size, MPI_BYTE, kernel_sched_info.exec_rank, 0, comm_daemon);
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Sent data to rank " << kernel_sched_info.exec_rank << std::endl;
        }
      }

      // 此rank执行kernel 且必然需要从其他rank拿数据
      if (daemon_rank == kernel_sched_info.exec_rank) {
        DAEMON_TRACE_STREAM << "OfflineCommExecInfo: Rank " << daemon_rank << ": Exec, receive" << std::endl;
        for (SyclReqData &req : req_for_rank) {
          int elem_size = req.elem_size;
          int buff_size = req.buff_size;
          int data_rank = kernel_sched_info.req_rank[req];
          std::vector<DATA_TYPE> host_data(elem_size * buff_size);

          DAEMON_TRACE_STREAM << "---RECV Rank " << daemon_rank << ": Req data from rank " << data_rank << " elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;
          
          // [7] [双rank][MPI] irecv:buffer->host
          MPI_Recv(host_data.data(), elem_size * buff_size, MPI_BYTE, data_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Received data from rank " << data_rank << std::endl;

          // [8] 把从其他rank接受的data发给SYCL进程
          SharedMemoryHandle handle = initSharedMemory(local_pid, kernel_sched_info.kernel_count, req.req_count, elem_size * buff_size);
          writeToSharedMemory(handle, host_data.data(), elem_size * buff_size);
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": Write to shared" << std::endl;

          waitForReadCompletion(handle);
          cleanupSharedMemory(handle, elem_size * buff_size);
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": waitForReadCompletion" << std::endl;
        }
      }
    }
  }
  mq_close(mq_id_program);
}

static std::string
serializeProfileSamples(const std::vector<S2DKernelProfileData> &profiles) {
  std::ostringstream oss;
  oss << profiles.size() << "\n";
  for (const S2DKernelProfileData &profile : profiles) {
    oss << profile.serialize();
  }
  return oss.str();
}

static std::vector<S2DKernelProfileData>
deserializeProfileSamples(const std::string &data) {
  std::istringstream iss(data);
  size_t profile_count = 0;
  iss >> profile_count;
  iss.ignore();

  std::vector<S2DKernelProfileData> profiles;
  for (size_t i = 0; i < profile_count; ++i) {
    profiles.push_back(S2DKernelProfileData::deserialize(iss));
  }
  return profiles;
}

static constexpr int OFFLINE_PROFILE_DATA_TAG = 202;
static constexpr const char *OFFLINE_PROFILE_DONE_MARKER =
    "SNMD_PROFILE_DONE\n";

struct PendingOfflineProfileSend {
  std::string payload;
  MPI_Request request = MPI_REQUEST_NULL;
};

static void progressPendingOfflineProfileSends(
    std::list<PendingOfflineProfileSend> &pending_sends) {
  for (auto it = pending_sends.begin(); it != pending_sends.end();) {
    int complete = 0;
    MPI_Test(&it->request, &complete, MPI_STATUS_IGNORE);
    if (complete) {
      it = pending_sends.erase(it);
    } else {
      ++it;
    }
  }
}

static void consumeOfflineProfilePayloadAtMaster(
    const std::string &payload, int source_rank,
    std::set<int> &finished_profile_ranks) {
  if (payload == OFFLINE_PROFILE_DONE_MARKER) {
    // The marker uses the same source/tag as samples, so MPI's non-overtaking
    // guarantee ensures every earlier sample from this rank was consumed.
    finished_profile_ranks.insert(source_rank);
    return;
  }
  std::vector<S2DKernelProfileData> remote_profiles =
      deserializeProfileSamples(payload);
  for (const S2DKernelProfileData &profile : remote_profiles) {
    updateProfileCostTable(profile, source_rank);
  }
}

static bool receiveOneOfflineProfilePayloadAtMaster(
    MPI_Comm &comm_daemon, bool blocking,
    std::set<int> &finished_profile_ranks) {
  MPI_Status status;
  if (blocking) {
    MPI_Probe(MPI_ANY_SOURCE, OFFLINE_PROFILE_DATA_TAG, comm_daemon, &status);
  } else {
    int available = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, OFFLINE_PROFILE_DATA_TAG, comm_daemon,
               &available, &status);
    if (!available) {
      return false;
    }
  }

  int payload_size = 0;
  MPI_Get_count(&status, MPI_CHAR, &payload_size);
  if (payload_size <= 0) {
    // Consume even a malformed zero-byte internal message so it cannot poison
    // subsequent probes. It is not accepted as a shutdown marker.
    MPI_Recv(nullptr, 0, MPI_CHAR, status.MPI_SOURCE,
             OFFLINE_PROFILE_DATA_TAG, comm_daemon, MPI_STATUS_IGNORE);
    return true;
  }

  std::string payload(static_cast<size_t>(payload_size), '\0');
  MPI_Recv(payload.data(), payload_size, MPI_CHAR, status.MPI_SOURCE,
           OFFLINE_PROFILE_DATA_TAG, comm_daemon, MPI_STATUS_IGNORE);
  consumeOfflineProfilePayloadAtMaster(payload, status.MPI_SOURCE,
                                       finished_profile_ranks);
  return true;
}

static void publishAndDrainOfflineProfiles(
    const std::vector<S2DKernelProfileData> &local_profiles,
    MPI_Comm &comm_daemon, int daemon_rank, int master_rank,
    std::list<PendingOfflineProfileSend> &pending_sends,
    std::set<int> &finished_profile_ranks) {
  if (daemon_rank == master_rank) {
    for (const S2DKernelProfileData &profile : local_profiles) {
      updateProfileCostTable(profile, daemon_rank);
    }
    while (receiveOneOfflineProfilePayloadAtMaster(
        comm_daemon, /*blocking=*/false, finished_profile_ranks)) {
    }
    return;
  }

  progressPendingOfflineProfileSends(pending_sends);
  if (local_profiles.empty()) {
    return;
  }

  pending_sends.emplace_back();
  PendingOfflineProfileSend &send = pending_sends.back();
  send.payload = serializeProfileSamples(local_profiles);
  MPI_Isend(send.payload.data(), static_cast<int>(send.payload.size()),
            MPI_CHAR, master_rank, OFFLINE_PROFILE_DATA_TAG, comm_daemon,
            &send.request);
}

static void flushOfflineProfilesOnExit(
    MPI_Comm &comm_daemon, int daemon_rank, int master_rank,
    const std::set<int> &onrun_ranks,
    std::list<PendingOfflineProfileSend> &pending_sends,
    std::set<int> &finished_profile_ranks) {
  if (daemon_rank == master_rank) {
    auto all_workers_finished = [&] {
      for (int rank : onrun_ranks) {
        if (rank != master_rank && !finished_profile_ranks.count(rank)) {
          return false;
        }
      }
      return true;
    };
    while (!all_workers_finished()) {
      receiveOneOfflineProfilePayloadAtMaster(
          comm_daemon, /*blocking=*/true, finished_profile_ranks);
    }
    return;
  }

  // Shutdown is the only point where profile delivery is allowed to join. No
  // further application work can be delayed, and the marker lets the master
  // drain all earlier nonblocking sends without guessing a message count.
  const std::string marker = OFFLINE_PROFILE_DONE_MARKER;
  MPI_Send(marker.data(), static_cast<int>(marker.size()), MPI_CHAR,
           master_rank, OFFLINE_PROFILE_DATA_TAG, comm_daemon);
  for (PendingOfflineProfileSend &send : pending_sends) {
    MPI_Wait(&send.request, MPI_STATUS_IGNORE);
  }
  pending_sends.clear();
}

static void parseOfflineKernelReqBatch(
    const std::string &received_data, int local_pid, int daemon_rank,
    int &daemon_wait_count, std::vector<S2DKernelReqData> &kernel_req_datas) {
  std::istringstream stream(received_data);
  std::string line;

  // **注意** 在最前面加上daemon_wait_count
  std::getline(stream, line);
  daemon_wait_count = std::stoi(line);
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }

    std::string obj_data = line + "\n";  // pid line
    std::getline(stream, line);
    obj_data += line + "\n";  // kernel_count line
    std::getline(stream, line);
    obj_data += line + "\n";  // kernel_identity line
    std::getline(stream, line);
    obj_data += line + "\n";  // req_size line
    std::getline(stream, line);
    obj_data += line + "\n";  // work_dim line
    std::getline(stream, line);
    obj_data += line + "\n";  // global_size0 line
    std::getline(stream, line);
    obj_data += line + "\n";  // global_size1 line
    std::getline(stream, line);
    obj_data += line + "\n";  // global_size2 line
    std::getline(stream, line);
    obj_data += line + "\n";  // req_count line
    int req_count = std::stoi(line);
    for (int i = 0; i < req_count * SYCL_REQ_DATA_SERIALIZED_LINES; ++i) {
      std::getline(stream, line);
      obj_data += line + "\n";
    }
    kernel_req_datas.push_back(S2DKernelReqData::deserialize(obj_data));
  }

  DAEMON_TRACE_STREAM << "Rank " << daemon_rank
            << ": mq_receive kernel_req_datas size: "
            << kernel_req_datas.size() << std::endl;

  for (const auto &kernel_req_data : kernel_req_datas) {
    for (const auto &req : kernel_req_data.reqs) {
      DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": mq_receive kernel_req_data pid: "
                << kernel_req_data.pid << " count: "
                << kernel_req_data.kernel_count
                << " req_count: " << req.req_count
                << " pointer: " << req.mem_pointer << std::endl;
      if (kernel_req_data.pid != local_pid) {
        std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) +
                               " kernel_req_data pid not match local_pid";
        perror(errorMsg.c_str());
        exit(1);
      }
    }
  }
}

static bool receiveOfflineKernelReqBatch(
    mqd_t mq_id_daemon, int local_pid, int daemon_rank, int &daemon_wait_count,
    std::vector<S2DKernelReqData> &kernel_req_datas,
    std::vector<S2DKernelProfileData> &local_profiles) {
  while (true) {
    DAEMON_TRACE_STREAM << "SystemSchedulerDaemonOffline: Rank " << daemon_rank
              << ": waiting reqs from handler" << std::endl;
    std::string received_data = receiveOfflineMqPayload(
        mq_id_daemon, MAX_MSG_DAEMON_SIZE, "offline kernel_req batch");
    if (received_data == "EXIT") {
      DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": SYCLAPP finish" << std::endl;
      return false;
    }

    if (S2DProfileBatchData::isProfileBatch(received_data)) {
      S2DProfileBatchData batch =
          S2DProfileBatchData::deserialize(received_data);
      local_profiles.insert(local_profiles.end(), batch.profiles.begin(),
                            batch.profiles.end());
      DAEMON_TRACE_STREAM << "Rank " << daemon_rank
                << ": mq_receive profile samples: "
                << batch.profiles.size() << std::endl;
      return true;
    }

    parseOfflineKernelReqBatch(received_data, local_pid, daemon_rank,
                               daemon_wait_count, kernel_req_datas);
    return true;
  }
}

void *SystemSchedulerDaemonOffline(void *arg) {
  int syclapp_count = *(int *)arg;
  int local_pid = globalcount_to_pid[syclapp_count];
  
  ProgramInfo &program_info = pid_to_program[local_pid];
  mqd_t mq_id_daemon = EstablishDaemon(local_pid);
  MPI_Comm &comm_daemon = program_info.comm_daemon;
  int &daemon_rank = program_info.daemon_rank;
  int &daemon_size = program_info.daemon_size;
  int &master_rank = program_info.master_rank;

  // std::map<int, int> exec_kernel_device; // kernel_count -> device_index in monitor_info

  //【scale】此daemon由master扩容来
  {
    // master中存在初始化为0的syclapp_count 非master存在scalecount的syclapp_count
    if (globalcount_to_scalecount.find(syclapp_count) == globalcount_to_scalecount.end()) {
      std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " globalcount_to_scalecount not found";
      perror(errorMsg.c_str());
      exit(1);
    }
    int scale_count = globalcount_to_scalecount[syclapp_count];
    
    // 与syclapp的programmanager通信 告知是否是扩容 使第一个handler走扩容分支
    char MESSAGE_QUEUE_PROGRAM_NAME[MESSAGE_QUEUE_PROGRAM_NAME_MAX];
    sprintf(MESSAGE_QUEUE_PROGRAM_NAME, MESSAGE_QUEUE_PROGRAM_PATTERN, local_pid);

    mqd_t mq_id_program = -1;
    while (mq_id_program == -1) {
      mq_id_program = mq_open(MESSAGE_QUEUE_PROGRAM_NAME, O_WRONLY);
      usleep(100000);
      DAEMON_TRACE_STREAM << "waiting" << std::endl;
    }
    DAEMON_TRACE_STREAM << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " opened mq_id_program: " << MESSAGE_QUEUE_PROGRAM_NAME << std::endl;

    // 因可能会在第一个scalecount扩容导致必须在syclapp最开始同步
    // TODO 在这里同步初始化代价
    std::string serialized_data = std::to_string(scale_count);
    size_t message_size = serialized_data.size();
    int ret = mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
    DAEMON_TRACE_STREAM << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " send to ProgramManager scale_count: " << scale_count << " ret: " << ret << std::endl;

    // struct mq_attr check_attr;
    // mq_getattr(mq_id_program, &check_attr);
    // std::cout << "[DM Debug] mq_curmsgs = " << check_attr.mq_curmsgs << ", mq_msgsize = " << check_attr.mq_msgsize << std::endl;

    // 用scalecount是否为默认值0区分master和非master扩容
    if (scale_count > 0) {
      //【与online不同】等待master传递D2D信息
      // 解释: online记录scalecount 扩容的daemon必定要执行
      //   offline的daemon执行一组kernel中的部分 必须额外传输
      DAEMON_TRACE_STREAM << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " with ScaleCount " << scale_count << " started." << std::endl;
      std::vector<D2DKernelSchedInfo> kernel_sched_order_infos;
      SendD2DKernelSchedInfos(comm_daemon, master_rank, daemon_rank, globalcount_to_onrun[syclapp_count], kernel_sched_order_infos);
      DAEMON_TRACE_STREAM << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " received D2DKernelSchedInfos size: " << kernel_sched_order_infos.size() << std::endl;

      // 【以下和一般流程相同】
      // 通过D2D解析出D2S
      // 返回D2S给handler
      // 满足依赖
      commExecInfo(kernel_sched_order_infos, local_pid, daemon_rank, comm_daemon);
    }
  }

  // 维护一个SYCLAPP的所有kernel的依赖关系 DAG相关
  std::vector<DAGNode *> kernel_dag_nodes; // 所有kernel对应的DAG
  std::list<PendingOfflineProfileSend> pending_profile_sends;
  std::set<int> finished_profile_ranks;
  DAEMON_TRACE_STREAM << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " started." << std::endl;

  //【通用情况】
  while (1) {
    // DONE ====【接收program通信】
    std::vector<S2DKernelReqData> kernel_req_datas;
    int daemon_wait_count = 0;
    std::vector<S2DKernelProfileData> local_profiles;
    if (!receiveOfflineKernelReqBatch(mq_id_daemon, local_pid, daemon_rank,
                                      daemon_wait_count, kernel_req_datas,
                                      local_profiles)) {
      flushOfflineProfilesOnExit(
          comm_daemon, daemon_rank, master_rank,
          globalcount_to_onrun[syclapp_count], pending_profile_sends,
          finished_profile_ranks);
      break;
    }

    // DELE 找出所有空闲rank供算法选择
    // std::vector<int> idle_ranks;
    std::set<int> &onrun_ranks = globalcount_to_onrun[syclapp_count];
    // for (int i = 0; i < ranks_idle.size(); i++) {
    //   if (ranks_idle[i] && onrun_ranks.find(i) == onrun_ranks.end()) {
    //     idle_ranks.push_back(i);
    //   }
    // }
    int onrun_size = onrun_ranks.size();

    // Profile publication is deliberately not a collective. A rank that
    // finishes early must remain able to receive/schedule useful work instead
    // of waiting for the slowest rank merely to make a performance sample
    // globally visible.
    publishAndDrainOfflineProfiles(local_profiles, comm_daemon, daemon_rank,
                                   master_rank, pending_profile_sends,
                                   finished_profile_ranks);

    if (kernel_req_datas.empty()) {
      DAEMON_TRACE_STREAM
          << "SystemSchedulerDaemonOffline: Rank " << daemon_rank
          << " processed profile-only batch" << std::endl;
      continue;
    }

    // ====【调度决策并发给其他rank】
    std::vector<D2DKernelSchedInfo> kernel_sched_order_infos;
    std::vector<int> scale_ranks;
    bool completion_window_handled = false;
    bool completion_exit_requested = false;
    bool completion_protocol_failed = false;
    {
      // 算法计算适合的rank数 以及每个kernel的执行顺序和device 需要同时考虑每个rank的device空闲
      if (daemon_rank == master_rank) {
        // 1. 构建DAG 确定依赖
        // A local wait fences every device queue used by the single-rank SNMD
        // path. Multi-rank execution needs explicit remote completion
        // acknowledgements before the same rebase is safe.
        if (onrun_size == 1) {
          rebaseCompletedOfflineDAG(kernel_dag_nodes);
        }
        std::vector<DAGNode *> nodes; // 所有kernel对应的DAG
        for (S2DKernelReqData & kernel_req_data : kernel_req_datas) {
          DAGNode *node = new DAGNode(kernel_req_data);
          // DAEMON_TRACE_STREAM << "Rank " << daemon_rank << ": generate DAGNode for kernel_count: " << node->kernel_count << std::endl;
          nodes.push_back(node);
        }
        generateDAGs(kernel_dag_nodes, nodes);

        // 2. 调度算法 更新node和sched_info
        const bool use_static_batch =
            algorithmHEFT(nodes, kernel_sched_order_infos);
        DAEMON_TRACE_STREAM << "Rank " << daemon_rank << " TEST kernel_sched_order_infos size: " << kernel_sched_order_infos.size() << std::endl;

#ifdef SNMD_OFFLINE_COMPLETION_DRIVEN_QUEUE
        if (daemon_size == 1 && onrun_size == 1 &&
            completionDrivenQueueRuntimeEnabled() &&
            !use_static_batch) {
          const CompletionWindowResult result = runCompletionDrivenWindow(
              nodes, daemon_wait_count, local_pid, mq_id_daemon);
          completion_window_handled =
              result == CompletionWindowResult::Completed;
          completion_exit_requested =
              result == CompletionWindowResult::ExitRequested;
          completion_protocol_failed =
              result == CompletionWindowResult::FailedAfterStart;
          if (completion_window_handled || completion_exit_requested) {
            kernel_sched_order_infos.clear();
          }
        }
#endif

        // TEST-START ===【固定测试】
        // globalDevices只取掉了加速器 0号是CPU
        // D2DKernelSchedInfo kernel_1;
        // kernel_1.kernel_count = 1;
        // kernel_1.exec_order = 1;
        // kernel_1.exec_rank = 1;
        // kernel_1.exec_device = 1;
        // kernel_sched_order_infos.push_back(kernel_1);
        // D2DKernelSchedInfo kernel_2;
        // kernel_2.kernel_count = 2;
        // kernel_2.exec_order = 2;
        // kernel_2.exec_rank = 1; // 从idle_ranks中选择 存到scale_ranks
        // kernel_2.exec_device = 1;
        // kernel_sched_order_infos.push_back(kernel_2);
        // D2DKernelSchedInfo kernel_3;
        // kernel_3.kernel_count = 3;
        // kernel_3.exec_order = 3;
        // kernel_3.exec_rank = 1;
        // kernel_3.exec_device = 1;
        // kernel_sched_order_infos.push_back(kernel_3);
        // D2DKernelSchedInfo kernel_4;
        // kernel_4.kernel_count = 4;
        // kernel_4.exec_order = 4;
        // kernel_4.exec_rank = 0;
        // kernel_4.exec_device = 1;
        // kernel_sched_order_infos.push_back(kernel_4);
        // TEST-END ===【固定测试】

        // 3. 处理扩容 从sched_info获取
        // TODO 扩容代价可以加入gpu_available_time
        // 需要通过实验确定规模初始化与时间的关系
        std::set<int> sched_ranks;
        for (D2DKernelSchedInfo &kernel_sched_info : kernel_sched_order_infos) {
          sched_ranks.insert(kernel_sched_info.exec_rank);
        }
        // scale_ranks是sched_ranks减去onrun_ranks
        std::set_difference(sched_ranks.begin(), sched_ranks.end(),
                            onrun_ranks.begin(), onrun_ranks.end(),
                            std::back_inserter(scale_ranks));
        DAEMON_TRACE_STREAM << "Rank " << daemon_rank << " scale_ranks: " << scale_ranks.size() << std::endl;

        for (int scale_rank : scale_ranks) {
          DAEMON_TRACE_STREAM << "Rank " << daemon_rank << " NEED SCALE rank: " << scale_rank << std::endl;
          // **注意** 在扩容逻辑中 一个scale_count可以扩容多个rank 目前没看到
          std::lock_guard<std::mutex> lock(*pid_to_scalecount_mutex[local_pid]);
          pid_to_scalecount_queue[local_pid]->push(std::make_pair(daemon_wait_count, scale_rank)); // (scale_count, rank) 这里不是kerne_count 是wait_count了
          pid_to_scalecount_cv[local_pid]->notify_one();
          // **注意** 整个syclapp的第一个kernel一定会在master上执行 因其他rank会有初始化代价 前提是master不由其他syclapp占用
        }

        // DELE ==== 算法中已填充node.exec_rank和kernel_sched_info.req_rank
        // // 1. 填充node的exec_rank 紧接req_rank要用
        // for (DAGNode *node : nodes) {
        //   // 在kernel_sched_order_infos中找到对应的kernel_sched_info
        //   auto it = std::find_if(kernel_sched_order_infos.begin(), kernel_sched_order_infos.end(),
        //     [node](const D2DKernelSchedInfo &info) { return info.kernel_count == node->kernel_count; });
        //   if (it != kernel_sched_order_infos.end()) {
        //     D2DKernelSchedInfo &kernel_sched_info = *it;
        //     node->exec_rank = kernel_sched_info.exec_rank;
        //   } else {
        //     std::cerr << "Error: Kernel " << node->kernel_count << " not found in kernel_sched_order_infos." << std::endl;
        //   }
        // }
        // // 2. 填充每个kernel的req_rank
        // // online中 一个req只找一个最近写作为依赖 但一个kernel可能不同req导致依赖多个前置kernel
        // // OPTI 先不做online的优化 只找依赖中的最近写 直接得出req_rank
        // for (DAGNode *node : nodes) {
        //   std::map<SyclReqData, int> req_rank;
        //   auto it = std::find_if(kernel_sched_order_infos.begin(), kernel_sched_order_infos.end(),
        //     [node](const D2DKernelSchedInfo &info) { return info.kernel_count == node->kernel_count; });
        //   if (it != kernel_sched_order_infos.end()) {
        //     D2DKernelSchedInfo &kernel_sched_info = *it;
        //     // OPTI 又跑了一遍generateDAGs的逻辑 太重复
        //     for (SyclReqData &req : node->req_data) {
        //       if (req.req_accmode == acc_mode::read || req.req_accmode == acc_mode::read_write || req.req_accmode == acc_mode::atomic) {
        //         bool found = false;
        //         for (DAGNode *prev_node : node->depend_on) {
        //           for (SyclReqData &prev_req : prev_node->req_data) {
        //             if (prev_req.req_accmode != acc_mode::read && prev_req.mem_pointer == req.mem_pointer) {
        //               req_rank[req] = prev_node->exec_rank;
        //               found = true;
        //               break;
        //             }
        //           }
        //           if (found) {
        //             break;
        //           }
        //         }
        //       }
        //     }
        //     kernel_sched_info.req_rank = req_rank;
        //   } else {
        //     std::cerr << "Error: Kernel " << node->kernel_count << " not found in kernel_sched_order_infos." << std::endl;
        //   }
        // }
        // std::sort(kernel_sched_order_infos.begin(), kernel_sched_order_infos.end());
      }

      if (completion_exit_requested) {
        flushOfflineProfilesOnExit(
            comm_daemon, daemon_rank, master_rank,
            globalcount_to_onrun[syclapp_count], pending_profile_sends,
            finished_profile_ranks);
        mq_close(mq_id_daemon);
        return NULL;
      }
      if (completion_protocol_failed) {
        std::cerr << "CompletionQueue: terminating scheduler thread after "
                     "an in-window protocol failure"
                  << std::endl;
        mq_close(mq_id_daemon);
        return NULL;
      }
      if (completion_window_handled) {
        continue;
      }

      // DELE 这里写错了 scale_ranks填充逻辑没写
      // scale_ranks.push_back(0);
      int scale_size = scale_ranks.size();
      // ====【扩容需要的rank数】
      // 选不在onrun中最空闲的
      // OPTI 优化空间 根据可能的空闲时间来安排rank
      // if (daemon_rank == master_rank) {
      //   if (scale_ranks.size() > 0) {
      //     // 通知scale
      //     for (int scale_rank : scale_ranks) {
      //       std::lock_guard<std::mutex> lock(*pid_to_scalecount_mutex[local_pid]);
      //       pid_to_scalecount_queue[local_pid]->push(std::make_pair(0, scale_rank));
      //       pid_to_scalecount_cv[local_pid]->notify_one();
      //       std::cout << "Rank " << daemon_rank << " NOTIFY SCALE rank: " << scale_rank << std::endl;
      //     }
      //   }
      // }

      // ====【调度发送给所有daemon】
      // **注意** 先扩容后通信 master将调度决策发送给所有daemon 包括scale
      while (globalcount_to_onrun[syclapp_count].size() < onrun_size + scale_size) {
        usleep(100000);
      }
      DAEMON_TRACE_STREAM << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " waiting new daemon, onrun size: " << globalcount_to_onrun[syclapp_count].size() << std::endl;

      SendD2DKernelSchedInfos(comm_daemon, master_rank, daemon_rank, globalcount_to_onrun[syclapp_count], kernel_sched_order_infos);

      DAEMON_TRACE_STREAM << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " onrun size: " << globalcount_to_onrun[syclapp_count].size() << " sent D2DKernelSchedInfos size: " << kernel_sched_order_infos.size() << std::endl;
    }

    commExecInfo(kernel_sched_order_infos, local_pid, daemon_rank, comm_daemon);
  }

  mq_close(mq_id_daemon);
  return NULL;
}

void *SystemSchedulerScale(void *arg) {
  int syclapp_count = *(int *)arg;
  const std::vector<std::string> &submit_args =
      globalcount_to_submit_args[syclapp_count];
  const char *binary_path = submit_args[0].c_str();

  // 选择一个master负责此SYCLAPP
  int master_rank = syclapp_count % submit_size;
  // 所有rank负责此SYCLAPP的scale都会参与新的通信域 color为global_syclapp_count
  MPI_Comm comm_syclapp;
  MPI_Comm_split(comm_submit, syclapp_count, submit_rank, &comm_syclapp);
  int syclapp_rank, syclapp_size;
  MPI_Comm_rank(comm_syclapp, &syclapp_rank);
  MPI_Comm_size(comm_syclapp, &syclapp_size);

  MPI_Comm comm_daemon;
  MPI_Comm_split(comm_submit, syclapp_count, submit_rank, &comm_daemon);
  int daemon_rank, daemon_size;
  MPI_Comm_rank(comm_daemon, &daemon_rank);
  MPI_Comm_size(comm_daemon, &daemon_size);
  DAEMON_TRACE_STREAM << "SystemSchedulerScale: SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << " SYCLAPP_Size " << syclapp_size << " DAEMON_Rank " << daemon_rank << " DAEMON_Size " << daemon_size << std::endl;

  // 始终由master开始
  if (syclapp_rank == master_rank) {
    // master记录目前参与计算的rank
    std::set<int> onrun_ranks = {master_rank};
    globalcount_to_onrun.insert(std::pair<int, std::set<int>>(syclapp_count, onrun_ranks));
    // **注意** 这边0和1是初始化一个不会出现的数字
    // online里扩容必然不是从1开始 而offline中可以 且要考虑扩容多rank
    // globalcount_to_scalecount.insert(std::pair<int, int>(syclapp_count, 1));
    globalcount_to_scalecount.insert(std::pair<int, int>(syclapp_count, 0)); // offline用

    // 必须要pid 不能像singlenode只监听接收 创建对应SYCLAPP的Daemon 创建mq
    ProgramInfo program_info;
    pid_t pid = fork();
    if (pid == 0) { // 子进程
      std::vector<char *> exec_argv = BuildExecArgv(submit_args);
      execv(binary_path, exec_argv.data());
      std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SYCLAPP_Rank " + std::to_string(syclapp_rank) + " Failed to execute binary";
      perror(errorMsg.c_str());
      exit(1);
    } else if (pid > 0) { // 父进程
      program_info.pid = pid;
      DAEMON_TRACE_STREAM << "SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << ": Launched binary with PID " << pid << std::endl;
    } else {
      std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SYCLAPP_Rank " + std::to_string(syclapp_rank) + " Fork failed";
      perror(errorMsg.c_str());
      exit(1);
    }
    globalcount_to_pid.insert(std::pair<int, pid_t>(syclapp_count, pid));
    pid_to_scalecount_queue[pid] = std::make_shared<std::queue<std::pair<int, int>>>();
    pid_to_scalecount_mutex[pid] = std::make_shared<std::mutex>();
    pid_to_scalecount_cv[pid] = std::make_shared<std::condition_variable>();

    program_info.global_syclapp_count = syclapp_count;
    program_info.set_mpi(comm_daemon, daemon_rank, daemon_size, master_rank);
    pid_to_program.insert(std::pair<pid_t, ProgramInfo>(pid, program_info));

    // **注意** 每个rank的pid不同 但global_syclapp_count相同
    pthread_t daemon_tid;
    pthread_create(&daemon_tid, NULL, (void *(*)(void *))SystemSchedulerDaemonOffline, &syclapp_count);
    // 不等待 不影响主线程 无需cancel和join
    pthread_detach(daemon_tid);


    // master等待daemon的信号需要扩容
    while (1) {
      DAEMON_TRACE_STREAM << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " waiting DAEMON NOTIFY" << std::endl;
      auto queue = pid_to_scalecount_queue[pid];
      auto mutex = pid_to_scalecount_mutex[pid];
      auto cv = pid_to_scalecount_cv[pid];
      std::unique_lock<std::mutex> lock(*mutex);
      cv->wait(lock, [&queue] { return !queue->empty(); });
      while (!queue->empty()) {
        auto scale_pair = queue->front();
        queue->pop();
        DAEMON_TRACE_STREAM << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " scale_count: " << scale_pair.first << " rank: " << scale_pair.second << std::endl;
      
        // DISCARD 不应该在Scale选择扩容的rank 否则要多一次向Daemon通信rank
        // int scale_rank;
        // for (int i = 0; i < submit_size; i++) {
        //   if (globalcount_to_onrun[syclapp_count].find(i) == globalcount_to_onrun[syclapp_count].end()) {
        //     scale_rank = i;
        //     break;
        //   }
        // }
        globalcount_to_onrun[syclapp_count].insert(scale_pair.second);

        MPI_Send(&scale_pair.first, 1, MPI_INT, scale_pair.second, 0, comm_syclapp);
        DAEMON_TRACE_STREAM << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " scale_count: " << scale_pair.first << " sent to rank " << scale_pair.second << std::endl;
      }
    }
  }
  // 非master等待master的扩容请求
  else {
    while (1) {
      DAEMON_TRACE_STREAM << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " waiting scale_count" << std::endl;
      int scale_count;
      MPI_Recv(&scale_count, 1, MPI_INT, master_rank, 0, comm_syclapp, MPI_STATUS_IGNORE);
      DAEMON_TRACE_STREAM << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " scale_count: " << scale_count << " received from rank " << master_rank << std::endl;

      // 非master从master通信接受scalecount 写入daemon全局可见数组中
      globalcount_to_scalecount.insert(std::pair<int, int>(syclapp_count, scale_count));
      
      ProgramInfo program_info;
      pid_t pid = fork();
      if (pid == 0) { // 子进程
        std::vector<char *> exec_argv = BuildExecArgv(submit_args);
        execv(binary_path, exec_argv.data());
        std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SYCLAPP_Rank " + std::to_string(syclapp_rank) + " Failed to execute binary";
        perror(errorMsg.c_str());
        exit(1);
      } else if (pid > 0) { // 父进程
        program_info.pid = pid;
        DAEMON_TRACE_STREAM << "SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << ": Launched binary with PID " << pid << std::endl;
      } else {
        std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SYCLAPP_Rank " + std::to_string(syclapp_rank) + " Fork failed";
        perror(errorMsg.c_str());
        exit(1);
      }
      globalcount_to_pid.insert(std::pair<int, pid_t>(syclapp_count, pid));
      program_info.global_syclapp_count = syclapp_count;
      program_info.set_mpi(comm_daemon, daemon_rank, daemon_size, master_rank);
      pid_to_program.insert(std::pair<pid_t, ProgramInfo>(program_info.pid, program_info));

      pthread_t daemon_tid;
      pthread_create(&daemon_tid, NULL, (void *(*)(void *))SystemSchedulerDaemonOffline, &syclapp_count);
      pthread_detach(daemon_tid);
    }
  }

  return NULL;
}

void *SystemSchedulerSubmit(void *arg) {
  MPI_Comm_rank(comm_submit, &submit_rank);
  MPI_Comm_size(comm_submit, &submit_size);
  DAEMON_TRACE_STREAM << "SystemSchedulerSubmit: SUBMIT_Rank " << submit_rank << " started." << std::endl;

  // while(1)用户向rank0提交bin_dir
  // 与管理单节点内的SystemSchedulerDaemon是两个不同的pthread
  while (1) {
    char submit_payload[MAX_MSG_SUBMIT_SIZE] = {};
    int submit_payload_size = 0;

    // rank0会阻塞在此等待
    if (submit_rank == 0) {
      ssize_t bytes_received = mq_receive(mq_id_submit, submit_payload, MAX_MSG_SUBMIT_SIZE, NULL);
      if (bytes_received == -1) {
        std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SUBMIT mq_receive failed";
        perror(errorMsg.c_str());
        exit(1);
      }
      submit_payload_size = static_cast<int>(bytes_received);
    }

    // 非rank0会阻塞在此等待
    MPI_Bcast(&submit_payload_size, 1, MPI_INT, 0, comm_submit);
    if (submit_payload_size <= 0 ||
        submit_payload_size > MAX_MSG_SUBMIT_SIZE) {
      std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " invalid submit payload size: " + std::to_string(submit_payload_size);
      std::cerr << errorMsg << std::endl;
      exit(1);
    }
    MPI_Bcast(submit_payload, submit_payload_size, MPI_CHAR, 0, comm_submit);

    std::vector<std::string> submit_args;
    try {
      submit_args = DeserializeSubmitArgs(submit_payload,
                                          submit_payload_size);
    } catch (const std::exception &e) {
      std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " invalid submit payload: " + e.what();
      std::cerr << errorMsg << std::endl;
      exit(1);
    }
    DAEMON_TRACE_STREAM << "SUBMIT_Rank " << submit_rank
                        << ": Received submit command: "
                        << JoinSubmitArgs(submit_args) << std::endl;

    global_syclapp_count++;
    globalcount_to_submit_args.insert(std::pair<int, std::vector<std::string>>(global_syclapp_count, submit_args));

    pthread_t scale_tid;
    pthread_create(&scale_tid, NULL, (void *(*)(void *))SystemSchedulerScale, &global_syclapp_count);
    // 不等待 不影响主线程 无需cancel和join
    pthread_detach(scale_tid);
  }

  return NULL;
}

int main(int argc, char *argv[]) {
  // ====【signal】
  // signal(SIGINT, SignalHandler);
  // sigset_t set;
  // sigemptyset(&set);
  // sigaddset(&set, SIGINT);
  // pthread_sigmask(SIG_BLOCK, &set, NULL);

  // ====【MPI】
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided < MPI_THREAD_MULTIPLE) {
      perror("Error: MPI NO SUPPORT MPI_THREAD_MULTIPLE");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
  }
  char proc_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Get_processor_name(proc_name, &name_len);
  DAEMON_TRACE_STREAM << "MPI_Rank " << mpi_rank << ": " << proc_name << " of " << mpi_size << " started" << std::endl;
  // Device identity must be stable before persisted profiles are loaded.
  // Otherwise CUDA visibility masks can remap multiple physical GPUs to the
  // same logical ordinal and merge incompatible service-time histories.
  int monitor_device_count = MonitorInit();
  initializePersistentProfileStore();
  // split_key==mpi_rank 所以local_rank和mpi_rank相同 在线程中仍可以使用mpi_rank和mpi_size
  MPI_Comm_split(MPI_COMM_WORLD, 0, mpi_rank, &comm_submit);
  MPI_Comm_split(MPI_COMM_WORLD, 0, mpi_rank, &comm_monitor);

  // ====【mq】
  EstablishSubmit();
  DAEMON_TRACE_STREAM << "MPI_Rank " << mpi_rank << ": Established" << std::endl;

  // ====【pthread】
  pthread_t monitor_tid, submit_tid;
  pthread_create(&monitor_tid, NULL, (void *(*)(void *))SystemSchedulerMonitor,
                 &monitor_device_count);
  pthread_create(&submit_tid, NULL, (void *(*)(void *))SystemSchedulerSubmit, NULL);
  DAEMON_TRACE_STREAM << "MPI_Rank " << mpi_rank << ": SystemSchedulerSubmit started" << std::endl;

  // ====【signal】
  // while (!is_interrupted) {
  //   pause();
  // }
  // int signum;
  // while (!is_interrupted) {
  //   sigwait(&set, &signum);
  //   if (signum == SIGINT) {
  //     std::cout << "Interrupted by SIGINT!" << std::endl;
  //     is_interrupted = true;
  //   }
  // }
  while (1) {
    if (is_interrupted) {
      break;
    }
    usleep(10000);
  }

  // ====【pthread】
  // pthread_cancel(monitor_tid);
  pthread_cancel(submit_tid);
  // pthread_join(monitor_tid, NULL);
  pthread_join(submit_tid, NULL);

  // ====【mq】
  CloseSubmit();

  // ====【MPI】
  MPI_Comm_free(&comm_submit);
  shutdownPersistentProfileStore();
  MPI_Finalize();

  return 0;
}
