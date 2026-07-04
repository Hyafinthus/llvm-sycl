#include <iostream>
#include <fstream>
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
#include <cctype>
#include <cmath>
#include <limits>
#include <tuple>
#include <unordered_set>
#include <cuda_runtime_api.h>
#include <nvml.h>

#include "daemon.hpp"
#include <sycl/device.hpp>
// #include <sycl/access/access.hpp>

volatile bool is_interrupted = false;

// signal pthread MPI mq shmem

// 一个SYCLAPP的全局信息
int global_syclapp_count = 0; // 对于整个集群的SYCLAPP计数 因都会在Submit的Bcast前阻塞 每个节点的计数保持相等
std::map<int, std::string> globalcount_to_binpath; // SYCLAPP计数_binpath
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
std::vector<MonitorInfo> device_monitor_info(1); // 每个设备的监控信息, 0号设备固定是CPU
std::vector<ComputeCapability> device_capability(1); // 本rank设备原始能力, 编号与handler的globalDevices一致
std::vector<std::vector<MonitorInfo>> cluster_monitor_info;
std::vector<std::vector<ComputeCapability>> cluster_device_capability;
std::vector<int> ranks_idle;

// ====【Algorithm】
std::vector<std::vector<ComputeCapability>> gpu_capability; // 不同rank的设备算力, 包含fp32/fp64两套归一化能力
std::vector<std::vector<double>> gpu_available_time; // 不同gpu的可用时间 monitor查看空闲？不空闲怎么做

struct ProfileCostKey {
  std::string kernel_key;
  int rank = 0;
  int device = 0;
  int num_parts = 1;

  bool operator<(const ProfileCostKey &other) const {
    return std::tie(kernel_key, rank, device, num_parts) <
           std::tie(other.kernel_key, other.rank, other.device,
                    other.num_parts);
  }
};

struct ProfileCostEntry {
  double ewma_cost = 0.0;
  int samples = 0;
};

std::map<ProfileCostKey, ProfileCostEntry> profile_cost_table;

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

  std::cout << "EstablishDaemon: Rank " << mpi_rank << " created mq_id_daemon: " << mq_id_daemon << std::endl;

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

void SendD2DKernelSchedInfo(MPI_Comm comm_daemon, int master_rank, int daemon_rank, const std::set<int>& onrun_ranks, D2DKernelSchedInfo& kernel_sched_info) {
  if (daemon_rank == master_rank) {
    std::string serialized_data = kernel_sched_info.serialize();
    int str_length = static_cast<int>(serialized_data.size());
    for (int rank : onrun_ranks) {
      if (rank != master_rank) {
        MPI_Send(&str_length, 1, MPI_INT, rank, 0, comm_daemon);
        std::cout << "SendD2DKernelSchedInfo: Rank " << daemon_rank << " to Rank " << rank << " with str_length:" << str_length << std::endl;
        MPI_Send(serialized_data.c_str(), str_length, MPI_CHAR, rank, 0, comm_daemon);
        std::cout << "SendD2DKernelSchedInfo: Rank " << daemon_rank << " to Rank " << rank << " with serialized_data" << std::endl;
      }
    }
  } else {
    int str_length;
    MPI_Recv(&str_length, 1, MPI_INT, master_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
    std::cout << "SendD2DKernelSchedInfo: Rank " << daemon_rank << " received str_length:" << str_length << std::endl;

    char* buffer = new char[str_length + 1];
    MPI_Recv(buffer, str_length, MPI_CHAR, master_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
    buffer[str_length] = '\0';

    std::string serialized_data(buffer);
    kernel_sched_info = D2DKernelSchedInfo::deserialize(serialized_data);
    std::cout << "SendD2DKernelSchedInfo: Rank " << daemon_rank << " received serialized_data" << std::endl;

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
        std::cout << "SendD2DKernelSchedInfos: Rank " << daemon_rank << " to Rank " << rank << " with str_length:" << str_length << std::endl;

        MPI_Send(serialized_data.c_str(), str_length, MPI_CHAR, rank, 0, comm_daemon);
        std::cout << "SendD2DKernelSchedInfos: Rank " << daemon_rank << " to Rank " << rank << " with serialized_data" << std::endl;
      }
    }
  } else {
    int str_length;
    MPI_Recv(&str_length, 1, MPI_INT, master_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
    std::cout << "SendD2DKernelSchedInfos: Rank " << daemon_rank << " received str_length:" << str_length << std::endl;

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
        for (int j = 0; j < 6; ++j) {
          std::getline(stream, line);
          obj_data += line + "\n";  // SyclReqData (6 lines)
        }
        std::getline(stream, line);
        obj_data += line + "\n";  // int: rank
      }

      received_sched_infos.push_back(D2DKernelSchedInfo::deserialize(obj_data));
    }

    std::cout << "SendD2DKernelSchedInfos: Rank " << daemon_rank << " received " << received_sched_infos.size() << " kernel_sched_order_infos" << std::endl;
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

static double getCommElem(DAGNode *node, DAGNode *pre_node) {
  auto it = node->comm_elem.find(pre_node);
  return it == node->comm_elem.end() ? 0.0 : it->second;
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
      std::cout << "regenerateReqRanksAfterHEFT: Kernel "
                << node->kernel_count << " req " << req.req_count
                << " source rank " << producer->exec_rank
                << " from Kernel " << producer->kernel_count << std::endl;
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

        if (prev_node_added) {
          std::cout << "generateDAGs: Kernel " << node->kernel_count
                    << " depends on Kernel " << prev_node->kernel_count
                    << " for req " << req.req_count << std::endl;
        }

        if (prev_node_has_writer) {
          found_prev_writer = true;
          break;
        }
      }
    }
    kernel_dag_nodes.push_back(node);
  }
}

static constexpr double PROFILE_NS_TO_COST = 10000.0;
static constexpr double SAME_RANK_BANDWIDTH = 100.0;
static constexpr double CROSS_RANK_BANDWIDTH = 1.0;
static constexpr double SPLIT_EFFICIENCY = 0.85;
static constexpr double SPLIT_MIN_ELEMS = 65536.0;

struct TaskCandidate {
  int rank = -1;
  int proc = -1;
  int num_parts = 1;
  double start_time = 0.0;
  double finish_time = std::numeric_limits<double>::infinity();
  double exec_cost = std::numeric_limits<double>::infinity();
  std::vector<int> occupied_procs;
};

static std::string profileKeyForNode(const DAGNode *node) {
  return buildKernelProfileKey(node->req_data);
}

static double reqBytes(const SyclReqData &req) {
  return static_cast<double>(req.elem_size) * static_cast<double>(req.buff_size);
}

static double totalReqElems(const DAGNode *node) {
  double elems = 0.0;
  for (const SyclReqData &req : node->req_data) {
    elems += req.buff_size;
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
      elems += req.buff_size;
    }
  }
  return elems;
}

static double totalWriteElems(const DAGNode *node) {
  double elems = 0.0;
  for (const SyclReqData &req : node->req_data) {
    if (isWriteAccess(req.req_accmode)) {
      elems += req.buff_size;
    }
  }
  return elems;
}

static void updateProfileCostTable(const S2DKernelProfileData &profile,
                                   int sample_rank) {
  if (profile.duration_ns == 0 || profile.kernel_key.empty()) {
    return;
  }

  ProfileCostKey key{profile.kernel_key, sample_rank, profile.device_index,
                     std::max(1, profile.num_parts)};
  const double sample_cost =
      static_cast<double>(profile.duration_ns) / PROFILE_NS_TO_COST;
  ProfileCostEntry &entry = profile_cost_table[key];
  if (entry.samples == 0) {
    entry.ewma_cost = sample_cost;
  } else {
    entry.ewma_cost = entry.ewma_cost * 0.7 + sample_cost * 0.3;
  }
  entry.samples++;

  std::cout << "ProfileCostTable: key " << profile.kernel_key
            << " rank " << sample_rank << " device " << profile.device_index
            << " parts " << std::max(1, profile.num_parts)
            << " sample_cost " << sample_cost
            << " ewma_cost " << entry.ewma_cost
            << " samples " << entry.samples << std::endl;
}

static bool lookupProfileCost(const std::string &kernel_key, int rank,
                              int device, int num_parts, double &cost) {
  ProfileCostKey exact{kernel_key, rank, device, std::max(1, num_parts)};
  auto exact_it = profile_cost_table.find(exact);
  if (exact_it != profile_cost_table.end()) {
    cost = exact_it->second.ewma_cost;
    return true;
  }

  double sum = 0.0;
  int samples = 0;
  for (const auto &entry : profile_cost_table) {
    if (entry.first.kernel_key == kernel_key &&
        entry.first.num_parts == std::max(1, num_parts)) {
      sum += entry.second.ewma_cost * entry.second.samples;
      samples += entry.second.samples;
    }
  }
  if (samples > 0) {
    cost = sum / samples;
    return true;
  }

  return false;
}

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
    return ComputeCapability{3.2, 1.6};
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
    return ComputeCapability{91.0, 1.42};
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

static double initialAvailableTimeFromUtil(double util, double capability) {
  util = std::max(0.0, std::min(100.0, util));
  if (util < MONITOR_THRESHOLD) {
    return 0.0;
  }

  const double busy_ratio =
      (util - MONITOR_THRESHOLD) / (100.0 - MONITOR_THRESHOLD);
  return busy_ratio * 1000.0 / std::max(0.1, capability);
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

static void ensureOfflineDeviceModel() {
  const size_t rank_count = offlineRankCount();

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

  for (size_t rank = 0; rank < rank_count; ++rank) {
    gpu_capability[rank].resize(raw_capability[rank].size());
    gpu_available_time[rank].resize(raw_capability[rank].size());

    for (size_t proc = 0; proc < raw_capability[rank].size(); ++proc) {
      gpu_capability[rank][proc] = ComputeCapability{
          std::max(0.1, raw_capability[rank][proc].fp32 / min_capability),
          std::max(0.1, raw_capability[rank][proc].fp64 / min_capability)};

      const MonitorInfo *info =
          monitorInfoForDevice(static_cast<int>(rank), static_cast<int>(proc));
      const double util = info == nullptr ? 0.0 : info->util_used;
      const double available_capability =
          std::max(gpu_capability[rank][proc].fp32,
                   gpu_capability[rank][proc].fp64);
      gpu_available_time[rank][proc] =
          initialAvailableTimeFromUtil(util, available_capability);

      std::cout << "ensureOfflineDeviceModel: Rank " << rank
                << " Proc " << proc
                << " FP32Capability " << gpu_capability[rank][proc].fp32
                << " FP64Capability " << gpu_capability[rank][proc].fp64
                << " Util " << util
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
      fp32_elems += req.buff_size;
    } else if (req.elem_size == 8) {
      fp64_elems += req.buff_size;
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

static double monitorPenalty(int rank, int proc) {
  const MonitorInfo *info = monitorInfoForDevice(rank, proc);
  if (info == nullptr) {
    return 1.0;
  }
  const double util = std::max(0.0, std::min(100.0, info->util_used));
  return 1.0 + util / 100.0;
}

static bool monitorMemoryFits(const DAGNode *node, int rank, int proc,
                              int num_parts) {
  const MonitorInfo *info = monitorInfoForDevice(rank, proc);
  if (info == nullptr || info->mem_available == 0) {
    return true;
  }

  double required_bytes = totalReqBytes(node);
  if (num_parts > 1 && !node->req_data.empty()) {
    required_bytes =
        (totalReadElems(node) + totalWriteElems(node) / num_parts) *
        node->req_data.front().elem_size;
  }
  const double available_bytes = static_cast<double>(info->mem_available) * 1024.0;
  return required_bytes < available_bytes * 0.85;
}

static double estimateSingleExecCost(DAGNode *node, int rank, int proc) {
  double profile_cost = 0.0;
  const std::string key = profileKeyForNode(node);
  if (lookupProfileCost(key, rank, proc, 1, profile_cost)) {
    return std::max(0.001, profile_cost) * monitorPenalty(rank, proc);
  }

  const KernelPrecision precision = inferKernelPrecisionFromReqs(node->req_data);
  const double cold_cost = std::max(1.0, node->total_elem) /
                           std::max(0.1, deviceCapability(rank, proc,
                                                          precision));
  return cold_cost * monitorPenalty(rank, proc);
}

static double estimateCommCost(DAGNode *node, DAGNode *pre_node, int rank,
                               int proc, int num_parts) {
  if (pre_node->exec_rank < 0) {
    return 0.0;
  }

  const double comm_elem = getCommElem(node, pre_node);
  if (comm_elem == 0.0) {
    return 0.0;
  }

  if (pre_node->exec_rank != rank) {
    return comm_elem / CROSS_RANK_BANDWIDTH;
  }

  if (pre_node->num_parts > 1 || num_parts > 1 || pre_node->exec_proc != proc) {
    return comm_elem / SAME_RANK_BANDWIDTH;
  }

  return 0.0;
}

static double dependencyReadyTime(DAGNode *node, int rank, int proc,
                                  int num_parts) {
  double ready_time = 0.0;
  for (DAGNode *pre_node : node->depend_on) {
    ready_time = std::max(
        ready_time,
        pre_node->finish_time +
            estimateCommCost(node, pre_node, rank, proc, num_parts));
  }
  return ready_time;
}

static bool worthConsideringSplit(DAGNode *node, int num_parts) {
  if (num_parts <= 1) {
    return false;
  }
  if (totalWriteElems(node) == 0.0) {
    return false;
  }
  if (totalReqElems(node) < SPLIT_MIN_ELEMS) {
    return false;
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

static double estimateSplitExecCost(DAGNode *node, int rank,
                                    const std::vector<int> &split_devices) {
  const int num_parts = static_cast<int>(split_devices.size());
  if (num_parts <= 1) {
    return estimateSingleExecCost(
        node, rank, split_devices.empty() ? 1 : split_devices.front());
  }

  double profile_cost = 0.0;
  const std::string key = profileKeyForNode(node);
  if (lookupProfileCost(key, rank, split_devices.front(), num_parts,
                        profile_cost)) {
    double penalty = 1.0;
    for (int proc : split_devices) {
      penalty = std::max(penalty, monitorPenalty(rank, proc));
    }
    return std::max(0.001, profile_cost) * penalty;
  }

  double best_single = std::numeric_limits<double>::infinity();
  for (int proc : split_devices) {
    best_single = std::min(best_single, estimateSingleExecCost(node, rank, proc));
  }

  if (!std::isfinite(best_single)) {
    best_single = estimateSingleExecCost(node, rank, 1);
  }

  const double copy_overhead =
      (totalReadElems(node) * (num_parts - 1) + totalWriteElems(node)) /
      1000.0 / SAME_RANK_BANDWIDTH;
  const double launch_overhead = 0.2 * num_parts;
  return best_single / (num_parts * SPLIT_EFFICIENCY) + copy_overhead +
         launch_overhead;
}

static TaskCandidate makeSingleCandidate(DAGNode *node, int rank, int proc) {
  TaskCandidate candidate;
  candidate.rank = rank;
  candidate.proc = proc;
  candidate.num_parts = 1;
  candidate.occupied_procs.push_back(proc);

  if (!monitorMemoryFits(node, rank, proc, 1)) {
    return candidate;
  }

  const double device_ready = gpu_available_time[rank][proc];
  const double dep_ready = dependencyReadyTime(node, rank, proc, 1);
  candidate.start_time = std::max(device_ready, dep_ready);
  candidate.exec_cost = estimateSingleExecCost(node, rank, proc);
  candidate.finish_time = candidate.start_time + candidate.exec_cost;
  return candidate;
}

static TaskCandidate makeSplitCandidate(DAGNode *node, int rank,
                                        int num_parts) {
  TaskCandidate candidate;
  candidate.rank = rank;
  candidate.num_parts = num_parts;

  if (!worthConsideringSplit(node, num_parts)) {
    return candidate;
  }
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

    const double dep_ready =
        dependencyReadyTime(node, rank, split_devices.front(), num_parts);
    const double start_time = std::max(device_ready, dep_ready);
    const double exec_cost = estimateSplitExecCost(node, rank, split_devices);
    const double finish_time = start_time + exec_cost;
    if (finish_time < candidate.finish_time) {
      candidate.proc = split_devices.front();
      candidate.occupied_procs = split_devices;
      candidate.start_time = start_time;
      candidate.exec_cost = exec_cost;
      candidate.finish_time = finish_time;
    }
  }
  return candidate;
}

static double estimateAverageRankCost(DAGNode *node) {
  double sum = 0.0;
  int count = 0;
  for (int rank = 0; rank < static_cast<int>(gpu_available_time.size()); ++rank) {
    for (int proc = 0; proc < static_cast<int>(gpu_available_time[rank].size());
         ++proc) {
      if (monitorMemoryFits(node, rank, proc, 1)) {
        sum += estimateSingleExecCost(node, rank, proc);
        count++;
      }
    }
  }
  if (count == 0) {
    return std::max(1.0, node->total_elem);
  }
  return sum / count;
}

// nodes: 这批要调度的所有kernel
void algorithmHEFT(std::vector<DAGNode *> &nodes, std::vector<D2DKernelSchedInfo> &kernel_sched_order_infos) {
  // 根据当前monitor得到每个rank的设备数量、归一化算力和初始可用时间。
  // 0号设备固定是CPU，后续编号保持和handler端globalDevices一致。
  ensureOfflineDeviceModel();

  // 1.根据(规模+monitor)生成执行时间表 对于每个任务t_i计算w(i) 以及任务在不同proc上时各个pre的传输代价
  for (DAGNode *node : nodes) {
    // 1.1. 计算每个任务的平均计算时间
    double total_elem = 0;
    for (const SyclReqData &req : node->req_data) {
      total_elem += req.buff_size; // buff_size就是总数据量 不需要除以elem_size
    }
    node->total_elem = total_elem / 1000; // TODO 归一化
    std::cout << "algorithmHEFT: Kernel " << node->kernel_count
              << " total_elem: " << total_elem
              << " precision: "
              << precisionName(inferKernelPrecisionFromReqs(node->req_data))
              << std::endl;

    // 1.2. 计算每个任务需要从各个前序接收多少数据
    for (std::pair<DAGNode *, std::set<SyclReqData>> dep_pair : node->depend_on_mem) {
      DAGNode *pre_node = dep_pair.first;
      int pre_comm_elem = 0;
      for (const SyclReqData &req : dep_pair.second) {
        pre_comm_elem += req.buff_size;
      }
      node->comm_elem[pre_node] = pre_comm_elem / 1000;
      std::cout << "algorithmHEFT: Kernel " << node->kernel_count << " depends on Kernel " << pre_node->kernel_count << " pre_comm_elem: " << pre_comm_elem << std::endl;
    }
  }

  // 2.从后向前 对于每个任务u->v 计算rank(u)=w_mean(u)+max_v[c(u,v)+rank(v)] 并从大到小排序
  // 必须使用逆拓扑序，保证计算一个节点时同批次所有后继已经计算过rank_u。
  std::unordered_set<DAGNode *> current_nodes(nodes.begin(), nodes.end());
  std::vector<DAGNode *> visited = reverseTopologicalOrder(nodes);
  for (DAGNode *node : visited) {
    double max_succ = 0;
    for (DAGNode *succ_node : node->depend_by) {
      if (!current_nodes.count(succ_node)) {
        continue;
      }
      const double succ_cost = getCommElem(succ_node, node) + succ_node->rank_u;
      if (succ_cost > max_succ) {
        max_succ = succ_cost;
      }
    }
    node->rank_u = estimateAverageRankCost(node) + max_succ;
    if (max_succ == 0) {
      std::cout << "algorithmHEFT: Kernel " << node->kernel_count
                << " is exit node rank_u: " << node->rank_u << std::endl;
    }
  }
  // 以rank_u从大到小排序
  std::sort(visited.begin(), visited.end(), [](DAGNode *a, DAGNode *b) {
    return a->rank_u > b->rank_u;
  });

  // 3.每个任务计算 对于每个proc 计算start_v(p)=max_[last_finish(p),finish(u_1)+comm(u_1,v),...]
  // 和finish_v(p)=start_v(p)+w(v,p)
  // 同时把SNMD split作为候选放置方式，选择earliest_finish_p(v)最小者。
  for (int order = 0; order < visited.size(); order++) {
    DAGNode *node = visited[order];
    TaskCandidate best_candidate;

    for (int rank = 0; rank < gpu_available_time.size(); rank++) {
      for (int proc = 0; proc < gpu_available_time[rank].size(); proc++) {
        TaskCandidate candidate = makeSingleCandidate(node, rank, proc);
        if (candidate.finish_time < best_candidate.finish_time) {
          best_candidate = candidate;
        }
      }

      const int max_split_parts =
          std::min<int>(4, static_cast<int>(gpu_available_time[rank].size()) - 1);
      for (int num_parts = 2; num_parts <= max_split_parts; ++num_parts) {
        TaskCandidate candidate = makeSplitCandidate(node, rank, num_parts);
        if (candidate.finish_time < best_candidate.finish_time) {
          best_candidate = candidate;
        }
      }
    }

    if (best_candidate.rank < 0 || best_candidate.proc < 0) {
      best_candidate = makeSingleCandidate(node, 0, 0);
    }

    node->exec_rank = best_candidate.rank;
    node->exec_proc = best_candidate.proc;
    node->num_parts = best_candidate.num_parts;
    node->split_devices = best_candidate.occupied_procs;
    node->finish_time = best_candidate.finish_time;
    for (int proc : best_candidate.occupied_procs) {
      gpu_available_time[node->exec_rank][proc] = node->finish_time;
    }

    D2DKernelSchedInfo kernel_sched_info;
    kernel_sched_info.kernel_count = node->kernel_count;
    kernel_sched_info.exec_order = order + 1;
    kernel_sched_info.exec_rank = node->exec_rank;
    kernel_sched_info.exec_device = node->exec_proc;
    kernel_sched_info.num_parts = node->num_parts;
    kernel_sched_info.split_devices = node->split_devices;
    kernel_sched_order_infos.push_back(kernel_sched_info);

    std::cout << "algorithmHEFT: Kernel " << node->kernel_count
              << " assigned to Rank " << node->exec_rank
              << " Proc " << node->exec_proc
              << " NumParts " << node->num_parts
              << " SplitDevices";
    for (int split_device : node->split_devices) {
      std::cout << " " << split_device;
    }
    std::cout
              << " start_time " << best_candidate.start_time
              << " exec_cost " << best_candidate.exec_cost
              << " finish_time " << node->finish_time << std::endl;
  }

  regenerateReqRanksAfterHEFT(nodes, kernel_sched_order_infos);

  // TODO 最合适用几个节点去跑
  // 通信代价和贪心避免了扩张代价大于运行代价

}

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
          std::sscanf(line.c_str(), "MemAvailable: %llu kB", &mem_available);
          break;
      }
  }
  file.close();
  // std::cout << "Memory available: " << mem_available << " kB" << std::endl;

  device_monitor_info[0] = MonitorInfo{"CPU", utilization, mem_available};
}

int getCudaPciBusId(const sycl::device &device) {
  if (device.get_backend() != sycl::backend::ext_oneapi_cuda) {
      return -1;
  }
  int cudaDevice;
  cudaError_t err = cudaGetDevice(&cudaDevice);
  if (err != cudaSuccess) {
      throw std::runtime_error("Failed to get current CUDA device.");
  }
  int busId;
  err = cudaDeviceGetAttribute(&busId, cudaDevAttrPciBusId, cudaDevice);
  if (err != cudaSuccess) {
      throw std::runtime_error("Failed to get PCI Bus ID for the CUDA device.");
  }
  return busId;
}

int MonitorInit() {
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
  std::cout << "Number of GPUs: " << device_count << std::endl;
  device_capability.resize(device_count + 1, fallbackCapabilityForProc(1));
  
  std::vector<int> nvmlBusIds;
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

    int nvmlBusId = std::stoi(std::string(pciInfo.busId).substr(9, 2), nullptr, 16);
    std::cout << "GPU " << i << ": PCI Bus ID: " << pciInfo.busId << " int: " << nvmlBusId << std::endl;
    nvmlBusIds.push_back(nvmlBusId);
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
    std::cout << "SYCL Device " << i << " (" << sycl_device_name
              << "): PCI Bus ID: " << busId << std::endl;

    if (busId != -1) {
      auto it = std::find(nvmlBusIds.begin(), nvmlBusIds.end(), busId);
      if (it != nvmlBusIds.end()) {
        index_sycl_nvml[i] = std::distance(nvmlBusIds.begin(), it) + 1;
        index_nvml_sycl[index_sycl_nvml[i]] = i;
        device_capability[i] =
            inferDeviceCapabilityFromName(sycl_device_name, false);
      }
    }
  }
  for (auto pair : index_sycl_nvml) {
    std::cout << "SYCL Device " << pair.first << " mapped to GPU "
              << pair.second << " fp32 capability "
              << device_capability[pair.first].fp32
              << " fp64 capability "
              << device_capability[pair.first].fp64 << std::endl;
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

      int device_index = i + 1;
      auto sycl_it = index_nvml_sycl.find(device_index);
      if (sycl_it != index_nvml_sycl.end()) {
        device_index = sycl_it->second;
      }
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

void *SystemMonitor(void *arg) {
  int device_count = MonitorInit();
  if (device_count != -1) {
    device_monitor_info.resize(device_count + 1);
    device_capability.resize(device_count + 1, fallbackCapabilityForProc(1));
  }
  
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
  std::cout << "SystemSchedulerMonitor: MONITOR_Rank " << monitor_rank << " started." << std::endl;

  pthread_t monitor_tid;
  pthread_create(&monitor_tid, NULL, (void *(*)(void *))SystemMonitor, NULL);
  pthread_detach(monitor_tid);

  ranks_idle.resize(monitor_size, false);
  cluster_monitor_info.resize(monitor_size);
  cluster_device_capability.resize(monitor_size);

  // 每个rank彼此感知是否有空闲即可 无需传递所有状态？
  while (1) {
    int is_idle = 0;
    for (int i = 1; i < device_monitor_info.size(); i++) {
      if (device_monitor_info[i].util_used < MONITOR_THRESHOLD) {
        is_idle = 1;
        break;
      }
    }

    MPI_Allgather(&is_idle, 1, MPI_INT, ranks_idle.data(), 1, MPI_INT, comm_monitor);

    std::array<double, MAX_MONITOR_DEVICES * MONITOR_PACKED_FIELDS> local_monitor{};
    const int local_device_count =
        std::min<int>(device_monitor_info.size(), MAX_MONITOR_DEVICES);
    for (int i = 0; i < local_device_count; ++i) {
      const int offset = i * MONITOR_PACKED_FIELDS;
      local_monitor[offset + 0] = 1.0;
      local_monitor[offset + 1] = device_monitor_info[i].util_used;
      local_monitor[offset + 2] =
          static_cast<double>(device_monitor_info[i].mem_available);
      const ComputeCapability capability =
          i < static_cast<int>(device_capability.size())
              ? device_capability[i]
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
      std::cout << "SystemSchedulerDaemon: Rank " << daemon_rank << " for PID " << local_pid << " with ScaleCount " << scale_count << " started." << std::endl;

      // 接收
      char buffer[MAX_MSG_DAEMON_SIZE];
      ssize_t bytes_received = mq_receive(mq_id_daemon, buffer, MAX_MSG_DAEMON_SIZE, nullptr);
      if (bytes_received > 0) {
        // std::string received_data(buffer, bytes_received);
        // S2DKernelReqData kernel_req_data = S2DKernelReqData::deserialize(received_data);
        std::cout << "SystemSchedulerDaemon: Rank " << daemon_rank << " for PID " << local_pid << " received first kernel" << std::endl;
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
        int min_util = 100;
        int min_util_index = -1;
        for (int i = 1; i < device_monitor_info.size(); i++) {
          if (device_monitor_info[i].util_used < min_util) {
            min_util = device_monitor_info[i].util_used;
            min_util_index = i;
          }
        }
        kernel_exec_info.device_index = min_util_index;
        exec_kernel_device[scale_count] = kernel_exec_info.device_index;
        std::string serialized_data = kernel_exec_info.serialize();
        size_t message_size = serialized_data.size();
        mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
        std::cout << "SystemSchedulerDaemon: Rank " << daemon_rank << " for PID " << local_pid << " sent first kernel" << std::endl;
      }
    } else {
      std::cout << "SystemSchedulerDaemon: Rank " << daemon_rank << " for PID " << local_pid << " started." << std::endl;
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
            std::cout << "Rank " << daemon_rank << ": Scale mq_receive kernel_req_data pid: " << kernel_req_data.pid << " count: " << kernel_req_data.kernel_count << " req_count: " << req.req_count << " pointer: " << req.mem_pointer << std::endl;
          }
        } else {
          std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " DAEMON mq_receive failed";
          perror(errorMsg.c_str());
          exit(1);
        }
        std::cout << "Rank " << daemon_rank << ": Scale mq_receive kernel_req_data pid: " << kernel_req_data.pid << " count: " << kernel_req_data.kernel_count << std::endl;
      }

      {
        for (SyclReqData &req : kernel_req_data.reqs) {
          if (req.req_accmode == acc_mode::read || req.req_accmode == acc_mode::read_write || req.req_accmode == acc_mode::atomic) {
            int elem_size = req.elem_size;
            int buff_size = req.buff_size;
            int data_rank = master_rank;

            std::vector<DATA_TYPE> host_data(elem_size * buff_size);
            MPI_Recv(host_data.data(), elem_size * buff_size, MPI_BYTE, data_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
            std::cout << "Rank " << daemon_rank << ": Scale received data from rank " << data_rank << std::endl;

            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_req_data.kernel_count, req.req_count, elem_size * buff_size);
            writeToSharedMemory(handle, host_data.data(), elem_size * buff_size);
            std::cout << "Rank " << daemon_rank << ": Scale write to shared" << std::endl;
            waitForReadCompletion(handle);
            cleanupSharedMemory(handle, elem_size * buff_size);
            std::cout << "Rank " << daemon_rank << ": Scale waitForReadCompletion" << std::endl;
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
      std::cout << "Rank " << daemon_rank << ": waiting req from handler" << std::endl;
      char buffer[MAX_MSG_DAEMON_SIZE];
      ssize_t bytes_received = mq_receive(mq_id_daemon, buffer, MAX_MSG_DAEMON_SIZE, nullptr);
      if (bytes_received > 0) {
        if (std::string(buffer, bytes_received) == "EXIT") {
          std::cout << "Rank " << daemon_rank << ": SYCLAPP finish" << std::endl;
          break;
        }
        std::string received_data(buffer, bytes_received);
        kernel_req_data = S2DKernelReqData::deserialize(received_data);
        for (SyclReqData &req : kernel_req_data.reqs) {
          std::cout << "Rank " << daemon_rank << ": mq_receive kernel_req_data pid: " << kernel_req_data.pid << " count: " << kernel_req_data.kernel_count << " req_count: " << req.req_count << " pointer: " << req.mem_pointer << std::endl;
        }
      } else {
        std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " DAEMON mq_receive failed";
        perror(errorMsg.c_str());
        exit(1);
      }
      std::cout << "Rank " << daemon_rank << ": mq_receive kernel_req_data pid: " << kernel_req_data.pid << " count: " << kernel_req_data.kernel_count << std::endl;
    }

    // ====【调度决策并发给其他rank】
    D2DKernelSchedInfo kernel_sched_info;
    bool scale = false;
    {
      if (daemon_rank == master_rank) {
        // [1] [rank0] 构建DAG 确定依赖的kernel 查找依赖的kernel在哪个rank执行
        DAGNode *node = new DAGNode(kernel_req_data.kernel_count, kernel_req_data.reqs);
        std::map<SyclReqData, std::set<int>> req_ranks = generateDAG(kernel_dag_nodes, node);
        for (auto pair : req_ranks) {
          std::cout << "Rank " << daemon_rank << " req_rank: " << pair.first.kernel_count << "-" << pair.first.req_count << " pointer: " << pair.first.mem_pointer << " rank: ";
          for (int rank : pair.second) {
            std::cout << rank << " ";
          }
          std::cout << std::endl;
        }
        kernel_sched_info.kernel_count = kernel_req_data.kernel_count;
        // OPTI 2 [allrank] 调度决策(设备负载/性能预测/通信开销/计算开销)
        // 在master上只需要确认rank是否还有device空闲 具体的device应由各daemon监控和选择
        int most_rank = mostDepdRank(req_ranks);
        // 没依赖 master有空闲就执行 无空闲应该扩容
        // 不太可能没依赖 必然会依赖初始化的write 无空闲扩容的情况很少
        if (most_rank == -1) { 
          bool idle = false;
          for (int i = 1; i < device_monitor_info.size(); i++) {
            if (device_monitor_info[i].util_used < MONITOR_THRESHOLD) {
              idle = true;
              break;
            }
          }
          if (idle) {
            kernel_sched_info.exec_rank = daemon_rank;
            kernel_sched_info.exec_device = -1;
          } else {
            for (int i = 0; i < ranks_idle.size(); i++) {
              if (i == monitor_rank) {
                continue;
              }
              if (ranks_idle[i]) {
                kernel_sched_info.exec_rank = i;

                std::lock_guard<std::mutex> lock(*pid_to_scalecount_mutex[local_pid]);
                pid_to_scalecount_queue[local_pid]->push(std::make_pair(kernel_req_data.kernel_count, 0));
                pid_to_scalecount_cv[local_pid]->notify_one();

                scale = true;
                std::cout << "Rank " << daemon_rank << " NO DEPD NOTIFY SCALE rank: " << i << std::endl;
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
              for (int i = 1; i < device_monitor_info.size(); i++) {
                if (device_monitor_info[i].util_used < MONITOR_THRESHOLD) {
                  idle = true;
                  break;
                }
              }
              if (idle) { // master空闲
                kernel_sched_info.exec_rank = daemon_rank;
                kernel_sched_info.exec_device = -1;
              } else { // master不空闲 找空闲rank扩容
                for (int i = 0; i < ranks_idle.size(); i++) {
                  if (i == monitor_rank) {
                    continue;
                  }
                  if (ranks_idle[i]) {
                    kernel_sched_info.exec_rank = i;

                    std::lock_guard<std::mutex> lock(*pid_to_scalecount_mutex[local_pid]);
                    pid_to_scalecount_queue[local_pid]->push(std::make_pair(kernel_req_data.kernel_count, 0));
                    pid_to_scalecount_cv[local_pid]->notify_one();

                    scale = true;
                    std::cout << "Rank " << daemon_rank << " YES DEPD NOTIFY SCALE rank: " << i << std::endl;
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
      std::cout << "Rank " << daemon_rank << " kernel_sched_info.exec_rank: " << kernel_sched_info.exec_rank << std::endl;
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
          for (int i = 1; i < device_monitor_info.size(); i++) {
            if (device_monitor_info[i].util_used < MONITOR_THRESHOLD) {
              idle_devices.push_back(i);
            }
          }
          if (idle_devices.size() > 0) { // 随机选择
            int rand_index = rand() % idle_devices.size();
            kernel_exec_info.device_index = idle_devices[rand_index];
          } else { // 找利用率最低的
            int min_util = 100;
            int min_util_index = -1;
            for (int i = 1; i < device_monitor_info.size(); i++) {
              if (device_monitor_info[i].util_used < min_util) {
                min_util = device_monitor_info[i].util_used;
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
      std::cout << "Rank " << daemon_rank << ": mq_send kernel_exec_info exec: " << kernel_exec_info.exec << " req_counts.size: " << kernel_exec_info.req_counts.size() << " device_index: " << kernel_exec_info.device_index << std::endl;
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
          std::cout << "Rank " << daemon_rank << ": Scale data read successfully." << std::endl;
          cleanupSharedMemory(handle, elem_size * buff_size);

          MPI_Send(host_data.data(), elem_size * buff_size, MPI_BYTE, kernel_sched_info.exec_rank, 0, comm_daemon);
          std::cout << "Rank " << daemon_rank << ": Scale sent data to rank " << kernel_sched_info.exec_rank << std::endl;
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
        std::cout << "Rank " << daemon_rank << ": Exec Rank: " << kernel_sched_info.exec_rank << " need data from other" << std::endl;
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
            std::cout << "Rank " << daemon_rank << ": Data read successfully." << std::endl;
            cleanupSharedMemory(handle, elem_size * buff_size);

            // [7] [双rank][MPI] isend:host->buffer
            MPI_Send(host_data.data(), elem_size * buff_size, MPI_BYTE, kernel_sched_info.exec_rank, 0, comm_daemon);
            std::cout << "Rank " << daemon_rank << ": Sent data to rank " << kernel_sched_info.exec_rank << std::endl;
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
            std::cout << "Rank " << daemon_rank << ": Received data from rank " << data_rank << std::endl;

            // [8] 把从其他rank接受的data发给SYCL进程
            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_sched_info.kernel_count, req.req_count, elem_size * buff_size);
            writeToSharedMemory(handle, host_data.data(), elem_size * buff_size);
            std::cout << "Rank " << daemon_rank << ": Write to shared" << std::endl;
            waitForReadCompletion(handle);
            cleanupSharedMemory(handle, elem_size * buff_size);
            std::cout << "Rank " << daemon_rank << ": waitForReadCompletion" << std::endl;
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
    mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
    std::cout << "commExecInfo === Rank " << daemon_rank
              << ": mq_send kernel_exec_infos size: "
              << kernel_exec_infos.size() << " mqsize: " << message_size
              << std::endl;
    for (const D2SKernelExecInfo &kernel_exec_info : kernel_exec_infos) {
      std::cout << "commExecInfo === Rank " << daemon_rank
                << ": kernel_count " << kernel_exec_info.kernel_count
                << " exec " << kernel_exec_info.exec
                << " device_index " << kernel_exec_info.device_index
                << " num_parts " << kernel_exec_info.num_parts
                << " split_devices";
      for (int split_device : kernel_exec_info.split_devices) {
        std::cout << " " << split_device;
      }
      std::cout
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
    std::cout << "OfflineCommExecInfo: Rank " << daemon_rank << ": Kernel_order: " << order << " : Kernel_count: " << kernel_sched_info.kernel_count << std::endl;

    if (kernel_sched_info.req_rank.size() != kernel_sched_info.get_req_for_rank(kernel_sched_info.exec_rank).size()) {
      std::cout << "OfflineCommExecInfo: Rank " << daemon_rank << ": Exec Rank: " << kernel_sched_info.exec_rank << " need data from other" << std::endl;

      // 此rank不执行kernel 且kernel有依赖此rank的数据
      if (daemon_rank != kernel_sched_info.exec_rank && req_for_rank.size() > 0) {
        std::cout << "OfflineCommExecInfo: Rank " << daemon_rank << ": NO exec, provide" << std::endl;
        for (SyclReqData &req : req_for_rank) {
          // [6] 从SYCL进程接受host的data
          // 因为是写读共享内存是阻塞的 不需要等待SYCL进程的通知
          int elem_size = req.elem_size;
          int buff_size = req.buff_size;
          std::vector<DATA_TYPE> host_data(elem_size * buff_size);

          std::cout << "===SEND Rank " << daemon_rank << ": Req hostdata: " << elem_size << "*" << buff_size << std::endl;

          SharedMemoryHandle handle = initSharedMemory(local_pid, kernel_sched_info.kernel_count, req.req_count, elem_size * buff_size);
          std::cout << "Rank " << daemon_rank << ": initSharedMemory" << std::endl;

          readFromSharedMemory(handle, host_data.data(), elem_size * buff_size);
          std::cout << "Rank " << daemon_rank << ": Data read successfully." << std::endl;

          cleanupSharedMemory(handle, elem_size * buff_size);

          // [7] [双rank][MPI] isend:host->buffer
          MPI_Send(host_data.data(), elem_size * buff_size, MPI_BYTE, kernel_sched_info.exec_rank, 0, comm_daemon);
          std::cout << "Rank " << daemon_rank << ": Sent data to rank " << kernel_sched_info.exec_rank << std::endl;
        }
      }

      // 此rank执行kernel 且必然需要从其他rank拿数据
      if (daemon_rank == kernel_sched_info.exec_rank) {
        std::cout << "OfflineCommExecInfo: Rank " << daemon_rank << ": Exec, receive" << std::endl;
        for (SyclReqData &req : req_for_rank) {
          int elem_size = req.elem_size;
          int buff_size = req.buff_size;
          int data_rank = kernel_sched_info.req_rank[req];
          std::vector<DATA_TYPE> host_data(elem_size * buff_size);

          std::cout << "---RECV Rank " << daemon_rank << ": Req data from rank " << data_rank << " elem_size: " << elem_size << " buff_size: " << buff_size << std::endl;
          
          // [7] [双rank][MPI] irecv:buffer->host
          MPI_Recv(host_data.data(), elem_size * buff_size, MPI_BYTE, data_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
          std::cout << "Rank " << daemon_rank << ": Received data from rank " << data_rank << std::endl;

          // [8] 把从其他rank接受的data发给SYCL进程
          SharedMemoryHandle handle = initSharedMemory(local_pid, kernel_sched_info.kernel_count, req.req_count, elem_size * buff_size);
          writeToSharedMemory(handle, host_data.data(), elem_size * buff_size);
          std::cout << "Rank " << daemon_rank << ": Write to shared" << std::endl;

          waitForReadCompletion(handle);
          cleanupSharedMemory(handle, elem_size * buff_size);
          std::cout << "Rank " << daemon_rank << ": waitForReadCompletion" << std::endl;
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

static void exchangeOfflineProfilesWithMaster(
    const std::vector<S2DKernelProfileData> &local_profiles,
    MPI_Comm &comm_daemon, int daemon_rank, int master_rank,
    const std::set<int> &onrun_ranks) {
  static constexpr int PROFILE_LEN_TAG = 201;
  static constexpr int PROFILE_DATA_TAG = 202;

  std::string payload = serializeProfileSamples(local_profiles);
  int payload_size = static_cast<int>(payload.size());

  if (daemon_rank == master_rank) {
    for (const S2DKernelProfileData &profile : local_profiles) {
      updateProfileCostTable(profile, daemon_rank);
    }

    for (int rank : onrun_ranks) {
      if (rank == master_rank) {
        continue;
      }

      int remote_size = 0;
      MPI_Recv(&remote_size, 1, MPI_INT, rank, PROFILE_LEN_TAG, comm_daemon,
               MPI_STATUS_IGNORE);
      if (remote_size <= 0) {
        continue;
      }

      std::string remote_payload(remote_size, '\0');
      MPI_Recv(remote_payload.data(), remote_size, MPI_CHAR, rank,
               PROFILE_DATA_TAG, comm_daemon, MPI_STATUS_IGNORE);

      std::vector<S2DKernelProfileData> remote_profiles =
          deserializeProfileSamples(remote_payload);
      for (const S2DKernelProfileData &profile : remote_profiles) {
        updateProfileCostTable(profile, rank);
      }
    }
  } else {
    MPI_Send(&payload_size, 1, MPI_INT, master_rank, PROFILE_LEN_TAG,
             comm_daemon);
    if (payload_size > 0) {
      MPI_Send(payload.data(), payload_size, MPI_CHAR, master_rank,
               PROFILE_DATA_TAG, comm_daemon);
    }
  }
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
    obj_data += line + "\n";  // req_size line
    std::getline(stream, line);
    obj_data += line + "\n";  // req_count line
    int req_count = std::stoi(line);
    for (int i = 0; i < req_count * 6; ++i) {
      std::getline(stream, line);
      obj_data += line + "\n";
    }
    kernel_req_datas.push_back(S2DKernelReqData::deserialize(obj_data));
  }

  std::cout << "Rank " << daemon_rank
            << ": mq_receive kernel_req_datas size: "
            << kernel_req_datas.size() << std::endl;

  for (const auto &kernel_req_data : kernel_req_datas) {
    for (const auto &req : kernel_req_data.reqs) {
      std::cout << "Rank " << daemon_rank << ": mq_receive kernel_req_data pid: "
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
    std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank
              << ": waiting reqs from handler" << std::endl;
    char buffer[MAX_MSG_DAEMON_SIZE];
    ssize_t bytes_received =
        mq_receive(mq_id_daemon, buffer, MAX_MSG_DAEMON_SIZE, nullptr);

    if (bytes_received <= 0) {
      std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) +
                             " DAEMON mq_receive failed";
      perror(errorMsg.c_str());
      exit(1);
    }

    std::string received_data(buffer, bytes_received);
    if (received_data == "EXIT") {
      std::cout << "Rank " << daemon_rank << ": SYCLAPP finish" << std::endl;
      return false;
    }

    if (S2DProfileBatchData::isProfileBatch(received_data)) {
      S2DProfileBatchData batch =
          S2DProfileBatchData::deserialize(received_data);
      local_profiles.insert(local_profiles.end(), batch.profiles.begin(),
                            batch.profiles.end());
      std::cout << "Rank " << daemon_rank
                << ": mq_receive profile samples: "
                << batch.profiles.size() << std::endl;
      continue;
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
      std::cout << "waiting" << std::endl;
    }
    std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " opened mq_id_program: " << MESSAGE_QUEUE_PROGRAM_NAME << std::endl;

    // 因可能会在第一个scalecount扩容导致必须在syclapp最开始同步
    // TODO 在这里同步初始化代价
    std::string serialized_data = std::to_string(scale_count);
    size_t message_size = serialized_data.size();
    int ret = mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
    std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " send to ProgramManager scale_count: " << scale_count << " ret: " << ret << std::endl;

    // struct mq_attr check_attr;
    // mq_getattr(mq_id_program, &check_attr);
    // std::cout << "[DM Debug] mq_curmsgs = " << check_attr.mq_curmsgs << ", mq_msgsize = " << check_attr.mq_msgsize << std::endl;

    // 用scalecount是否为默认值0区分master和非master扩容
    if (scale_count > 0) {
      //【与online不同】等待master传递D2D信息
      // 解释: online记录scalecount 扩容的daemon必定要执行
      //   offline的daemon执行一组kernel中的部分 必须额外传输
      std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " with ScaleCount " << scale_count << " started." << std::endl;
      std::vector<D2DKernelSchedInfo> kernel_sched_order_infos;
      SendD2DKernelSchedInfos(comm_daemon, master_rank, daemon_rank, globalcount_to_onrun[syclapp_count], kernel_sched_order_infos);
      std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " received D2DKernelSchedInfos size: " << kernel_sched_order_infos.size() << std::endl;

      // 【以下和一般流程相同】
      // 通过D2D解析出D2S
      // 返回D2S给handler
      // 满足依赖
      commExecInfo(kernel_sched_order_infos, local_pid, daemon_rank, comm_daemon);
    }
  }

  // 维护一个SYCLAPP的所有kernel的依赖关系 DAG相关
  std::vector<DAGNode *> kernel_dag_nodes; // 所有kernel对应的DAG
  std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " started." << std::endl;

  //【通用情况】
  while (1) {
    // DONE ====【接收program通信】
    std::vector<S2DKernelReqData> kernel_req_datas;
    int daemon_wait_count = 0;
    std::vector<S2DKernelProfileData> local_profiles;
    if (!receiveOfflineKernelReqBatch(mq_id_daemon, local_pid, daemon_rank,
                                      daemon_wait_count, kernel_req_datas,
                                      local_profiles)) {
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

    exchangeOfflineProfilesWithMaster(local_profiles, comm_daemon, daemon_rank,
                                      master_rank, onrun_ranks);

    // ====【调度决策并发给其他rank】
    std::vector<D2DKernelSchedInfo> kernel_sched_order_infos;
    std::vector<int> scale_ranks;
    {
      // 算法计算适合的rank数 以及每个kernel的执行顺序和device 需要同时考虑每个rank的device空闲
      if (daemon_rank == master_rank) {
        // 1. 构建DAG 确定依赖
        std::vector<DAGNode *> nodes; // 所有kernel对应的DAG
        for (S2DKernelReqData & kernel_req_data : kernel_req_datas) {
          DAGNode *node = new DAGNode(kernel_req_data.kernel_count, kernel_req_data.reqs);
          std::cout << "Rank " << daemon_rank << ": generate DAGNode for kernel_count: " << node->kernel_count << std::endl;
          nodes.push_back(node);
        }
        generateDAGs(kernel_dag_nodes, nodes);

        // 2. 调度算法 更新node和sched_info
        algorithmHEFT(nodes, kernel_sched_order_infos);
        std::cout << "Rank " << daemon_rank << " TEST kernel_sched_order_infos size: " << kernel_sched_order_infos.size() << std::endl;

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
        std::cout << "Rank " << daemon_rank << " scale_ranks: " << scale_ranks.size() << std::endl;

        for (int scale_rank : scale_ranks) {
          std::cout << "Rank " << daemon_rank << " NEED SCALE rank: " << scale_rank << std::endl;
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
      std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " waiting new daemon, onrun size: " << globalcount_to_onrun[syclapp_count].size() << std::endl;

      SendD2DKernelSchedInfos(comm_daemon, master_rank, daemon_rank, globalcount_to_onrun[syclapp_count], kernel_sched_order_infos);

      std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " onrun size: " << globalcount_to_onrun[syclapp_count].size() << " sent D2DKernelSchedInfos size: " << kernel_sched_order_infos.size() << std::endl;
    }

    commExecInfo(kernel_sched_order_infos, local_pid, daemon_rank, comm_daemon);
  }

  mq_close(mq_id_daemon);
  return NULL;
}

void *SystemSchedulerScale(void *arg) {
  int syclapp_count = *(int *)arg;
  const char *binary_path = globalcount_to_binpath[syclapp_count].c_str();

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
  std::cout << "SystemSchedulerScale: SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << " SYCLAPP_Size " << syclapp_size << " DAEMON_Rank " << daemon_rank << " DAEMON_Size " << daemon_size << std::endl;

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
      execl(binary_path, binary_path, NULL);
      std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SYCLAPP_Rank " + std::to_string(syclapp_rank) + " Failed to execute binary";
      perror(errorMsg.c_str());
      exit(1);
    } else if (pid > 0) { // 父进程
      program_info.pid = pid;
      std::cout << "SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << ": Launched binary with PID " << pid << std::endl;
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
      std::cout << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " waiting DAEMON NOTIFY" << std::endl;
      auto queue = pid_to_scalecount_queue[pid];
      auto mutex = pid_to_scalecount_mutex[pid];
      auto cv = pid_to_scalecount_cv[pid];
      std::unique_lock<std::mutex> lock(*mutex);
      cv->wait(lock, [&queue] { return !queue->empty(); });
      while (!queue->empty()) {
        auto scale_pair = queue->front();
        queue->pop();
        std::cout << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " scale_count: " << scale_pair.first << " rank: " << scale_pair.second << std::endl;
      
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
        std::cout << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " scale_count: " << scale_pair.first << " sent to rank " << scale_pair.second << std::endl;
      }
    }
  }
  // 非master等待master的扩容请求
  else {
    while (1) {
      std::cout << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " waiting scale_count" << std::endl;
      int scale_count;
      MPI_Recv(&scale_count, 1, MPI_INT, master_rank, 0, comm_syclapp, MPI_STATUS_IGNORE);
      std::cout << "SystemSchedulerScale: SYCLAPP_Rank " << syclapp_rank << " scale_count: " << scale_count << " received from rank " << master_rank << std::endl;

      // 非master从master通信接受scalecount 写入daemon全局可见数组中
      globalcount_to_scalecount.insert(std::pair<int, int>(syclapp_count, scale_count));
      
      ProgramInfo program_info;
      pid_t pid = fork();
      if (pid == 0) { // 子进程
        execl(binary_path, binary_path, NULL);
        std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SYCLAPP_Rank " + std::to_string(syclapp_rank) + " Failed to execute binary";
        perror(errorMsg.c_str());
        exit(1);
      } else if (pid > 0) { // 父进程
        program_info.pid = pid;
        std::cout << "SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << ": Launched binary with PID " << pid << std::endl;
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
  std::cout << "SystemSchedulerSubmit: SUBMIT_Rank " << submit_rank << " started." << std::endl;

  // while(1)用户向rank0提交bin_dir
  // 与管理单节点内的SystemSchedulerDaemon是两个不同的pthread
  while (1) {
    char binary_path[MAX_MSG_SUBMIT_SIZE];

    // rank0会阻塞在此等待
    ssize_t bytes_received;
    if (submit_rank == 0) {
      bytes_received = mq_receive(mq_id_submit, binary_path, MAX_MSG_SUBMIT_SIZE, NULL);
      if (bytes_received == -1) {
        std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SUBMIT mq_receive failed";
        perror(errorMsg.c_str());
        exit(1);
      }
      std::cout << "SUBMIT_Rank " << submit_rank << ": Received submit path: " << binary_path << std::endl;
    }

    // 非rank0会阻塞在此等待
    MPI_Bcast(binary_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, 0, comm_submit);

    global_syclapp_count++;
    globalcount_to_binpath.insert(std::pair<int, std::string>(global_syclapp_count, std::string(binary_path)));

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
  std::cout << "MPI_Rank " << mpi_rank << ": " << proc_name << " of " << mpi_size << " started" << std::endl;
  // split_key==mpi_rank 所以local_rank和mpi_rank相同 在线程中仍可以使用mpi_rank和mpi_size
  MPI_Comm_split(MPI_COMM_WORLD, 0, mpi_rank, &comm_submit);
  MPI_Comm_split(MPI_COMM_WORLD, 0, mpi_rank, &comm_monitor);

  // ====【mq】
  EstablishSubmit();
  std::cout << "MPI_Rank " << mpi_rank << ": Established" << std::endl;

  // ====【pthread】
  pthread_t monitor_tid, submit_tid;
  pthread_create(&monitor_tid, NULL, (void *(*)(void *))SystemSchedulerMonitor, NULL);
  pthread_create(&submit_tid, NULL, (void *(*)(void *))SystemSchedulerSubmit, NULL);
  std::cout << "MPI_Rank " << mpi_rank << ": SystemSchedulerSubmit started" << std::endl;

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
  MPI_Finalize();

  return 0;
}
