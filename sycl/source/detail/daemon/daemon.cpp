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
std::map<int, int> index_sycl_nvml; // 根据busid确定sycl::device到gpu映射
std::map<int, int> index_nvml_sycl;
std::vector<MonitorInfo> device_monitor_info(1); // 每个设备的监控信息, 0号设备固定是CPU
std::vector<int> ranks_idle;

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

// INPUT: 已有的kernel组成的DAG 新的一组wait中所有的kernel
// 仅通过req的mem依赖建立DAG 不涉及req_rank和exec_rank
// 分析见NOTION
void generateDAGs(std::vector<DAGNode *> &kernel_dag_nodes, std::vector<DAGNode *> &nodes) {
  for (DAGNode *node : nodes) {
    for (SyclReqData &req : node->req_data) {
      // 依赖前序kernel相同mem的写 read | read_write | atomic
      if (req.req_accmode == acc_mode::read || req.req_accmode == acc_mode::read_write || req.req_accmode == acc_mode::atomic) {
        // 由后向前遍历kernel 找到相同mem最近的写作为依赖 // 只找最近的写是合理的
        bool found_write = false;
        for (auto it = kernel_dag_nodes.rbegin(); it != kernel_dag_nodes.rend(); ++it) {
          DAGNode *prev_node = *it;
          for (SyclReqData &prev_req : prev_node->req_data) {
            if (prev_req.mem_pointer == req.mem_pointer) {
              if (prev_req.req_accmode != acc_mode::read) {
                node->depend_on.push_back(prev_node);
                prev_node->depend_by.push_back(node);
                node->depth = std::max(node->depth, prev_node->depth + 1);
                found_write = true;
                // 所有的req_ranks都由调度结束确定exec_rank后生成
                // kernel间通信代价 即DAG边的权重由算法预估
              }
            }
          }
          if (found_write) {
            break;
          }
        }
      }
    }
    kernel_dag_nodes.push_back(node);
  }
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

  size_t mem_available;
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
    // **注意** 获取device::name必不可少 不然无法切换cuda上下文
    device.get_info<sycl::info::device::name>();
    int busId = getCudaPciBusId(device);
    std::cout << "SYCL Device " << i << ": PCI Bus ID: " << busId << std::endl;

    if (busId != -1) {
      auto it = std::find(nvmlBusIds.begin(), nvmlBusIds.end(), busId);
      if (it != nvmlBusIds.end()) {
        index_sycl_nvml[i] = std::distance(nvmlBusIds.begin(), it) + 1;
        index_nvml_sycl[index_sycl_nvml[i]] = i;
      }
    }
  }
  for (auto pair : index_sycl_nvml) {
    std::cout << "SYCL Device " << pair.first << " mapped to GPU " << pair.second << std::endl;
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

      device_monitor_info[i + 1] = MonitorInfo{name, utilization.gpu, memoryInfo.free / 1024.0};
  }
}

void *SystemMonitor(void *arg) {
  int device_count = MonitorInit();
  if (device_count != -1) {
    device_monitor_info.resize(device_count + 1);
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
    std::cout << "commExecInfo === Rank " << daemon_rank << ": mq_send kernel_exec_infos size: " << kernel_exec_infos.size() << " mqsize: " << message_size  << std::endl;
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
  // TODO 完全没写扩容判断逻辑
  {
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

    std::string serialized_data = std::to_string(scale_count);
    size_t message_size = serialized_data.size();
    int ret = mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
    std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << " for PID " << local_pid << " send to ProgramManager scale_count: " << scale_count << " ret: " << ret << std::endl;

    // struct mq_attr check_attr;
    // mq_getattr(mq_id_program, &check_attr);
    // std::cout << "[DM Debug] mq_curmsgs = " << check_attr.mq_curmsgs << ", mq_msgsize = " << check_attr.mq_msgsize << std::endl;

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
    {
      std::cout << "SystemSchedulerDaemonOffline: Rank " << daemon_rank << ": waiting reqs from handler" << std::endl;
      char buffer[MAX_MSG_DAEMON_SIZE];
      ssize_t bytes_received = mq_receive(mq_id_daemon, buffer, MAX_MSG_DAEMON_SIZE, nullptr);

      if (bytes_received > 0) {
        if (std::string(buffer, bytes_received) == "EXIT") {
          std::cout << "Rank " << daemon_rank << ": SYCLAPP finish" << std::endl;
          break;
        }

        std::string received_data(buffer, bytes_received);
        std::istringstream stream(received_data);
        std::string line;

        while (std::getline(stream, line)) {
          std::string obj_data = line + "\n";  // pid line
          std::getline(stream, line);
          obj_data += line + "\n";  // kernel_count line
          int kernel_count = std::stoi(line);
          std::getline(stream, line);
          obj_data += line + "\n";  // req_size line
          std::getline(stream, line);
          obj_data += line + "\n";  // req_count line
          int req_count = std::stoi(line);
          // read req_count * 6 lines
          for (int i = 0; i < req_count * 6; ++i) {
            std::getline(stream, line);
            obj_data += line + "\n";
          }
          // std::cout << " one obj_data " << std::endl;
          kernel_req_datas.push_back(S2DKernelReqData::deserialize(obj_data));
        }

        std::cout << "Rank " << daemon_rank << ": mq_receive kernel_req_datas size: " << kernel_req_datas.size() << std::endl;

        for (const auto &kernel_req_data : kernel_req_datas) {
          for (const auto &req : kernel_req_data.reqs) {
            std::cout << "Rank " << daemon_rank << ": mq_receive kernel_req_data pid: " 
                      << kernel_req_data.pid << " count: " << kernel_req_data.kernel_count
                      << " req_count: " << req.req_count << " pointer: " << req.mem_pointer << std::endl;
            if (kernel_req_data.pid != local_pid) {
              std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " kernel_req_data pid not match local_pid";
              perror(errorMsg.c_str());
              exit(1);
            }
          }
        }
      } else {
        std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " DAEMON mq_receive failed";
        perror(errorMsg.c_str());
        exit(1);
      }
    }

    // 找出所有空闲rank供算法选择
    std::vector<int> idle_ranks;
    std::set<int> &onrun_ranks = globalcount_to_onrun[syclapp_count];
    for (int i = 0; i < ranks_idle.size(); i++) {
      if (ranks_idle[i] && onrun_ranks.find(i) == onrun_ranks.end()) {
        idle_ranks.push_back(i);
      }
    }
    int onrun_size = onrun_ranks.size();

    // ====【调度决策并发给其他rank】
    // int scale_num = 0; // 需要scale的数量
    std::vector<int> scale_ranks;
    int scale_size = 0;
    std::vector<D2DKernelSchedInfo> kernel_sched_order_infos;
    // std::vector<D2DKernelSchedInfo> kernel_sched_infos;
    // std::vector<D2DKernelSchedInfo> kernel_sched_order_infos = kernel_sched_infos;
    {
      // TODO 算法计算适合的rank数 以及每个kernel的执行顺序和device 需要同时考虑每个rank的device空闲
      if (daemon_rank == master_rank) {
        // 构建DAG 确定依赖
        std::vector<DAGNode *> nodes; // 所有kernel对应的DAG
        for (S2DKernelReqData & kernel_req_data : kernel_req_datas) {
          DAGNode *node = new DAGNode(kernel_req_data.kernel_count, kernel_req_data.reqs);
          std::cout << "Rank " << daemon_rank << ": generate DAGNode for kernel_count: " << node->kernel_count << std::endl;
          nodes.push_back(node);
        }
        generateDAGs(kernel_dag_nodes, nodes);
        
        // TEST-START ===【固定测试】
        // globalDevices只取掉了加速器 0号是CPU
        D2DKernelSchedInfo kernel_1;
        kernel_1.kernel_count = 1;
        kernel_1.exec_order = 1;
        kernel_1.exec_rank = 1;
        kernel_1.exec_device = 1;
        kernel_sched_order_infos.push_back(kernel_1);

        D2DKernelSchedInfo kernel_2;
        kernel_2.kernel_count = 2;
        kernel_2.exec_order = 2;
        kernel_2.exec_rank = 1; // 从idle_ranks中选择 存到scale_ranks
        kernel_2.exec_device = 1;
        kernel_sched_order_infos.push_back(kernel_2);
        
        D2DKernelSchedInfo kernel_3;
        kernel_3.kernel_count = 3;
        kernel_3.exec_order = 3;
        kernel_3.exec_rank = 1;
        kernel_3.exec_device = 1;
        kernel_sched_order_infos.push_back(kernel_3);

        D2DKernelSchedInfo kernel_4;
        kernel_4.kernel_count = 4;
        kernel_4.exec_order = 4;
        kernel_4.exec_rank = 0;
        kernel_4.exec_device = 1;
        kernel_sched_order_infos.push_back(kernel_4);

        // TODO 这里知道要扩容了 开始扩容
        std::lock_guard<std::mutex> lock(*pid_to_scalecount_mutex[local_pid]);
        pid_to_scalecount_queue[local_pid]->push(std::make_pair(1, 0)); // (scale_count, rank) 这里不是kerne_count 是wait_count了
        // TODO 第一个就不在master上执行如何解决 先不考虑后起的需要满足以来
        pid_to_scalecount_cv[local_pid]->notify_one();

        std::cout << "Rank " << daemon_rank << " TEST kernel_sched_order_infos size: " << kernel_sched_order_infos.size() << std::endl;
        // TEST-END ===【固定测试】

        // 1. 填充node的exec_rank 紧接req_rank要用
        for (DAGNode *node : nodes) {
          // 在kernel_sched_order_infos中找到对应的kernel_sched_info
          auto it = std::find_if(kernel_sched_order_infos.begin(), kernel_sched_order_infos.end(),
            [node](const D2DKernelSchedInfo &info) { return info.kernel_count == node->kernel_count; });
          if (it != kernel_sched_order_infos.end()) {
            D2DKernelSchedInfo &kernel_sched_info = *it;
            
            node->exec_rank = kernel_sched_info.exec_rank;
          } else {
            std::cerr << "Error: Kernel " << node->kernel_count << " not found in kernel_sched_order_infos." << std::endl;
          }
        }

        // 2. 填充每个kernel的req_rank
        // online中 一个req只找一个最近写作为依赖 但一个kernel可能不同req导致依赖多个前置kernel
        // OPTI 先不做online的优化 只找依赖中的最近写 直接得出req_rank
        for (DAGNode *node : nodes) {
          std::map<SyclReqData, int> req_rank;
          auto it = std::find_if(kernel_sched_order_infos.begin(), kernel_sched_order_infos.end(),
            [node](const D2DKernelSchedInfo &info) { return info.kernel_count == node->kernel_count; });
          if (it != kernel_sched_order_infos.end()) {
            D2DKernelSchedInfo &kernel_sched_info = *it;

            // OPTI 又跑了一遍generateDAGs的逻辑 太重复
            for (SyclReqData &req : node->req_data) {
              if (req.req_accmode == acc_mode::read || req.req_accmode == acc_mode::read_write || req.req_accmode == acc_mode::atomic) {
                bool found = false;
                for (DAGNode *prev_node : node->depend_on) {
                  for (SyclReqData &prev_req : prev_node->req_data) {
                    if (prev_req.req_accmode != acc_mode::read && prev_req.mem_pointer == req.mem_pointer) {
                      req_rank[req] = prev_node->exec_rank;
                      found = true;
                      break;
                    }
                  }
                  if (found) {
                    break;
                  }
                }
              }
            }
            kernel_sched_info.req_rank = req_rank;
          } else {
            std::cerr << "Error: Kernel " << node->kernel_count << " not found in kernel_sched_order_infos." << std::endl;
          }
        }

        std::sort(kernel_sched_order_infos.begin(), kernel_sched_order_infos.end());
      }
      scale_ranks.push_back(0);
      scale_size = scale_ranks.size();

      // TODO 这里写错了 scale_ranks填充逻辑没写
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
