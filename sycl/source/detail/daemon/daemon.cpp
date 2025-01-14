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
              req_ranks[req].insert(prev_node->exec_rank);
              found_write = true;
              break;
            }
            else { // 优化
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
  // DISCARD 【优化】写后被读的rank可以作为依赖
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

// TODO 根据数据量判断不同rank所需的总体通信量
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

// TODO 同一个req在多个rank上有最新时的选择方案 尽量集中还是分散？
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
        // ========【固定测试】
        // kernel_exec_info.exec = true;
        // kernel_exec_info.kernel_count = 1;
        // kernel_exec_info.device_index = 1; // scalecount也用这个device
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
        DAGNode *node = new DAGNode(kernel_req_data.reqs);
        std::map<SyclReqData, std::set<int>> req_ranks = generateDAG(kernel_dag_nodes, node);
        for (auto pair : req_ranks) {
          std::cout << "Rank " << daemon_rank << " req_rank: " << pair.first.kernel_count << "-" << pair.first.req_count << " pointer: " << pair.first.mem_pointer << " rank: ";
          for (int rank : pair.second) {
            std::cout << rank << " ";
          }
          std::cout << std::endl;
        }
        kernel_sched_info.kernel_count = kernel_req_data.kernel_count;
        // TODO 2 [allrank] 调度决策(设备负载/性能预测/通信开销/计算开销)
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
        // 依赖在master上就扩容 在其他rank就继续执行 TODO 未实现其他rank满足依赖
        else {
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
          }
        }
        kernel_sched_info.req_rank = chooseReqRank(req_ranks, daemon_rank);
        node->exec_rank = kernel_sched_info.exec_rank;

        // ========【固定测试】3mm
        // A dep no - rank1
        // B dep no - rank0
        // C dep A:rank1, B:rank0 - rank1
        // if (kernel_req_data.kernel_count == 1) {
        //   kernel_sched_info.exec_rank = 1;
        // } else if (kernel_req_data.kernel_count == 2) {
        //   // A
        //   kernel_sched_info.exec_rank = 1;
        // } else if (kernel_req_data.kernel_count == 3) {
        //   // B
        //   kernel_sched_info.exec_rank = 0;
        //   {
        //     std::lock_guard<std::mutex> lock(*pid_to_scalecount_mutex[local_pid]);
        //     pid_to_scalecount_queue[local_pid]->push(std::make_pair(kernel_req_data.kernel_count, 0));
        //     pid_to_scalecount_cv[local_pid]->notify_one();
        //   }
        //   scale = true;
        //   std::cout << "Rank " << daemon_rank << " NOTIFY SCALE" << std::endl;
        // } else if (kernel_req_data.kernel_count == 4) {
        //   // C
        //   kernel_sched_info.exec_rank = 1;
        //   kernel_sched_info.req_rank.insert({kernel_req_data.reqs[0], 1});
        //   kernel_sched_info.req_rank.insert({kernel_req_data.reqs[1], 0});
        // }
        // ========【固定测试】calc 在daemon决定执行rank 后续还要彼此通信 scale决定还要通知daemon
        // if (daemon_rank == 1) {
        //   if (kernel_req_data.kernel_count == 1) {
        //     kernel_sched_info.exec_rank = 1;
        //   } else if (kernel_req_data.kernel_count == 2) {
        //     kernel_sched_info.exec_rank = 1;
        //   } else if (kernel_req_data.kernel_count == 3) {
        //     kernel_sched_info.exec_rank = 1;
        //   } else if (kernel_req_data.kernel_count == 4) {
        //     kernel_sched_info.exec_rank = 1;
        //   } else if (kernel_req_data.kernel_count == 5) {
        //     kernel_sched_info.exec_rank = 0;
        //     {
        //       std::lock_guard<std::mutex> lock(*pid_to_scalecount_mutex[local_pid]);
        //       pid_to_scalecount_queue[local_pid]->push(std::make_pair(kernel_req_data.kernel_count, 0));
        //       pid_to_scalecount_cv[local_pid]->notify_one();
        //     }
        //     scale = true;
        //   } else if (kernel_req_data.kernel_count == 6) {
        //     kernel_sched_info.exec_rank = 1;
        //   } else if (kernel_req_data.kernel_count == 7) {
        //     kernel_sched_info.exec_rank = 1;
        //   } else if (kernel_req_data.kernel_count == 8) {
        //     kernel_sched_info.exec_rank = 1;
        //     kernel_sched_info.req_rank.insert({kernel_req_data.reqs[1], 0});
        //   } else if (kernel_req_data.kernel_count == 9) {
        //     kernel_sched_info.exec_rank = 1;
        //   }
        // }
      }

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
      else {
        req_for_rank = kernel_sched_info.get_req_for_rank(daemon_rank);
      }
      kernel_exec_info.req_counts.resize(req_for_rank.size());
      for (int i = 0; i < req_for_rank.size(); ++i) {
        kernel_exec_info.req_counts[i] = req_for_rank[i].req_count;
      }

      // ========【固定测试】calc
      // if (daemon_rank == 1) {
      //   if (kernel_req_data.kernel_count == 1) {
      //     kernel_exec_info.device_index = 1;
      //   } else if (kernel_req_data.kernel_count == 2) {
      //     kernel_exec_info.device_index = 2;
      //   } else if (kernel_req_data.kernel_count == 3) {
      //     kernel_exec_info.device_index = 3;
      //   } else if (kernel_req_data.kernel_count == 4) {
      //     kernel_exec_info.device_index = 4;
      //   } else if (kernel_req_data.kernel_count == 5) {
      //     kernel_exec_info.device_index = 1;
      //   } else if (kernel_req_data.kernel_count == 6) {
      //     kernel_exec_info.device_index = 1;
      //   } else if (kernel_req_data.kernel_count == 7) {
      //     kernel_exec_info.device_index = 3;
      //   } else if (kernel_req_data.kernel_count == 8) {
      //     kernel_exec_info.device_index = 2;
      //   } else if (kernel_req_data.kernel_count == 9) {
      //     kernel_exec_info.device_index = 1;
      //   }
      // }
      // else 
      //   kernel_exec_info.device_index = 1;

      // 向负责的进程mq发送 需要device->host的data
      std::string serialized_data = kernel_exec_info.serialize();
      size_t message_size = serialized_data.size();
      mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
      std::cout << "Rank " << daemon_rank << ": mq_send kernel_exec_info exec: " << kernel_exec_info.exec << " req_counts.size: " << kernel_exec_info.req_counts.size() << " device_index: " << kernel_exec_info.device_index << std::endl;
    }

    // ====【scale 向daemon发送依赖数据】
    // TODO master发送 若其他rank发送还需等待master通知 默认scale所需master都有最新？
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
            // std::cout << "Rank " << daemon_rank << ": Req " << i << std::endl;x x x x x x
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
  std::cout << "SystemSchedulerSubmit: SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << " SYCLAPP_Size " << syclapp_size << " DAEMON_Rank " << daemon_rank << " DAEMON_Size " << daemon_size << std::endl;

  // 始终由master开始
  if (syclapp_rank == master_rank) {
    // master记录目前参与计算的rank
    std::set<int> onrun_ranks = {master_rank};
    globalcount_to_onrun.insert(std::pair<int, std::set<int>>(syclapp_count, onrun_ranks));
    globalcount_to_scalecount.insert(std::pair<int, int>(syclapp_count, 1));

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
    pthread_create(&daemon_tid, NULL, (void *(*)(void *))SystemSchedulerDaemon, &syclapp_count);
    // 不等待 不影响主线程 无需cancel和join
    pthread_detach(daemon_tid);


    // master等待daemon的信号需要扩容
    while (1) {
      std::cout << "SYCLAPP_Rank " << syclapp_rank << " waiting DAEMON NOTIFY" << std::endl;
      auto queue = pid_to_scalecount_queue[pid];
      auto mutex = pid_to_scalecount_mutex[pid];
      auto cv = pid_to_scalecount_cv[pid];
      std::unique_lock<std::mutex> lock(*mutex);
      cv->wait(lock, [&queue] { return !queue->empty(); });
      while (!queue->empty()) {
        auto scale_pair = queue->front();
        queue->pop();
        std::cout << "SYCLAPP_Rank " << syclapp_rank << " scale_count: " << scale_pair.first << " rank: " << scale_pair.second << std::endl;
      
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
        std::cout << "SYCLAPP_Rank " << syclapp_rank << " scale_count: " << scale_pair.first << " sent to rank " << scale_pair.second << std::endl;
      }
    }
  }
  // 非master等待master的扩容请求
  else {
    while (1) {
      std::cout << "SYCLAPP_Rank " << syclapp_rank << " waiting scale_count" << std::endl;
      int scale_count;
      MPI_Recv(&scale_count, 1, MPI_INT, master_rank, 0, comm_syclapp, MPI_STATUS_IGNORE);
      std::cout << "SYCLAPP_Rank " << syclapp_rank << " scale_count: " << scale_count << " received from rank " << master_rank << std::endl;

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
      pthread_create(&daemon_tid, NULL, (void *(*)(void *))SystemSchedulerDaemon, &syclapp_count);
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
