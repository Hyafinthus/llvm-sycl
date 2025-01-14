#pragma once

#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <queue>
#include <set>
#include <unordered_map>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>
#include <cstring>
#include <mqueue.h>
#include <mpi.h>

// 与sycl::access::mode保持一致
enum class acc_mode {
  read = 1024,
  write = 1025,
  read_write = 1026,
  discard_write = 1027,
  discard_read_write = 1028,
  atomic = 1029
};

#ifndef MESSAGE_QUEUE
#define MESSAGE_QUEUE

#define MONITOR_INTERVAL 50000 // us
#define MONITOR_GATHER_INTERVAL 2000000
#define MONITOR_THRESHOLD 20

#define MAX_MSG_NUM 10

#define MAX_MSG_SUBMIT_SIZE 256
#define MESSAGE_QUEUE_SUBMIT_NAME "/sycl_mq_submit"

#define MAX_MSG_DAEMON_SIZE 1024
#define MESSAGE_QUEUE_DAEMON_NAME_MAX 50
#define MESSAGE_QUEUE_DAEMON_PATTERN "/sycl_mq_daemon_%d" // pid

#define MAX_MSG_PROGRAM_SIZE 512
#define MESSAGE_QUEUE_PROGRAM_NAME_MAX 50
#define MESSAGE_QUEUE_PROGRAM_PATTERN "/sycl_mq_program_%d" // pid

// D2D: Daemon to Daemon
// D2S: Daemon to SYCL
// S2D: SYCL to Daemon

// NOTE: All counts start from 1: kernel_count, req_count
// NOTE: SYCL Device starts from 0

// =================================

struct SyclReqData { // daemon需要的一个Req(AccessorImplHost)中Mem的信息
  // SYCL运行时中 不同kernel中同一个数组的Req不是同一个对象
  void *mem_pointer;
  int kernel_count;
  int req_count;
  acc_mode req_accmode;
  int elem_size;
  int buff_size;

  // 用于map/set/sort/priority_queue
  bool operator<(const SyclReqData &other) const {
    if (req_count != other.req_count) {
      return req_count < other.req_count;
    }
    if (kernel_count != other.kernel_count) {
      return kernel_count < other.kernel_count;
    }
    return mem_pointer < other.mem_pointer;
  }

  // 不同daemon间定位同一个数组
  // 如果指针相同 或kernel_count和req_count相同
  // 用于find/count/unique/remove/unordered
  bool operator==(const SyclReqData &other) const {
    return (mem_pointer == other.mem_pointer) ||
           (kernel_count == other.kernel_count && req_count == other.req_count);
  }

  std::string serialize() const {
    std::ostringstream oss;
    oss << reinterpret_cast<uintptr_t>(mem_pointer) << "\n"
        << kernel_count << "\n"
        << req_count << "\n"
        << static_cast<int>(req_accmode) << "\n"
        << elem_size << "\n"
        << buff_size << "\n";
    return oss.str();
  }

  static SyclReqData deserialize(const std::string &data) {
    std::istringstream iss(data);
    SyclReqData req;
    uintptr_t ptr;
    iss >> ptr;
    req.mem_pointer = reinterpret_cast<void *>(ptr);
    iss >> req.kernel_count;
    iss >> req.req_count;
    int accmode;
    iss >> accmode;
    req.req_accmode = static_cast<acc_mode>(accmode);
    iss >> req.elem_size;
    iss >> req.buff_size;
    return req;
  }
};

struct S2DKernelReqData { // daemon需要的一个kernel的信息
  pid_t pid;

  int kernel_count;
  int req_size;
  std::vector<SyclReqData> reqs;

  std::string serialize() const {
    std::ostringstream oss;
    oss << pid << "\n"
        << kernel_count << "\n"
        << req_size << "\n";

    oss << reqs.size() << "\n";
    for (const auto &req : reqs) {
      oss << req.serialize();
    }

    return oss.str();
  }

  static S2DKernelReqData deserialize(const std::string &data) {
    std::istringstream iss(data);
    S2DKernelReqData kernel_data;

    iss >> kernel_data.pid;
    iss >> kernel_data.kernel_count;
    iss >> kernel_data.req_size;

    size_t req_count;
    iss >> req_count;
    iss.ignore();

    for (size_t i = 0; i < req_count; ++i) {
      std::string req_data_serialized;
      for (int j = 0; j < 6; ++j) {
        std::string line;
        std::getline(iss, line);
        req_data_serialized += line + "\n";
      }
      kernel_data.reqs.push_back(SyclReqData::deserialize(req_data_serialized));
    }

    return kernel_data;
  }
};

struct D2DKernelSchedInfo { // daemon间广播(发送)的一个kernel由哪个rank执行的信息
  // std::string kernel_id; // 没有kernel这个对象 CommandGroup还未创建
  int kernel_count; // 是一个SYCL进程中kernel的唯一标识 体现在用户代码的顺序中

  int exec_rank; // 要执行的daemon的rank
  std::map<SyclReqData, int> req_rank; // 需要的数据 在哪个rank上

  std::vector<SyclReqData> get_req_for_rank(int rank) {
    std::vector<SyclReqData> reqs;
    for (const auto &pair : req_rank) {
      if (pair.second == rank) {
        reqs.push_back(pair.first);
      }
    }
    return reqs;
  }

  std::vector<SyclReqData> get_req_for_exec_rank(int rank) {
    std::vector<SyclReqData> reqs;
    for (const auto &pair : req_rank) {
      if (pair.second != rank) {
        reqs.push_back(pair.first);
      }
    }
    return reqs;
  }
  
  std::string serialize() const {
    std::ostringstream oss;
    oss << kernel_count << "\n"
        << exec_rank << "\n";

    oss << req_rank.size() << "\n";
    for (const auto &pair : req_rank) {
      oss << pair.first.serialize();
      oss << pair.second << "\n";
    }

    return oss.str();
  }

  static D2DKernelSchedInfo deserialize(const std::string &data) {
    std::istringstream iss(data);
    D2DKernelSchedInfo sched_info;

    iss >> sched_info.kernel_count;
    iss >> sched_info.exec_rank;

    size_t map_size;
    iss >> map_size;
    iss.ignore();

    for (size_t i = 0; i < map_size; ++i) {
      std::string req_data_serialized;
      for (int j = 0; j < 6; ++j) {
        std::string line;
        std::getline(iss, line);
        req_data_serialized += line + "\n";
      }
      SyclReqData req = SyclReqData::deserialize(req_data_serialized);

      int rank;
      iss >> rank;
      iss.ignore();

      sched_info.req_rank[req] = rank;
    }

    return sched_info;
  }
};

struct D2SKernelExecInfo { // daemon向SYCL进程发送的一个kernel是否执行等依赖数据信息
  int kernel_count; // 唯一标识
  bool exec = false; // 是否执行
  int device_index; // 执行设备

  // 快速跳过前几个kernel
  // 0: kernel_count从1开始 默认值
  // >1: 告知daemon需要scale
  // -1: 告知master提供依赖
  int scale_count = 0;

  // handler只有当前kernel的req 不需处理哪个req给哪个rank 只需发送给daemon
  // exec==false: 记录当前rank需要shmem给daemon的req
  // exec==true: 记录需要从其他所有rank获取的req
  std::vector<int> req_counts;

  std::string serialize() const {
    std::ostringstream oss;
    oss << kernel_count << "\n"
        << exec << "\n"
        << device_index << "\n"
        << scale_count << "\n";

    // 序列化 req_counts 的大小
    oss << req_counts.size() << "\n";
    for (const auto &count : req_counts) {
      oss << count << "\n";
    }

    return oss.str();
  }

  static D2SKernelExecInfo deserialize(const std::string &data) {
    std::istringstream iss(data);
    D2SKernelExecInfo kernel_info;

    iss >> kernel_info.kernel_count;
    iss >> kernel_info.exec;
    iss >> kernel_info.device_index;
    iss >> kernel_info.scale_count;

    size_t req_count;
    iss >> req_count;
    iss.ignore();

    kernel_info.req_counts.resize(req_count);
    for (size_t i = 0; i < req_count; ++i) {
      iss >> kernel_info.req_counts[i];
      iss.ignore();
    }

    return kernel_info;
  }
};

// =================================

struct ProgramInfo {
  int global_syclapp_count;
  pid_t pid;

  MPI_Comm comm_daemon;
  int daemon_rank;
  int daemon_size;
  int master_rank;

  // 只有master收集
  // std::vector<pid_t> pids; // 这个SYCLAPP在所有rank上的pid
  // std::map<pid_t, int> pid_to_rank_syclapp; // rank是syclapp_rank 不是mpi/submit_rank
  // std::map<int, int> rank_syclapp_to_submit; // 每个syclapp_rank对应哪个物理mpi/submit_rank

  void set_mpi(MPI_Comm comm, int rank, int size, int master) {
    comm_daemon = comm;
    daemon_rank = rank;
    daemon_size = size;
    master_rank = master;
  }
};

struct DAGNode { // 一个kernel的依赖关系
  std::vector<SyclReqData> req_data;

  std::vector<DAGNode *> depend_on;
  std::vector<DAGNode *> depend_by;

  int exec_rank = -1;
  // bool executed = false; // ？

  DAGNode(const std::vector<SyclReqData> &reqs) : req_data(reqs) {}
};

struct CpuTimes {
  unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
};

struct MonitorInfo {
  std::string name;
  double util_used;
  size_t mem_available; //kB
};

#endif

// =================================

#ifndef SHARED_MEMORY
#define SHARED_MEMORY

#define SHARED_MEMORY_NAME_MAX 50
// pid_kernelcnt_reqcnt 才能确保唯一
#define SHARED_MEMORY_NAME_PATTERN "/sycl_shm_kernel_%d_%d_%d"
#define SHARED_MEMORY_WRITE_NAME_PATTERN "/sycl_shm_write_%d_%d_%d"
#define SHARED_MEMORY_READ_NAME_PATTERN "/sycl_shm_read_%d_%d_%d"

using DATA_TYPE = std::byte;
// #define VECTOR_SIZE (256*256)
// #define MEMORY_SIZE (VECTOR_SIZE * sizeof(DATA_TYPE))

struct SharedMemoryHandle {
  int shm_fd;
  void *shared_memory;
  sem_t *sem_write;
  sem_t *sem_read;
  char shared_memory_name[SHARED_MEMORY_NAME_MAX];
  char sem_write_name[SHARED_MEMORY_NAME_MAX];
  char sem_read_name[SHARED_MEMORY_NAME_MAX];
};

inline SharedMemoryHandle initSharedMemory(int pid, int kernel_count, int req_count, int MEMORY_SIZE) {
  SharedMemoryHandle handle;

  sprintf(handle.shared_memory_name, SHARED_MEMORY_NAME_PATTERN, pid, kernel_count, req_count);
  sprintf(handle.sem_write_name, SHARED_MEMORY_WRITE_NAME_PATTERN, pid, kernel_count, req_count);
  sprintf(handle.sem_read_name, SHARED_MEMORY_READ_NAME_PATTERN, pid, kernel_count, req_count);

  // 创建共享内存对象
  handle.shm_fd = shm_open(handle.shared_memory_name, O_CREAT | O_RDWR, 0666);
  if (handle.shm_fd == -1) {
    perror("shm_open");
    exit(EXIT_FAILURE);
  }
  // 调整共享内存大小
  if (ftruncate(handle.shm_fd, MEMORY_SIZE) == -1) {
    perror("ftruncate");
    exit(EXIT_FAILURE);
  }
  // 映射共享内存
  handle.shared_memory = mmap(nullptr, MEMORY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, handle.shm_fd, 0);
  if (handle.shared_memory == MAP_FAILED) {
    perror("mmap");
    exit(EXIT_FAILURE);
  }
  // 打开信号量
  handle.sem_write = sem_open(handle.sem_write_name, O_CREAT, 0666, 0);
  if (handle.sem_write == SEM_FAILED) {
    perror("sem_open sem_write");
    exit(EXIT_FAILURE);
  }
  handle.sem_read = sem_open(handle.sem_read_name, O_CREAT, 0666, 0);
  if (handle.sem_read == SEM_FAILED) {
    perror("sem_open sem_read");
    exit(EXIT_FAILURE);
  }

  return handle;
}

inline void writeToSharedMemory(SharedMemoryHandle &handle, DATA_TYPE *DataPtr, int MEMORY_SIZE) {
  // 写入数据到共享内存
  memcpy(handle.shared_memory, DataPtr, MEMORY_SIZE);
  // 通知读取端数据已准备好
  sem_post(handle.sem_write);
}

inline void waitForReadCompletion(SharedMemoryHandle &handle) {
  // 等待读取端完成读取
  sem_wait(handle.sem_read);
}

inline void readFromSharedMemory(SharedMemoryHandle &handle, DATA_TYPE *DataPtr, int MEMORY_SIZE) {
  // 等待数据准备完成
  sem_wait(handle.sem_write);
  // 读取数据
  memcpy(DataPtr, handle.shared_memory, MEMORY_SIZE);
  // 通知写入端读取完成
  sem_post(handle.sem_read);
}

inline void cleanupSharedMemory(SharedMemoryHandle &handle, int MEMORY_SIZE) {
  munmap(handle.shared_memory, MEMORY_SIZE);
  close(handle.shm_fd);
  sem_close(handle.sem_write);
  sem_close(handle.sem_read);
  shm_unlink(handle.shared_memory_name);
  sem_unlink(handle.sem_write_name);
  sem_unlink(handle.sem_read_name);
}

#endif
