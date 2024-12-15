#pragma once

#include <vector>
#include <map>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>
#include <cstring>

// #include <sycl/access/access.hpp>
namespace sycl {
    namespace access {
        enum class mode;
    }
}

#ifndef MESSAGE_QUEUE
#define MESSAGE_QUEUE

#define MONITOR_INTERVAL 500 // us

#define MAX_MSG_NUM 10

#define MAX_MSG_SUBMIT_SIZE 256
#define MESSAGE_QUEUE_SUBMIT_NAME "/sycl_mq_submit"

#define MAX_MSG_DAEMON_SIZE 1024
#define MESSAGE_QUEUE_DAEMON_NAME "/sycl_mq_daemon"

#define MAX_MSG_PROGRAM_SIZE 512
#define MESSAGE_QUEUE_PROGRAM_NAME_MAX 50
#define MESSAGE_QUEUE_PROGRAM_PATTERN "/sycl_mq_program_%d"

// D2D: Daemon to Daemon
// D2S: Daemon to SYCL
// S2D: SYCL to Daemon

// NOTE: All counts start from 1
// NOTE: SYCL Device starts from 0

// =================================

struct SyclReqData { // daemon需要的一个Req(AccessorImplHost)的信息
  void *req_pointer;
  int kernel_count;
  int req_count;
  sycl::access::mode req_accmode;

  bool operator<(const SyclReqData &other) const {
    return req_count < other.req_count;
  }

  // 不同daemon间定位同一个数组
  // 如果指针相同 或kernel_count和req_count相同
  bool operator==(const SyclReqData &other) const {
    return (req_pointer == other.req_pointer) ||
           (kernel_count == other.kernel_count && req_count == other.req_count);
  }

  std::string serialize() const {
    std::ostringstream oss;
    oss << reinterpret_cast<uintptr_t>(req_pointer) << "\n"
        << kernel_count << "\n"
        << req_count << "\n"
        << static_cast<int>(req_accmode) << "\n";
    return oss.str();
  }

  static SyclReqData deserialize(const std::string &data) {
    std::istringstream iss(data);
    SyclReqData req;
    uintptr_t ptr;
    iss >> ptr;
    req.req_pointer = reinterpret_cast<void *>(ptr);
    iss >> req.kernel_count;
    iss >> req.req_count;
    int accmode;
    iss >> accmode;
    req.req_accmode = static_cast<sycl::access::mode>(accmode);
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
      for (int j = 0; j < 4; ++j) {
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
      for (int j = 0; j < 4; ++j) {
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

  std::vector<int> req_counts; // handler只有当前kernel的req 不需处理哪个req给哪个rank 只需发送给daemon

  std::string serialize() const {
    std::ostringstream oss;
    oss << kernel_count << "\n"
        << exec << "\n"
        << device_index << "\n";

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
  pid_t pid;
  std::vector<pid_t> pids; // 这个SYCLAPP在所有rank上的pid
  std::map<pid_t, int> pid_to_rank;

  // TODO 执行此SYCLAPP的rank
};

struct DAGNode {
  std::string kernel_id; // 在每个daemon中不同
  int kernel_count; // 同一个SYCL进程中靠计数区分

  int mpi_rank;
  int pid;

  std::vector<std::string> read_only;
  std::vector<std::string> write_only;
  std::vector<std::string> read_write;

  std::vector<DAGNode *> depend_on;
  std::vector<DAGNode *> depend_by;

  bool executed = false;  
};

#endif

// =================================

#ifndef SHARED_MEMORY
#define SHARED_MEMORY

using DATA_TYPE = float;

#define SHARED_MEMORY_NAME_MAX 50
// pid_kernelcount 才能确保唯一
#define SHARED_MEMORY_NAME_PATTERN "/sycl_shm_kernel_%d_%d"
#define SHARED_MEMORY_WRITE_NAME_PATTERN "/sycl_shm_write_%d_%d"
#define SHARED_MEMORY_READ_NAME_PATTERN "/sycl_shm_read_%d_%d"

#define VECTOR_SIZE (256*256)
#define MEMORY_SIZE (VECTOR_SIZE * sizeof(DATA_TYPE))

struct SharedMemoryHandle {
  int shm_fd;
  void *shared_memory;
  sem_t *sem_write;
  sem_t *sem_read;
  char shared_memory_name[SHARED_MEMORY_NAME_MAX];
  char sem_write_name[SHARED_MEMORY_NAME_MAX];
  char sem_read_name[SHARED_MEMORY_NAME_MAX];
};

inline SharedMemoryHandle initSharedMemory(int pid, int kernel_count) {
  SharedMemoryHandle handle;

  sprintf(handle.shared_memory_name, SHARED_MEMORY_NAME_PATTERN, pid, kernel_count);
  sprintf(handle.sem_write_name, SHARED_MEMORY_WRITE_NAME_PATTERN, pid, kernel_count);
  sprintf(handle.sem_read_name, SHARED_MEMORY_READ_NAME_PATTERN, pid, kernel_count);

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

inline void writeToSharedMemory(SharedMemoryHandle &handle, DATA_TYPE *DataPtr) {
  // 写入数据到共享内存
  memcpy(handle.shared_memory, DataPtr, MEMORY_SIZE);
  // 通知读取端数据已准备好
  sem_post(handle.sem_write);
}

inline void waitForReadCompletion(SharedMemoryHandle &handle) {
  // 等待读取端完成读取
  sem_wait(handle.sem_read);
}

inline void readFromSharedMemory(SharedMemoryHandle &handle, DATA_TYPE *DataPtr) {
  // 等待数据准备完成
  sem_wait(handle.sem_write);
  // 读取数据
  memcpy(DataPtr, handle.shared_memory, MEMORY_SIZE);
  // 通知写入端读取完成
  sem_post(handle.sem_read);
}

inline void cleanupSharedMemory(SharedMemoryHandle &handle) {
  munmap(handle.shared_memory, MEMORY_SIZE);
  close(handle.shm_fd);
  sem_close(handle.sem_write);
  sem_close(handle.sem_read);
  shm_unlink(handle.shared_memory_name);
  sem_unlink(handle.sem_write_name);
  sem_unlink(handle.sem_read_name);
}

#endif