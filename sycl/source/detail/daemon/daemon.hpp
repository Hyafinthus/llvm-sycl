#ifndef MESSAGE_QUEUE
#define MESSAGE_QUEUE

#include <vector>
#include <map>
#include <sstream>
#include <string>

#pragma once

#define MONITOR_INTERVAL 500 // us

#define MAX_MSG_NUM 10

#define MAX_MSG_SUBMIT_SIZE 256
#define MESSAGE_QUEUE_SUBMIT_NAME "/sycl_user_submit"

#define MESSAGE_QUEUE_KERNEL_NAME "/sycl_mq_kernel"

#define MESSAGE_QUEUE_DEVICE_NAME_MAX 50
#define MESSAGE_QUEUE_DEVICE_PATTERN "/sycl_mq_device_%d"

struct KernelData {
  pid_t pid;
};

struct DeviceData {
  int dev;
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

struct KernelExecInfo { // 广播的一个kernel由哪个rank执行的信息
  std::string kernel_id;
  int kernel_count;

  int exec_rank; // 要执行的daemon的rank
  std::map<std::string, int> data_rank; // 需要的数据 在哪个rank上

  std::vector<std::string> get_data_for_rank(int rank) {
    std::vector<std::string> data;
    for (const auto &pair : data_rank) {
      if (pair.second == rank) {
        data.push_back(pair.first);
      }
    }
    return data;
  }

  std::string serialize() const {
    std::ostringstream oss;
    oss << kernel_id << "\n"
        << kernel_count << "\n"
        << exec_rank << "\n";

    oss << data_rank.size() << "\n";
    for (const auto &pair : data_rank) {
      oss << pair.first << "\n" << pair.second << "\n";
    }

    return oss.str();
  }

  static KernelExecInfo deserialize(const std::string &data) {
    std::istringstream iss(data);
    KernelExecInfo info;

    std::getline(iss, info.kernel_id);
    iss >> info.kernel_count;
    iss >> info.exec_rank;

    int map_size;
    iss >> map_size;
    iss.ignore();
    for (int i = 0; i < map_size; ++i) {
      std::string key;
      int value;
      std::getline(iss, key);
      iss >> value;
      iss.ignore();
      info.data_rank[key] = value;
    }

    return info;
  }
};

struct DataInfo { // daemon向SYCL进程发送的数据信息
  int kernel_count;
  bool exec = false; // 是否执行
  int data_count; // 测试 无实际用处
};

#endif

// TODO 共享内存扩展到多个并发SYCL进程
#ifndef SHARED_MEMORY
#define SHARED_MEMORY

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>
#include <cstring>

using DATA_TYPE = float;

#define SHARED_MEMORY_NAME "/shared_memory_data"
#define SEM_WRITE_NAME "/sem_write"
#define SEM_READ_NAME "/sem_read"

#define VECTOR_SIZE (256*256)
#define MEMORY_SIZE (VECTOR_SIZE * sizeof(DATA_TYPE))

struct SharedMemoryHandle {
  int shm_fd;
  void *shared_memory;
  sem_t *sem_write;
  sem_t *sem_read;
};

inline SharedMemoryHandle initSharedMemory() {
  SharedMemoryHandle handle;

  // 创建共享内存对象
  handle.shm_fd = shm_open(SHARED_MEMORY_NAME, O_CREAT | O_RDWR, 0666);
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
  handle.sem_write = sem_open(SEM_WRITE_NAME, O_CREAT, 0666, 0);
  if (handle.sem_write == SEM_FAILED) {
    perror("sem_open sem_write");
    exit(EXIT_FAILURE);
  }

  handle.sem_read = sem_open(SEM_READ_NAME, O_CREAT, 0666, 0);
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
  // 关闭和释放资源
  munmap(handle.shared_memory, MEMORY_SIZE);
  close(handle.shm_fd);
  sem_close(handle.sem_write);
  sem_close(handle.sem_read);
  shm_unlink(SHARED_MEMORY_NAME);
  sem_unlink(SEM_WRITE_NAME);
  sem_unlink(SEM_READ_NAME);
}

#endif