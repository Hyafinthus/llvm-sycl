#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <mqueue.h>
#include <fcntl.h>
#include <pthread.h>
// #include <mutex>

#include "daemon.hpp"

#define DISTRIBUTED 1

#ifdef DISTRIBUTED
#include <mpi.h>
#endif

// std::mutex mpi_mutex;
// mpi_mutex.lock();
// mpi_mutex.unlock();

bool is_interrupted = false;

mqd_t mq_id_kernel, mq_id_submit;

int mpi_rank, mpi_size;
MPI_Comm comm_daemon; // SystemScheduler
MPI_Comm comm_submit; // SystemSchedulerSubmit

void SignalHandler(int signum) {
  if (signum == SIGINT) {
    std::cout << "Interrupted!" << std::endl;
    is_interrupted = true;
  }
}

void EstablishConnection() {
  struct mq_attr mq_attr;
  mq_attr.mq_maxmsg = MAX_MSG_NUM;
  mq_attr.mq_msgsize = sizeof(KernelData);

  mq_id_kernel = mq_open(MESSAGE_QUEUE_KERNEL_NAME, O_CREAT | O_RDONLY, 0666, &mq_attr);
  if (mq_id_kernel == -1) {
    perror("Error: mq_kernel open failed");
    exit(1);
  }
}

void EstablishSubmit() {
  struct mq_attr mq_attr;
  mq_attr.mq_flags = 0;
  mq_attr.mq_maxmsg = MAX_MSG_NUM;
  mq_attr.mq_msgsize = MAX_MSG_SUBMIT_SIZE;

  mq_id_submit = mq_open(MESSAGE_QUEUE_SUBMIT_NAME, O_CREAT | O_RDONLY, 0666, &mq_attr);
  if (mq_id_submit == -1) {
    perror("Error: mq_submit open failed");
    exit(1);
  }
}

void CloseConnection() {
  mq_close(mq_id_kernel);
  mq_unlink(MESSAGE_QUEUE_KERNEL_NAME);
}

void CloseSubmit() {
  mq_close(mq_id_submit);
  mq_unlink(MESSAGE_QUEUE_SUBMIT_NAME);
}

void BcastKernelExecInfo(int rank, KernelExecInfo &info) {
  std::string serialized_data;
  if (rank == 0) {
    serialized_data = info.serialize();
  }

  int str_length = serialized_data.size();
  MPI_Bcast(&str_length, 1, MPI_INT, 0, comm_daemon);

  char *buffer = new char[str_length + 1];
  if (rank == 0) {
    std::copy(serialized_data.begin(), serialized_data.end(), buffer);
    buffer[str_length] = '\0';
  }

  MPI_Bcast(buffer, str_length + 1, MPI_CHAR, 0, comm_daemon);

  if (rank != 0) {
    serialized_data = std::string(buffer);
    info = KernelExecInfo::deserialize(serialized_data);
  }

  delete[] buffer;
}

void SystemMonitor() {
  while(1) {
    // TODO: Implement the monitor logic
    {

    }
    usleep(MONITOR_INTERVAL);
  }
}

// ========【固定测试 只有一个SYCL进程】
KernelData kernel_data;

// pid是唯一区分SYCL进程的标识 其内kernel_id不唯一 只能靠kernel_count区分
std::vector<pid_t> pids;
int kernel_count = 0;

// Scheduler需要的数据结构
std::map<pid_t, int> pid_to_rank;
// 存储 所有的kernel 和 其读写数据
std::vector<DAGNode *> dag_nodes;
// DAG leaves kernel间的依赖
std::map<std::string, DAGNode *> kid_to_dag;
std::vector<DAGNode *> dag_leaves;

#ifdef DISTRIBUTED
void SystemScheduler() {
  // pthread_t thread_id = pthread_self();
  // std::cout << "Rank " << mpi_rank << ": Thread created with ID: " << thread_id << std::endl;

  int rank;
  MPI_Comm_rank(comm_daemon, &rank);
  std::cout << "SystemScheduler: Rank " << rank << " started." << std::endl;

  while (1) {
    mq_receive(mq_id_kernel, (char *)&kernel_data, sizeof(KernelData), NULL);
    std::cout << "Rank " << mpi_rank << ": Received kernel data: " << kernel_data.pid << std::endl;

    // 分布式 主动运行进程 还需要监听等待每个进程的kernel
    char MESSAGE_QUEUE_DEVICE_NAME[MESSAGE_QUEUE_DEVICE_NAME_MAX];
    sprintf(MESSAGE_QUEUE_DEVICE_NAME, MESSAGE_QUEUE_DEVICE_PATTERN, kernel_data.pid);
    mqd_t mq_id_device = mq_open(MESSAGE_QUEUE_DEVICE_NAME, O_WRONLY);
    if (mq_id_device == -1) {
      perror("Error: mq_device open failed");
      exit(1);
    }

    {
      kernel_count++;
      // TODO 1 [rank0] 构建DAG 确定依赖的kernel 查找依赖的kernel在哪个rank执行
      //                遍历pid对应的kernel
      // ========【固定测试】
      // A dep no
      // B dep no
      // C dep A:rank0, B:rank1;
      if (kernel_count == 1) {
        // A
      } else if (kernel_count == 2) {
        // B
      } else if (kernel_count == 3) {
        // C
      }

      // TODO 2 [allrank] 其他调度决策(设备负载/性能预测?)
      // TODO 3 [rank0] 确定执行执行rank
      // ========【固定测试】
      // A:rank0 B:rank1 C:rank0

      // TODO 4 [rank0][MPI] bcast (这个kernel 由哪个rank执行 依赖于哪些数据 这些数据在哪些rank上)
      //        [rank!0][MPI] bcast 接收并记录
      KernelExecInfo info;
      if (kernel_count == 1) {
        // A
        info.kernel_id = "kernel1";
        info.kernel_count = 1;
        info.exec_rank = 0;
      } else if (kernel_count == 2) {
        // B
        info.kernel_id = "kernel2";
        info.kernel_count = 2;
        info.exec_rank = 1;
      } else if (kernel_count == 3) {
        // C
        info.kernel_id = "kernel3";
        info.kernel_count = 3;
        info.exec_rank = 0;
        info.data_rank["E"] = 0;
        info.data_rank["F"] = 1;
      }

      BcastKernelExecInfo(mpi_rank, info);
      std::cout << "Rank " << mpi_rank << ": info.kernel_id: " << info.kernel_id << " info.exec_rank: " << info.exec_rank << std::endl;

      // TODO 5 [单rank] 被依赖的kernel的数据device->host
      std::vector<std::string> data_for_rank = info.get_data_for_rank(mpi_rank);
      // 向SYCL进程发送要host的data
      DataInfo data_info;
      if (mpi_rank == info.exec_rank) {
        data_info.exec = true;
      }
      data_info.kernel_count = info.kernel_count;
      data_info.data_count = 1;
      std::cout << "Rank " << mpi_rank << ": Send data info: " << data_info.exec << std::endl;
      mq_send(mq_id_device, (char *)&data_info, sizeof(DataInfo), 0);
      // 以上每个rank都要给daemon发 因为需要告诉daemon是否执行

      // 如果执行 要检查是否要从其他rank获取数据
      // 如果不执行 要检查是否需要host->device 给其他rank发数据

      // 说明有需要从其他rank获取的数据
      if (info.data_rank.size() != info.get_data_for_rank(info.exec_rank).size()) {
        std::cout << "Rank " << mpi_rank << ": Exec Rank: " << info.exec_rank << " need data from other" << std::endl;
        // TODO 6 从SYCL进程接受host的data
        // 我不执行 且有依赖我的
        if (mpi_rank != info.exec_rank && data_for_rank.size() > 0) {
          std::cout << "Rank " << mpi_rank << ": Receive notification" << std::endl;
          // 从SYCL进程接受通知 才可以读取

          // DISCARD 检查消息队列内消息数量
          // struct mq_attr attr;
          // if (mq_getattr(mq_id_kernel, &attr) == -1) {
          //     perror("mq_getattr failed");
          //     exit(1);
          // }
          // std::cout << "Current messages in queue: " << attr.mq_curmsgs << std::endl;
          // std::cout << "Max message size: " << attr.mq_msgsize << std::endl;
          // ssize_t bytes_read = mq_receive(mq_id_kernel, notify, sizeof(KernelData), NULL);
          // if (bytes_read == -1) {
          //     perror("mq_receive failed");
          //     exit(1);
          // }
          // 错的写法 缓冲区必须和mq_msgsize一样大
          // char notify;
          // ssize_t bytes_read = mq_receive(mq_id_kernel, &notify, sizeof(notify), NULL);

          // DISCARD 通过消息队列通知daemon可以读取共享内存
          // char *notify = new char[sizeof(KernelData)];
          // mq_receive(mq_id_kernel, notify, sizeof(KernelData), NULL);
          // std::cout << "Rank " << mpi_rank << ": Received notification: '" << notify[0] << "'" << std::endl;
          // if (notify[0] != 'W') {
          //     std::cout << "Rank " << mpi_rank << ": Error: App notification failed" << std::endl;
          //     exit(1);
          // }
        
          std::vector<DATA_TYPE> host_data(VECTOR_SIZE);

          SharedMemoryHandle handle = initSharedMemory();
          readFromSharedMemory(handle, host_data.data());
          std::cout << "Rank " << mpi_rank << ": Data read successfully." << std::endl;
          cleanupSharedMemory(handle);

          MPI_Send(host_data.data(), VECTOR_SIZE, MPI_FLOAT, info.exec_rank, 0, comm_daemon);
          std::cout << "Rank " << mpi_rank << ": Sent data to rank " << info.exec_rank << std::endl;
        }

        // TODO 7 [双rank][MPI] isend:host->buffer irecv:buffer->host

        // TODO 8 把从其他rank接受的data发给SYCL进程
        // 外面要套一层循环 对应每个rank TEST 用rank1代替
        if (mpi_rank == info.exec_rank) {
          std::vector<DATA_TYPE> host_data(VECTOR_SIZE);
          MPI_Recv(host_data.data(), VECTOR_SIZE, MPI_FLOAT, 1, 0, comm_daemon, MPI_STATUS_IGNORE);
          std::cout << "Rank " << mpi_rank << ": Received data from rank 1" << std::endl;

          SharedMemoryHandle handle = initSharedMemory();
          writeToSharedMemory(handle, host_data.data());
          std::cout << "Rank " << mpi_rank << ": Write to shared" << std::endl;
          waitForReadCompletion(handle);
          cleanupSharedMemory(handle);

          // DISCARD 给SYCL进程发数据 通知可以取回
          // char notify = 'R';
          // mq_send(mq_id_device, &notify, sizeof(notify), 0);
        }
      }
      
      // TODO 9 [node内] 确定是否执行 如果执行具体执行device
      
    }

    mq_close(mq_id_device);
  }
}
#else
void SystemScheduler() {
  KernelData kernel_data;

  while (1) {
    mq_receive(mq_id_kernel, (char *)&kernel_data, sizeof(KernelData), NULL);
    std::cout << "Received kernel data: " << kernel_data.pid << std::endl;
    // TODO: Implement the scheduler logic
    {

    }
    DeviceData device_data;
    device_data.dev = 1;

    char MESSAGE_QUEUE_DEVICE_NAME[MESSAGE_QUEUE_DEVICE_NAME_MAX];
    sprintf(MESSAGE_QUEUE_DEVICE_NAME, MESSAGE_QUEUE_DEVICE_PATTERN, kernel_data.pid);

    mqd_t mq_id_device = mq_open(MESSAGE_QUEUE_DEVICE_NAME, O_WRONLY);
    if (mq_id_device == -1) {
      perror("Error: mq_device open failed");
      exit(1);
    }

    std::cout << "Send device data: " << device_data.dev << std::endl;
    mq_send(mq_id_device, (char *)&device_data, sizeof(DeviceData), 0);
    mq_close(mq_id_device);
  }
}
#endif

void SystemSchedulerSubmit() {
  // while(1)用户向rank0提交bin_dir
  // 与管理单节点内的SystemScheduler是两个不同的pthread

  int rank;
  MPI_Comm_rank(comm_submit, &rank);
  std::cout << "SystemSchedulerSubmit: Rank " << rank << " started." << std::endl;

  while (1) {
    char binary_path[MAX_MSG_SUBMIT_SIZE];

    // rank0会阻塞在此等待
    if (mpi_rank == 0) {
      ssize_t bytes_received = mq_receive(mq_id_submit, binary_path, MAX_MSG_SUBMIT_SIZE, NULL);
      if (bytes_received == -1) {
        perror("Error: mq_submit receive failed");
        exit(1);
      }
      std::cout << "Rank " << mpi_rank << ": Received submit path: " << binary_path << std::endl;
    }

    // 因为做了MPI_Comm_split 所以不会和另个线程的daemon冲突
    // 非rank0会阻塞在此等待
    MPI_Bcast(binary_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, 0, comm_submit);

    // 用fork获取子进程pid 暂时用作Debug
    // TODO 可以换成后台运行 向daemon发送进程号即可
    pid_t pid = fork();
    if (pid == 0) { // 子进程
      execl(binary_path, binary_path, NULL);
      perror("Error: Failed to execute binary");
      exit(1);
    } else if (pid > 0) { // 父进程
      kernel_data.pid = pid;
      std::cout << "Rank " << mpi_rank << ": Launched binary with PID " << pid << std::endl;
    } else {
      std::cout << "Error: fork failed" << std::endl;
      exit(1);
    }

    pids.resize(mpi_size);
    MPI_Gather(&kernel_data.pid, 1, MPI_INT, pids.data(), 1, MPI_INT, 0, comm_submit);
    if (mpi_rank == 0) {
      std::cout << "Rank 0 collected PIDs from all processes:" << std::endl;
      for (int i = 0; i < mpi_size; ++i) {
        pid_to_rank.insert(std::pair<pid_t, int>(pids[i], i));
        std::cout << "Rank " << i << " PID: " << pids[i] << std::endl;
      }
    }
  }
}

int main(int argc, char *argv[]) {
#ifdef DISTRIBUTED
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided < MPI_THREAD_MULTIPLE) {
      perror("Error: MPI NO SUPPORT MPI_THREAD_MULTIPLE");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  std::cout << "Rank " << mpi_rank << " of " << mpi_size << " started" << std::endl;

  MPI_Comm_split(MPI_COMM_WORLD, 1, mpi_rank, &comm_daemon);
  MPI_Comm_split(MPI_COMM_WORLD, 2, mpi_rank, &comm_submit);
#endif

  signal(SIGINT, SignalHandler);

  EstablishConnection();
#ifdef DISTRIBUTED
  EstablishSubmit();
#endif
  std::cout << "Rank " << mpi_rank << ": Established" << std::endl;

  pthread_t monitor_tid, scheduler_dis_tid, scheduler_tid;
  // pthread_create(&monitor_tid, NULL, (void *(*)(void *))SystemMonitor, NULL);

  pthread_create(&scheduler_tid, NULL, (void *(*)(void *))SystemScheduler, NULL);
  std::cout << "Rank " << mpi_rank << ": SystemScheduler started" << std::endl;

#ifdef DISTRIBUTED
  pthread_create(&scheduler_dis_tid, NULL, (void *(*)(void *))SystemSchedulerSubmit, NULL);
  std::cout << "Rank " << mpi_rank << ": SystemSchedulerSubmit started" << std::endl;
#endif

  while (1) {
    if (is_interrupted) {
      break;
    }
    usleep(1000);
  }

  // pthread_cancel(monitor_tid);
  pthread_cancel(scheduler_tid);
#ifdef DISTRIBUTED
  pthread_cancel(scheduler_dis_tid);
#endif

  // pthread_join(monitor_tid, NULL);
  pthread_join(scheduler_tid, NULL);
#ifdef DISTRIBUTED
  pthread_join(scheduler_dis_tid, NULL);
#endif

  CloseConnection();
#ifdef DISTRIBUTED
  CloseSubmit();

  MPI_Comm_free(&comm_daemon);
  MPI_Comm_free(&comm_submit);
  MPI_Finalize();
#endif
  return 0;
}
