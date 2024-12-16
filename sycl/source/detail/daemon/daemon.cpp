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

mqd_t mq_id_daemon, mq_id_submit;

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
  mq_attr.mq_flags = 0;
  mq_attr.mq_maxmsg = MAX_MSG_NUM;
  mq_attr.mq_msgsize = MAX_MSG_DAEMON_SIZE;

  mq_id_daemon = mq_open(MESSAGE_QUEUE_DAEMON_NAME, O_CREAT | O_RDONLY, 0666, &mq_attr);
  if (mq_id_daemon == -1) {
    std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " mq_id_daemon open failed";
    perror(errorMsg.c_str());
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
    std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " mq_id_submit open failed";
    perror(errorMsg.c_str());
    exit(1);
  }
}

void CloseConnection() {
  mq_close(mq_id_daemon);
  mq_unlink(MESSAGE_QUEUE_DAEMON_NAME);
}

void CloseSubmit() {
  mq_close(mq_id_submit);
  mq_unlink(MESSAGE_QUEUE_SUBMIT_NAME);
}

void BcastD2DKernelSchedInfo(int rank, D2DKernelSchedInfo &kernel_sched_info) {
  std::string serialized_data;
  if (rank == 0) {
    serialized_data = kernel_sched_info.serialize();
  }

  size_t str_length = serialized_data.size();
  MPI_Bcast(&str_length, 1, MPI_INT, 0, comm_daemon);

  char *buffer = new char[str_length + 1];
  if (rank == 0) {
    std::copy(serialized_data.begin(), serialized_data.end(), buffer);
    buffer[str_length] = '\0';
  }

  MPI_Bcast(buffer, str_length + 1, MPI_CHAR, 0, comm_daemon);

  if (rank != 0) {
    serialized_data = std::string(buffer);
    kernel_sched_info = D2DKernelSchedInfo::deserialize(serialized_data);
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

// TODO 并发的多个SYCL进程 逻辑没写完
std::map<pid_t, ProgramInfo> pid_to_program; // 一个SYCL进程的pid/rank在Submit中构造

// 一个SYCL进程的DAG相关
std::vector<DAGNode *> dag_nodes;
std::map<std::string, DAGNode *> kid_to_dag; // DAG leaves kernel间的依赖 ？
std::vector<DAGNode *> dag_leaves; // ？

#ifdef DISTRIBUTED
void SystemScheduler() {
  int rank;
  MPI_Comm_rank(comm_daemon, &rank);
  std::cout << "SystemScheduler: Rank " << rank << " started." << std::endl;  

  while (1) {
    // ====【接收program通信】
    S2DKernelReqData kernel_req_data;
    {
      char buffer[MAX_MSG_DAEMON_SIZE];
      ssize_t bytes_received = mq_receive(mq_id_daemon, buffer, MAX_MSG_DAEMON_SIZE, nullptr);
      if (bytes_received > 0) {
        std::string received_data(buffer, bytes_received);
        kernel_req_data = S2DKernelReqData::deserialize(received_data);
      } else {
        std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " DAEMON mq_receive failed";
        perror(errorMsg.c_str());
        exit(1);
      }
      std::cout << "Rank " << mpi_rank << ": mq_receive kernel_req_data pid: " << kernel_req_data.pid << std::endl;
    }
    ProgramInfo program_info = pid_to_program[kernel_req_data.pid];

    // ====【调度决策并发给其他rank】
    D2DKernelSchedInfo kernel_sched_info;
    {
      // TODO 1 [rank0] 构建DAG 确定依赖的kernel 查找依赖的kernel在哪个rank执行
      //                遍历pid对应的kernel
      // TODO 2 [allrank] 其他调度决策(设备负载/性能预测)
      // TODO 3 [rank0] 确定执行执行rank
      kernel_sched_info.kernel_count = kernel_req_data.kernel_count;
      // ========【固定测试】
      // A dep no - rank0
      // B dep no - rank1
      // C dep A:rank0, B:rank1 - rank0
      if (mpi_rank == 0) {
        if (kernel_req_data.kernel_count == 1) {
          // A
          kernel_sched_info.exec_rank = 0;
        } else if (kernel_req_data.kernel_count == 2) {
          // B
          kernel_sched_info.exec_rank = 1;
        } else if (kernel_req_data.kernel_count == 3) {
          // C
          kernel_sched_info.exec_rank = 0;
          kernel_sched_info.req_rank.insert({kernel_req_data.reqs[0], 0});
          kernel_sched_info.req_rank.insert({kernel_req_data.reqs[1], 1});
        }
      }

      // [4] [rank0][MPI] bcast (这个kernel 由哪个rank执行 依赖于哪些数据 这些数据在哪些rank上)
      //     [rank!0][MPI] bcast 接收并记录
      BcastD2DKernelSchedInfo(mpi_rank, kernel_sched_info);
      std::cout << "Rank " << mpi_rank << " kernel_sched_info.exec_rank: " << kernel_sched_info.exec_rank << std::endl;
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
        std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " mq_id_program open failed";
        perror(errorMsg.c_str());
        exit(1);
      }

      kernel_exec_info.kernel_count = kernel_sched_info.kernel_count;
      if (mpi_rank == kernel_sched_info.exec_rank) {
        kernel_exec_info.exec = true;
        req_for_rank = kernel_sched_info.get_req_for_exec_rank(mpi_rank);
      }
      else {
        req_for_rank = kernel_sched_info.get_req_for_rank(mpi_rank);
      }
      kernel_exec_info.req_counts.resize(req_for_rank.size());
      for (int i = 0; i < req_for_rank.size(); ++i) {
        kernel_exec_info.req_counts[i] = req_for_rank[i].req_count;
      }
      // ========【固定测试】
      kernel_exec_info.device_index = 1;

      // 向负责的进程mq发送 需要device->host的data
      std::string serialized_data = kernel_exec_info.serialize();
      size_t message_size = serialized_data.size();
      mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
      std::cout << "Rank " << mpi_rank << ": mq_send kernel_exec_info exec: " << kernel_exec_info.exec << std::endl;
    }

    // ====【为执行的rank满足依赖】
    {
      // [5] [单rank] 被依赖的kernel的数据device->host
      // 如果执行 要检查是否要从其他rank获取数据
      // 如果不执行 要检查是否需要host->device 给其他rank发数据
      // 说明有需要从其他rank获取的数据
      if (kernel_sched_info.req_rank.size() != kernel_sched_info.get_req_for_rank(kernel_sched_info.exec_rank).size()) {
        std::cout << "Rank " << mpi_rank << ": Exec Rank: " << kernel_sched_info.exec_rank << " need data from other" << std::endl;
        // 此rank不执行kernel 且kernel有依赖此rank的数据
        if (mpi_rank != kernel_sched_info.exec_rank && req_for_rank.size() > 0) {
          for (int i = 0; i < kernel_exec_info.req_counts.size(); i++) {
            // [6] 从SYCL进程接受host的data
            // 因为是写读共享内存是阻塞的 不需要等待SYCL进程的通知
            std::vector<DATA_TYPE> host_data(VECTOR_SIZE);
            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_sched_info.kernel_count, kernel_exec_info.req_counts[i]);
            readFromSharedMemory(handle, host_data.data());
            std::cout << "Rank " << mpi_rank << ": Data read successfully." << std::endl;
            cleanupSharedMemory(handle);

            // [7] [双rank][MPI] isend:host->buffer
            MPI_Send(host_data.data(), VECTOR_SIZE, MPI_FLOAT, kernel_sched_info.exec_rank, 0, comm_daemon);
            std::cout << "Rank " << mpi_rank << ": Sent data to rank " << kernel_sched_info.exec_rank << std::endl;
          }
        }

        // 此rank执行kernel 且必然需要从其他rank拿数据
        if (mpi_rank == kernel_sched_info.exec_rank) {
          for (int i = 0; i < kernel_exec_info.req_counts.size(); i++) {
            SyclReqData &req = req_for_rank[i];
            int data_rank = kernel_sched_info.req_rank[req];

            std::vector<DATA_TYPE> host_data(VECTOR_SIZE);
            // [7] [双rank][MPI] irecv:buffer->host
            MPI_Recv(host_data.data(), VECTOR_SIZE, MPI_FLOAT, data_rank, 0, comm_daemon, MPI_STATUS_IGNORE); // TODO rank写的不通用
            std::cout << "Rank " << mpi_rank << ": Received data from rank " << data_rank << std::endl;

            // [8] 把从其他rank接受的data发给SYCL进程
            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_sched_info.kernel_count, kernel_exec_info.req_counts[i]);
            writeToSharedMemory(handle, host_data.data());
            std::cout << "Rank " << mpi_rank << ": Write to shared" << std::endl;
            waitForReadCompletion(handle);
            cleanupSharedMemory(handle);

            std::cout << "Rank " << mpi_rank << ": waitForReadCompletion" << std::endl;
          }
        }
      }
    }
    mq_close(mq_id_program);
  }
}
#else
void SystemScheduler() {
  while (1) {
    char buffer[MAX_MSG_DAEMON_SIZE];
    S2DKernelReqData kernel_req_data;
    ssize_t bytes_received = mq_receive(mq_id_daemon, buffer, MAX_MSG_DAEMON_SIZE, nullptr);
    if (bytes_received > 0) {
      std::string received_data(buffer, bytes_received);
      kernel_req_data = S2DKernelReqData::deserialize(received_data);
    } else {
      perror("Error: mq_receive failed");
      exit(1);
    }
    std::cout << "Received kernel_req_data: " << kernel_req_data.pid << std::endl;

    // TODO: Implement the scheduler logic
    {

    }
    
    char MESSAGE_QUEUE_PROGRAM_NAME[MESSAGE_QUEUE_PROGRAM_NAME_MAX];
    sprintf(MESSAGE_QUEUE_PROGRAM_NAME, MESSAGE_QUEUE_PROGRAM_PATTERN, kernel_req_data.pid);
    mqd_t mq_id_program = mq_open(MESSAGE_QUEUE_PROGRAM_NAME, O_WRONLY);
    if (mq_id_program == -1) {
      perror("Error: mq_device open failed");
      exit(1);
    }

    D2SKernelExecInfo kernel_exec_info;
    std::string serialized_data = kernel_exec_info.serialize();
    size_t message_size = serialized_data.size();
    mq_send(mq_id_program, serialized_data.c_str(), message_size, 0);
    std::cout << "Send kernel_exec_info: " << kernel_exec_info.exec << std::endl;
    mq_close(mq_id_program);
  }
}
#endif

void SystemSchedulerSubmit() {
  int rank;
  MPI_Comm_rank(comm_submit, &rank);
  std::cout << "SystemSchedulerSubmit: Rank " << rank << " started." << std::endl;

  // while(1)用户向rank0提交bin_dir
  // 与管理单节点内的SystemScheduler是两个不同的pthread
  while (1) {
    char binary_path[MAX_MSG_SUBMIT_SIZE];

    // rank0会阻塞在此等待
    if (mpi_rank == 0) {
      ssize_t bytes_received = mq_receive(mq_id_submit, binary_path, MAX_MSG_SUBMIT_SIZE, NULL);
      if (bytes_received == -1) {
        std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " SUBMIT mq_receive failed";
        perror(errorMsg.c_str());
        exit(1);
      }
      std::cout << "Rank " << mpi_rank << ": Received submit path: " << binary_path << std::endl;
    }

    // 因为做了MPI_Comm_split 所以不会和另个线程的daemon冲突
    // 非rank0会阻塞在此等待
    MPI_Bcast(binary_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, 0, comm_submit);

    ProgramInfo program_info;
    // TODO 可以换成后台运行 向daemon发送进程号即可  
    pid_t pid = fork();
    if (pid == 0) { // 子进程
      execl(binary_path, binary_path, NULL);
      std::string errorMsg = "Error: Rank " + std::to_string(mpi_rank) + " Failed to execute binary";
      perror(errorMsg.c_str());
      exit(1);
    } else if (pid > 0) { // 父进程
      program_info.pid = pid;
      std::cout << "Rank " << mpi_rank << ": Launched binary with PID " << pid << std::endl;
    } else {
      std::cout << "Error: fork failed" << std::endl;
      exit(1);
    }

    program_info.pids.resize(mpi_size);
    MPI_Gather(&program_info.pid, 1, MPI_INT, program_info.pids.data(), 1, MPI_INT, 0, comm_submit);
    if (mpi_rank == 0) {
      std::cout << "Rank 0 collected PIDs from all processes:" << std::endl;
      for (int i = 0; i < mpi_size; ++i) {
        program_info.pid_to_rank.insert(std::pair<pid_t, int>(program_info.pids[i], i));
        std::cout << "Rank " << i << " PID: " << program_info.pids[i] << std::endl;
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
