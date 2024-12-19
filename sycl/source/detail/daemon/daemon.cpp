#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <mqueue.h>
#include <fcntl.h>
#include <pthread.h>
#include <mpi.h>

#include "daemon.hpp"

bool is_interrupted = false;

// signal pthread MPI mq shmem

// 一个SYCLAPP的全局信息
int global_syclapp_count = 0; // 对于整个集群的SYCLAPP计数 因都会在Submit的Bcast前阻塞 每个节点的计数保持相等
std::map<int, pid_t> globalcount_to_pid; // SYCLAPP计数_每个进程上不同的pid
std::map<pid_t, ProgramInfo> pid_to_program; // 当前rank上的pid_整个SYCLAPP的进程信息

// ====【MPI】
int mpi_rank, mpi_size; // main
MPI_Comm comm_submit; // SystemSchedulerSubmit
MPI_Comm comm_syclapp; // TODO 这个不能全局 每个进程有多个

// ====【mq】
mqd_t mq_id_submit;

// 一个SYCLAPP的DAG相关
std::vector<DAGNode *> dag_nodes;
std::map<std::string, DAGNode *> kid_to_dag; // DAG leaves kernel间的依赖 ？
std::vector<DAGNode *> dag_leaves; // ？

void SignalHandler(int signum) {
  if (signum == SIGINT) {
    std::cout << "Interrupted!" << std::endl;
    is_interrupted = true;
  }
}

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

void SystemMonitor() {
  while(1) {
    {

    }
    usleep(MONITOR_INTERVAL);
  }
}

void *SystemSchedulerDaemon(void *arg) {
  int syclapp_count = *(int *)arg;
  int local_pid = globalcount_to_pid[syclapp_count];
  
  ProgramInfo &program_info = pid_to_program[local_pid];
  mqd_t mq_id_daemon = EstablishDaemon(local_pid);
  MPI_Comm &comm_daemon = program_info.comm_syclapp;
  int &daemon_rank = program_info.syclapp_rank;
  int &daemon_size = program_info.syclapp_size;
  int &master_rank = program_info.master_rank;

  // int daemon_rank, daemon_size;
  // MPI_Comm_rank(comm_daemon, &daemon_rank);
  // MPI_Comm_size(comm_daemon, &daemon_size);
  std::cout << "SystemSchedulerDaemon: Rank " << daemon_rank << " for PID " << local_pid << " started." << std::endl;  

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
        std::string errorMsg = "Error: Rank " + std::to_string(daemon_rank) + " DAEMON mq_receive failed";
        perror(errorMsg.c_str());
        exit(1);
      }
      std::cout << "Rank " << daemon_rank << ": mq_receive kernel_req_data pid: " << kernel_req_data.pid << std::endl;
    }

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
      if (daemon_rank == 0) {
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
      BcastD2DKernelSchedInfo(comm_daemon, master_rank, daemon_rank, kernel_sched_info);
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

      kernel_exec_info.kernel_count = kernel_sched_info.kernel_count;
      if (daemon_rank == kernel_sched_info.exec_rank) {
        kernel_exec_info.exec = true;
        req_for_rank = kernel_sched_info.get_req_for_exec_rank(daemon_rank);
      }
      else {
        req_for_rank = kernel_sched_info.get_req_for_rank(daemon_rank);
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
      std::cout << "Rank " << daemon_rank << ": mq_send kernel_exec_info exec: " << kernel_exec_info.exec << std::endl;
    }

    // ====【为执行的rank满足依赖】
    {
      // [5] [单rank] 被依赖的kernel的数据device->host
      // 如果执行 要检查是否要从其他rank获取数据
      // 如果不执行 要检查是否需要host->device 给其他rank发数据
      // 说明有需要从其他rank获取的数据
      if (kernel_sched_info.req_rank.size() != kernel_sched_info.get_req_for_rank(kernel_sched_info.exec_rank).size()) {
        std::cout << "Rank " << daemon_rank << ": Exec Rank: " << kernel_sched_info.exec_rank << " need data from other" << std::endl;
        // 此rank不执行kernel 且kernel有依赖此rank的数据
        if (daemon_rank != kernel_sched_info.exec_rank && req_for_rank.size() > 0) {
          for (int i = 0; i < kernel_exec_info.req_counts.size(); i++) {
            // [6] 从SYCL进程接受host的data
            // 因为是写读共享内存是阻塞的 不需要等待SYCL进程的通知
            std::vector<DATA_TYPE> host_data(VECTOR_SIZE);
            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_sched_info.kernel_count, kernel_exec_info.req_counts[i]);
            readFromSharedMemory(handle, host_data.data());
            std::cout << "Rank " << daemon_rank << ": Data read successfully." << std::endl;
            cleanupSharedMemory(handle);

            // [7] [双rank][MPI] isend:host->buffer
            MPI_Send(host_data.data(), VECTOR_SIZE, MPI_FLOAT, kernel_sched_info.exec_rank, 0, comm_daemon);
            std::cout << "Rank " << daemon_rank << ": Sent data to rank " << kernel_sched_info.exec_rank << std::endl;
          }
        }

        // 此rank执行kernel 且必然需要从其他rank拿数据
        if (daemon_rank == kernel_sched_info.exec_rank) {
          for (int i = 0; i < kernel_exec_info.req_counts.size(); i++) {
            SyclReqData &req = req_for_rank[i];
            int data_rank = kernel_sched_info.req_rank[req];

            std::vector<DATA_TYPE> host_data(VECTOR_SIZE);
            // [7] [双rank][MPI] irecv:buffer->host
            MPI_Recv(host_data.data(), VECTOR_SIZE, MPI_FLOAT, data_rank, 0, comm_daemon, MPI_STATUS_IGNORE);
            std::cout << "Rank " << daemon_rank << ": Received data from rank " << data_rank << std::endl;

            // [8] 把从其他rank接受的data发给SYCL进程
            SharedMemoryHandle handle = initSharedMemory(kernel_req_data.pid, kernel_sched_info.kernel_count, kernel_exec_info.req_counts[i]);
            writeToSharedMemory(handle, host_data.data());
            std::cout << "Rank " << daemon_rank << ": Write to shared" << std::endl;
            waitForReadCompletion(handle);
            cleanupSharedMemory(handle);

            std::cout << "Rank " << daemon_rank << ": waitForReadCompletion" << std::endl;
          }
        }
      }
    }
    mq_close(mq_id_program);
  }
}

void *SystemSchedulerSubmit(void *arg) {
  int submit_rank, submit_size;
  MPI_Comm_rank(comm_submit, &submit_rank);
  MPI_Comm_size(comm_submit, &submit_size);
  std::cout << "SystemSchedulerSubmit: SUBMIT_Rank " << submit_rank << " started." << std::endl;

  // while(1)用户向rank0提交bin_dir
  // 与管理单节点内的SystemSchedulerDaemon是两个不同的pthread
  while (1) {
    char binary_path[MAX_MSG_SUBMIT_SIZE];
    std::vector<int> exec_flags(submit_size, 0);

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

      // 选择执行此SYCLAPP的ranks 并随机选择rank作为scheduler(master)
      // ====【固定测试】
      exec_flags[0] = 0; // non-exec
      exec_flags[1] = 2; // master
      exec_flags[2] = 1; // exec
    }

    // 非rank0会阻塞在此等待
    MPI_Bcast(exec_flags.data(), submit_size, MPI_INT, 0, comm_submit);
    global_syclapp_count++;
    int split_color = exec_flags[submit_rank] == 0 ? MPI_UNDEFINED : global_syclapp_count;
    // rank0将path发送给master
    if (submit_rank == 0) {
      for (int r = 0; r < submit_size; r++) {
        if (exec_flags[r] == 2) {
          MPI_Send(binary_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, r, 1, comm_submit);
          std::cout << "SUBMIT_Rank " << submit_rank << " sent binary path to master: " << binary_path << std::endl;
          break;
        }
      }
    }
    if (exec_flags[submit_rank] == 2) {
      MPI_Recv(binary_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, 0, 1, comm_submit, MPI_STATUS_IGNORE);
      std::cout << "SUBMIT_Rank " << submit_rank << " received binary path: " << binary_path << std::endl;
    }

    // 为选择的ranks创建新的通信域 color为global_syclapp_count
    // 所有rank必须参与 不参与的color=MPI_UNDEFINED
    MPI_Comm_split(comm_submit, split_color, submit_rank, &comm_syclapp);
    // rank0也可能non-exec
    if (exec_flags[submit_rank] == 0) {
      std::cout << "SUBMIT_Rank " << submit_rank << ": Not exec" << std::endl;
      continue;
    }
    std::cout << "SUBMIT_Rank " << submit_rank << ": continue" << std::endl;
    // comm_submit到此结束
    int syclapp_rank, syclapp_size;
    MPI_Comm_rank(comm_syclapp, &syclapp_rank);
    MPI_Comm_size(comm_syclapp, &syclapp_size);
    int master_rank = master_rank_syclapp(exec_flags);
    MPI_Bcast(binary_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, master_rank, comm_syclapp);
    std::cout << "SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << " from master_rank: " << master_rank <<  " received binary path: " << binary_path << std::endl;

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
    globalcount_to_pid.insert(std::pair<int, pid_t>(global_syclapp_count, pid));

    program_info.global_syclapp_count = global_syclapp_count;
    program_info.set_mpi(comm_syclapp, syclapp_rank, syclapp_size, master_rank);
    program_info.pids.resize(syclapp_size);
    MPI_Gather(&program_info.pid, 1, MPI_INT, program_info.pids.data(), 1, MPI_INT, master_rank, comm_syclapp);
    if (syclapp_rank == master_rank) {
      std::cout << "SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << master_rank << " collected PIDs from all processes:" << std::endl;
      for (int i = 0; i < syclapp_size; ++i) {
        program_info.pid_to_rank.insert(std::pair<pid_t, int>(program_info.pids[i], i));
        std::cout << "SYCLAPP_Rank " << i << " PID: " << program_info.pids[i] << std::endl;
      }
    }
    pid_to_program.insert(std::pair<pid_t, ProgramInfo>(program_info.pid, program_info));

    // **注意** 每个rank的pid不同 但global_syclapp_count相同
    pthread_t daemon_tid;
    pthread_create(&daemon_tid, NULL, (void *(*)(void *))SystemSchedulerDaemon, &global_syclapp_count);
    // TODO 要主动cancel和join此daemon
  }
}

int main(int argc, char *argv[]) {
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

  // ====【signal】
  signal(SIGINT, SignalHandler);

  // ====【mq】
  EstablishSubmit();
  std::cout << "MPI_Rank " << mpi_rank << ": Established" << std::endl;

  // ====【pthread】
  pthread_t monitor_tid, submit_tid;
  // pthread_create(&monitor_tid, NULL, (void *(*)(void *))SystemMonitor, NULL);
  pthread_create(&submit_tid, NULL, (void *(*)(void *))SystemSchedulerSubmit, NULL);
  std::cout << "MPI_Rank " << mpi_rank << ": SystemSchedulerSubmit started" << std::endl;

  // ====【signal】
  while (1) {
    if (is_interrupted) {
      break;
    }
    usleep(1000);
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
