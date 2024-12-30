// handler.cpp ========================================================

// DISCARD 通过消息队列通知daemon可以读取共享内存
// char *notify = new char[sizeof(S2DKernelReqData)];
// mq_receive(mq_id_program, notify, sizeof(S2DKernelReqData), NULL);
// if (notify[0] != 'R') {
//   std::cerr << "Error: Daemon notification failed" << std::endl;
//   exit(EXIT_FAILURE);
// }

// DISCARD 通过消息队列通知daemon可以读取共享内存
// char notify = 'W';
// if (mq_send(mq_id_daemon, &notify, sizeof(notify), 0) == -1) {
//     perror("mq_send failed");
//     exit(EXIT_FAILURE);
// }
// std::cout << "=== handler === REBIND === mq send" << std::endl;

// DISCARD 原定单节点的调度决策
// D2SDeviceData device_data;
// mq_receive(mq_id_program, (char *)&device_data, sizeof(D2SDeviceData), NULL);
// std::cout << "=== handler === REBIND === received device data: " << device_data.dev << std::endl;





// daemon.cpp ========================================================

// #include <mutex>
// std::mutex mpi_mutex;
// mpi_mutex.lock();
// mpi_mutex.unlock();

// pthread_t thread_id = pthread_self();
// std::cout << "Rank " << mpi_rank << ": Thread created with ID: " << thread_id << std::endl;

// DISCARD 用scatter代替bcast 实现选择ranks执行SYCLAPP 但选择的ranks无感知
// char ranks_path[mpi_size][MAX_MSG_SUBMIT_SIZE] = {{0}};
// char recv_path[MAX_MSG_SUBMIT_SIZE] = {0};
// std::vector<int> exec_ranks = {0, 1};
// for (int &exec_rank : exec_ranks) {
//   sprintf(ranks_path[exec_rank], "%s", binary_path);
// }
// MPI_Scatter(ranks_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, recv_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, 0, comm_submit);
// if (strlen(recv_path) == 0) {
//   std::cout << "Rank " << mpi_rank << ": Not exec" << std::endl;
//   continue;
// }

// DISCARD 检查消息队列内消息数量
// struct mq_attr attr;
// if (mq_getattr(mq_id_daemon, &attr) == -1) {
//     perror("mq_getattr failed");
//     exit(1);
// }
// std::cout << "Current messages in queue: " << attr.mq_curmsgs << std::endl;
// std::cout << "Max message size: " << attr.mq_msgsize << std::endl;
// 从SYCL进程接受通知 才可以读取
// ssize_t bytes_read = mq_receive(mq_id_daemon, notify, sizeof(S2DKernelReqData), NULL);
// if (bytes_read == -1) {
//     perror("mq_receive failed");
//     exit(1);
// }
// 错的写法 缓冲区必须和mq_msgsize一样大
// char notify;
// ssize_t bytes_read = mq_receive(mq_id_daemon, &notify, sizeof(notify), NULL);

// DISCARD 通过消息队列通知daemon可以读取共享内存
// char *notify = new char[sizeof(S2DKernelReqData)];
// mq_receive(mq_id_daemon, notify, sizeof(S2DKernelReqData), NULL);
// std::cout << "Rank " << mpi_rank << ": Received notification: '" << notify[0] << "'" << std::endl;
// if (notify[0] != 'W') {
//     std::cout << "Rank " << mpi_rank << ": Error: App notification failed" << std::endl;
//     exit(1);
// }

// DISCARD 给SYCL进程发数据 通知可以取回
// char notify = 'R';
// mq_send(mq_id_program, &notify, sizeof(notify), 0);





// DISCARD Submit确定执行的ranks
// void *SystemSchedulerSubmit(void *arg) {
//   // sigset_t set;
//   // sigemptyset(&set);
//   // sigaddset(&set, SIGINT);
//   // pthread_sigmask(SIG_BLOCK, &set, NULL);
//   int submit_rank, submit_size;
//   MPI_Comm_rank(comm_submit, &submit_rank);
//   MPI_Comm_size(comm_submit, &submit_size);
//   std::cout << "SystemSchedulerSubmit: SUBMIT_Rank " << submit_rank << " started." << std::endl;
//   // while(1)用户向rank0提交bin_dir
//   // 与管理单节点内的SystemSchedulerDaemon是两个不同的pthread
//   while (1) {
//     char binary_path[MAX_MSG_SUBMIT_SIZE];
//     std::vector<int> exec_flags(submit_size, 0);
//     // rank0会阻塞在此等待
//     ssize_t bytes_received;
//     if (submit_rank == 0) {
//       bytes_received = mq_receive(mq_id_submit, binary_path, MAX_MSG_SUBMIT_SIZE, NULL);
//       if (bytes_received == -1) {
//         std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SUBMIT mq_receive failed";
//         perror(errorMsg.c_str());
//         exit(1);
//       }
//       std::cout << "SUBMIT_Rank " << submit_rank << ": Received submit path: " << binary_path << std::endl;
//       // 选择执行此SYCLAPP的ranks 并随机选择rank作为scheduler(master)
//       // ====【固定测试】
//       exec_flags[0] = 0; // non-exec
//       exec_flags[1] = 2; // master
//       exec_flags[2] = 1; // exec
//     }
//     // 非rank0会阻塞在此等待
//     MPI_Bcast(exec_flags.data(), submit_size, MPI_INT, 0, comm_submit);
//     global_syclapp_count++;
//     int split_color = exec_flags[submit_rank] == 0 ? MPI_UNDEFINED : global_syclapp_count;
//     // rank0将path发送给master
//     if (submit_rank == 0) {
//       for (int r = 0; r < submit_size; r++) {
//         if (exec_flags[r] == 2) {
//           MPI_Send(binary_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, r, 1, comm_submit);
//           std::cout << "SUBMIT_Rank " << submit_rank << " sent binary path to master: " << binary_path << std::endl;
//           break;
//         }
//       }
//     }
//     if (exec_flags[submit_rank] == 2) {
//       MPI_Recv(binary_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, 0, 1, comm_submit, MPI_STATUS_IGNORE);
//       std::cout << "SUBMIT_Rank " << submit_rank << " received binary path: " << binary_path << std::endl;
//     }
//     // 为选择的ranks创建新的通信域 color为global_syclapp_count
//     // 所有rank必须参与 不参与的color=MPI_UNDEFINED
//     MPI_Comm comm_syclapp;
//     MPI_Comm_split(comm_submit, split_color, submit_rank, &comm_syclapp);
//     // rank0也可能non-exec
//     if (exec_flags[submit_rank] == 0) {
//       std::cout << "SUBMIT_Rank " << submit_rank << ": Not exec" << std::endl;
//       continue;
//     }
//     std::cout << "SUBMIT_Rank " << submit_rank << ": continue" << std::endl;
//     // comm_submit到此结束
//     int syclapp_rank, syclapp_size;
//     MPI_Comm_rank(comm_syclapp, &syclapp_rank);
//     MPI_Comm_size(comm_syclapp, &syclapp_size);
//     int master_rank = master_rank_syclapp(exec_flags);
//     MPI_Bcast(binary_path, MAX_MSG_SUBMIT_SIZE, MPI_CHAR, master_rank, comm_syclapp);
//     std::cout << "SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << " from master_rank: " << master_rank <<  " received binary path: " << binary_path << std::endl;
//     // 必须要pid 不能像singlenode只监听接收 创建对应SYCLAPP的Daemon 创建mq
//     ProgramInfo program_info;
//     pid_t pid = fork();
//     if (pid == 0) { // 子进程
//       execl(binary_path, binary_path, NULL);
//       std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SYCLAPP_Rank " + std::to_string(syclapp_rank) + " Failed to execute binary";
//       perror(errorMsg.c_str());
//       exit(1);
//     } else if (pid > 0) { // 父进程
//       program_info.pid = pid;
//       std::cout << "SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << syclapp_rank << ": Launched binary with PID " << pid << std::endl;
//     } else {
//       std::string errorMsg = "Error: SUBMIT_Rank " + std::to_string(submit_rank) + " SYCLAPP_Rank " + std::to_string(syclapp_rank) + " Fork failed";
//       perror(errorMsg.c_str());
//       exit(1);
//     }
//     globalcount_to_pid.insert(std::pair<int, pid_t>(global_syclapp_count, pid));
//     program_info.global_syclapp_count = global_syclapp_count;
//     program_info.set_mpi(comm_syclapp, syclapp_rank, syclapp_size, master_rank);
//     program_info.pids.resize(syclapp_size);
//     MPI_Gather(&program_info.pid, 1, MPI_INT, program_info.pids.data(), 1, MPI_INT, master_rank, comm_syclapp);
//     if (syclapp_rank == master_rank) {
//       std::cout << "SUBMIT_Rank " << submit_rank << " SYCLAPP_Rank " << master_rank << " collected PIDs from all processes:" << std::endl;
//       for (int i = 0; i < syclapp_size; ++i) {
//         program_info.pid_to_rank_syclapp.insert(std::pair<pid_t, int>(program_info.pids[i], i));
//         std::cout << "SYCLAPP_Rank " << i << " PID: " << program_info.pids[i] << std::endl;
//       }
//       program_info.rank_syclapp_to_submit = map_rank_syclapp_submit(exec_flags);
//     }
//     pid_to_program.insert(std::pair<pid_t, ProgramInfo>(program_info.pid, program_info));
//     // **注意** 每个rank的pid不同 但global_syclapp_count相同
//     pthread_t daemon_tid;
//     pthread_create(&daemon_tid, NULL, (void *(*)(void *))SystemSchedulerDaemon, &global_syclapp_count);
//     // 不等待 不影响主线程 无需cancel和join
//     pthread_detach(daemon_tid);
//   }
//   return NULL;
// }
