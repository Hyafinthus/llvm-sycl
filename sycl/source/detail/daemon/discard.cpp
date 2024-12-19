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