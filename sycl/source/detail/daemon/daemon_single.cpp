#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <mqueue.h>
#include <fcntl.h>
#include <pthread.h>

#include "daemon.hpp"

bool is_interrupted = false;

mqd_t mq_id_daemon;

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
    std::string errorMsg = "Error: Rank mq_id_daemon open failed";
    perror(errorMsg.c_str());
    exit(1);
  }
}

void CloseConnection() {
  mq_close(mq_id_daemon);
  mq_unlink(MESSAGE_QUEUE_DAEMON_NAME);
}

void SystemMonitor() {
  while(1) {
    // TODO: Implement the monitor logic
    {

    }
    usleep(MONITOR_INTERVAL);
  }
}

void SystemSchedulerDaemon() {
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

int main(int argc, char *argv[]) {
  signal(SIGINT, SignalHandler);

  EstablishConnection();

  pthread_t monitor_tid, scheduler_tid;
  // pthread_create(&monitor_tid, NULL, (void *(*)(void *))SystemMonitor, NULL);
  pthread_create(&scheduler_tid, NULL, (void *(*)(void *))SystemSchedulerDaemon, NULL);

  while (1) {
    if (is_interrupted) {
      break;
    }
    usleep(1000);
  }

  // pthread_cancel(monitor_tid);
  pthread_cancel(scheduler_tid);

  // pthread_join(monitor_tid, NULL);
  pthread_join(scheduler_tid, NULL);

  CloseConnection();

  return 0;
}
