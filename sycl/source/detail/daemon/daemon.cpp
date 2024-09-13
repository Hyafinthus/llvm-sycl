#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <mqueue.h>
#include <fcntl.h>
#include <pthread.h>

#include "daemon.hpp"

bool is_interrupted = false;

mqd_t mq_id_kernel;

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
    std::cerr << "Error: mq_kernel open failed" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CloseConnection() {
  mq_close(mq_id_kernel);
  mq_unlink(MESSAGE_QUEUE_KERNEL_NAME);
}

void SystemMonitor() {
  while(1) {
    // TODO: Implement the monitor logic

    usleep(MONITOR_INTERVAL);
  }
}

void SystemScheduler() {
  KernelData kernel_data;

  while (1) {
    mq_receive(mq_id_kernel, (char *)&kernel_data, sizeof(KernelData), NULL);
    std::cout << "Received kernel data: " << kernel_data.pid << std::endl;
    // TODO: Implement the scheduler logic
    DeviceData device_data;
    device_data.dev = 1;

    char MESSAGE_QUEUE_DEVICE_NAME[MESSAGE_QUEUE_DEVICE_NAME_MAX];
    sprintf(MESSAGE_QUEUE_DEVICE_NAME, MESSAGE_QUEUE_DEVICE_PATTERN, kernel_data.pid);

    mqd_t mq_id_device = mq_open(MESSAGE_QUEUE_DEVICE_NAME, O_WRONLY);
    if (mq_id_device == -1) {
      std::cerr << "Error: mq_device open failed" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "Send device data: " << device_data.dev << std::endl;
    mq_send(mq_id_device, (char *)&device_data, sizeof(DeviceData), 0);
    mq_close(mq_id_device);
  }
}

int main() {
  std::cout << "SYCL Daemon started" << std::endl;
  
  signal(SIGINT, SignalHandler);

  EstablishConnection();

  pthread_t monitor_tid, scheduler_tid;
  pthread_create(&monitor_tid, NULL, (void *(*)(void *))SystemMonitor, NULL);
  pthread_create(&scheduler_tid, NULL, (void *(*)(void *))SystemScheduler, NULL);

  while (1) {
    if (is_interrupted) {
      break;
    }

    usleep(1000);
  }


  pthread_cancel(monitor_tid);
  pthread_cancel(scheduler_tid);

  pthread_join(monitor_tid, NULL);
  pthread_join(scheduler_tid, NULL);

  CloseConnection();

  return 0;
}
