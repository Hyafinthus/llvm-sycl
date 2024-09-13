#pragma once

#define MONITOR_INTERVAL 500 // us

#define MAX_MSG_NUM 10

#define MESSAGE_QUEUE_KERNEL_NAME "/sycl_mq_kernel"

#define MESSAGE_QUEUE_DEVICE_NAME_MAX 50
#define MESSAGE_QUEUE_DEVICE_PATTERN "/sycl_mq_device_%d"

struct KernelData {
  pid_t pid;
};

struct DeviceData {
  int dev;
};
