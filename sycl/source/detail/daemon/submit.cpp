#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <mqueue.h>
#include <fcntl.h>
#include <pthread.h>
#include <string.h>

#define MESSAGE_QUEUE_SUBMIT_NAME "/sycl_user_submit"

#define MAX_MSG_NUM 10
#define MAX_MSG_SIZE 256

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <binary_path>" << std::endl;
        return 1;
    }

    struct mq_attr mq_attr;
    mq_attr.mq_flags = 0; // 堵塞模式
    mq_attr.mq_maxmsg = MAX_MSG_NUM;
    mq_attr.mq_msgsize = MAX_MSG_SIZE;

    mqd_t mq_id_submit = mq_open(MESSAGE_QUEUE_SUBMIT_NAME, O_WRONLY);
    if (mq_id_submit == -1) {
        perror("Error: mq_submit open failed");
        exit(1);
    }

    if (mq_send(mq_id_submit, argv[1], strlen(argv[1]) + 1, 0) == -1) {
        perror("Error: mq_send failed");
        mq_close(mq_id_submit);
        exit(1);
    }

    std::cout << "Submit to daemon: " << argv[1] << std::endl;
    mq_close(mq_id_submit);
    return 0;
}