#include <iostream>
#include <vector>
#include <cstdlib>
#include <signal.h>
#include <unistd.h>
#include <mqueue.h>
#include <fcntl.h>
#include <pthread.h>
#include <string.h>
#include <stdint.h>

#define MESSAGE_QUEUE_SUBMIT_NAME "/sycl_mq_submit"

#define MAX_MSG_NUM 10
#define MAX_MSG_SIZE 4096

static std::vector<char> BuildSubmitPayload(int argc, char *argv[]) {
    std::vector<char> payload;
    uint32_t arg_count = static_cast<uint32_t>(argc - 1);
    const char *arg_count_bytes = reinterpret_cast<const char *>(&arg_count);
    payload.insert(payload.end(), arg_count_bytes,
                   arg_count_bytes + sizeof(arg_count));

    for (int i = 1; i < argc; ++i) {
        size_t arg_len = strlen(argv[i]) + 1;
        payload.insert(payload.end(), argv[i], argv[i] + arg_len);
    }

    return payload;
}

static void PrintSubmitCommand(int argc, char *argv[]) {
    std::cout << "Submit to daemon:";
    for (int i = 1; i < argc; ++i) {
        std::cout << " " << argv[i];
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <binary_path> [args...]" << std::endl;
        return 1;
    }

    std::vector<char> payload = BuildSubmitPayload(argc, argv);
    if (payload.size() > MAX_MSG_SIZE) {
        std::cerr << "Error: submit command is too large (" << payload.size()
                  << " bytes, max " << MAX_MSG_SIZE << " bytes)" << std::endl;
        return 1;
    }

    mqd_t mq_id_submit = mq_open(MESSAGE_QUEUE_SUBMIT_NAME, O_WRONLY);
    if (mq_id_submit == -1) {
        perror("Error: USER mq_submit open failed");
        exit(1);
    }

    if (mq_send(mq_id_submit, payload.data(), payload.size(), 0) == -1) {
        perror("Error: USER mq_send failed");
        mq_close(mq_id_submit);
        exit(1);
    }

    PrintSubmitCommand(argc, argv);
    mq_close(mq_id_submit);
    return 0;
}
