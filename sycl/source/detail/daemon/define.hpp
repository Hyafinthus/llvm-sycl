// #define PRINT_CREATE 1 // pi_opencl.cpp中create相关
// #define PRINT_ELSE 1 // pi_opencl.cpp中除create以外调用
// #define PRINT_PI 1 // plugin.hpp中所有picall
// #define PRINT_KERNEL 1 // commands.cpp中kernel所有参数
// #define PRINT_TRACE 1 // 整个运行时trace
// #define MODIFY 1 // DAG和MPI相关 暂时弃用

#define REBIND 1 // 重绑定queue与device
#define SCHEDULE 1 // 重绑定的调度决策