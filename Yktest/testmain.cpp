#include "global/YkLog.h"
#include <driver_types.h>

// 各模块测试函数声明
int  main_fdk();
int  main_bp_runner();
int  main_bp_verify();
void test_fp_runner(cudaStream_t stream);

int main()
{
    YK::Logger::instance().set_level(YK::LogLevel::Debug);

    // 按需开启/注释
    //main_fdk();
    //main_bp_runner();
    main_bp_verify();
    //test_fp_runner(0);

    return 0;
}
