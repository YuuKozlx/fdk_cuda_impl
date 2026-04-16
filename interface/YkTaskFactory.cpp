// YkTaskFactory.cpp
#include "YkTaskFactory.hpp"

#include <cstdio>
#include "../FDK/YkFdkTaskHandle.hpp"
#include "../FP/YkFpTaskHandle.hpp"
#include "IYkTask.hpp"
#include "YkTaskTypes.hpp"

namespace YK {

    ITask* TaskFactory::create(ETask task)
    {
        switch (task) {
        case ETask::FDK:
            return new FdkTaskHandle();
        case ETask::FP_Joseph:
        case ETask::FP_Siddon:
        case ETask::FP_CVP:
            return new FpTaskHandle(task);
        case ETask::SART:
        case ETask::OSEM:
            fprintf(stderr, "[ProcFactory] task %d not yet implemented\n", (int)task);
            return nullptr;
        default:
            fprintf(stderr, "[ProcFactory] unknown task %d\n", (int)task);
            return nullptr;
        }
    }

    void TaskFactory::destroy(ITask* p)
    {
        delete p;
    }

} // namespace YK