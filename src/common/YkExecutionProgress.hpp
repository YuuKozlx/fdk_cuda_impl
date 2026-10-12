#pragma once
#include <cuda_runtime_api.h>
namespace YK::detail {
using ExecutionProgressCallback=void(*)(void*,int,int);
struct ProgressContext {ExecutionProgressCallback callback=nullptr;void* user=nullptr;int suppressed=0;};
inline thread_local ProgressContext execution_progress;
class ScopedExecutionProgress {
    ProgressContext previous_;
public:
    ScopedExecutionProgress(ExecutionProgressCallback callback,void* user):previous_(execution_progress){execution_progress={callback,user,0};}
    ~ScopedExecutionProgress(){execution_progress=previous_;}
};
class SuppressExecutionProgress {
public:
    SuppressExecutionProgress(){++execution_progress.suppressed;}
    ~SuppressExecutionProgress(){--execution_progress.suppressed;}
};
inline bool reportExecutionProgress(int completed,int total,cudaStream_t stream){
    const auto& progress=execution_progress;
    if(!progress.callback||progress.suppressed)return true;
    // Opt-in reporting describes GPU completion, not kernel enqueue progress.
    if(cudaStreamSynchronize(stream)!=cudaSuccess)return false;
    try{progress.callback(progress.user,completed,total);}catch(...){return false;}
    return true;
}
}
