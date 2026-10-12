#pragma once
#include "YkTimingWatchdog.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace YK::diagnostics {
template<class Work> void cudaTimed(const char* name,std::string context,Work&& work){
    if(!timingEnabled()){work();return;}
    auto status=cudaStreamSynchronize(nullptr);if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    TimingSpan span(name,std::move(context));work();status=cudaStreamSynchronize(nullptr);
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
}
