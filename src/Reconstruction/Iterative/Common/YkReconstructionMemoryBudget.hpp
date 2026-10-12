#pragma once
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
namespace YK::detail {
inline bool fitsReconstructionBudget(size_t required,const char* stage){
    size_t free=0,total=0;if(cudaMemGetInfo(&free,&total)!=cudaSuccess)return false;
    const size_t reserve=std::max(size_t(512)*1024*1024,total/5);
    if(free>=reserve && required<=free-reserve)return true;
    std::cerr<<"[reconstruction memory] "<<stage<<" requires_bytes="<<required
        <<" free_bytes="<<free<<" render_reserve_bytes="<<reserve
        <<"; reduce projection/subset/grid sizes"<<std::endl;
    return false;
}
}
