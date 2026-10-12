#pragma once
#include "common/YkDeviceWorkspace.hpp"
#include "global/YkMacro.hpp"
#include "global/YkLog.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace YK::detail {
// Exact float32 weights for one immutable backend geometry. Entries are selected
// once, not LRU: a cyclic sweep must not evict every weight before reuse.
class IterationWeightCache {
    static constexpr size_t missing=std::numeric_limits<size_t>::max();
    struct Entry {size_t count=0,offset=missing;bool host_allowed=false,gpu_ready=false;std::vector<float> host;};
    std::vector<Entry> entries_;
    DeviceWorkspaceF32 device_;
    cudaStream_t stream_=nullptr;
    int device_id_=0;
    bool planned_=false;
    size_t gpu_budget_=0,host_budget_=0,gpu_bytes_=0,host_bytes_=0;
    size_t gpu_hits_=0,host_hits_=0,computations_=0;
    static size_t budget(const char* name,size_t fallback,size_t maximum){
        if(const char* value=std::getenv(name))try{size_t used=0;const auto n=std::stoull(value,&used);
            if(used==std::string(value).size()&&n<=maximum)return size_t(n)*1024*1024;
        }catch(...){}
        return fallback*1024*1024;
    }
    float* slot(Entry& e){return e.offset==missing?nullptr:device_.data()+e.offset;}
public:
    void configure(const std::vector<size_t>& counts,int device_id,cudaStream_t stream){
        clear();device_id_=device_id;stream_=stream;
        gpu_budget_=budget("YK_TIGRE_WEIGHT_GPU_MIB",256,512);
        host_budget_=budget("YK_TIGRE_WEIGHT_CPU_MIB",4096,8192);
        size_t reserved=0;for(size_t count:counts){Entry e;e.count=count;
            if(count*sizeof(float)<=host_budget_-reserved){e.host_allowed=true;reserved+=count*sizeof(float);}
            entries_.push_back(std::move(e));}
    }
    template<class Produce> float* get(size_t index,float* workspace,Produce&& produce){
        auto& e=entries_.at(index);if(e.gpu_ready){++gpu_hits_;return slot(e);}
        float* target=slot(e);if(!target)target=workspace;
        if(!e.host.empty()){
            YK_CUDA_CHECK(cudaMemcpyAsync(target,e.host.data(),e.count*sizeof(float),cudaMemcpyHostToDevice,stream_));
            ++host_hits_;
        }else{
            if(!produce(target))return nullptr;++computations_;
            if(e.host_allowed){
                try{e.host.resize(e.count);}catch(const std::bad_alloc&){e.host_allowed=false;}
                if(!e.host.empty()){
                    YK_CUDA_CHECK(cudaMemcpyAsync(e.host.data(),target,e.count*sizeof(float),cudaMemcpyDeviceToHost,stream_));
                    // Pageable backing memory stays valid across reset/sweeps.
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream_));host_bytes_+=e.count*sizeof(float);
                }
            }
        }
        if(e.offset!=missing)e.gpu_ready=true;
        return target;
    }
    // Call after the first complete FP/BP update: operator textures and caller
    // projection/volume allocations must exist before spending optional memory.
    void planGpu(){
        if(planned_)return;planned_=true;if(!gpu_budget_)return;
        size_t free=0,total=0;YK_CUDA_CHECK(cudaMemGetInfo(&free,&total));
        const size_t reserve=std::max(size_t(512)*1024*1024,total/5);
        const size_t available=std::min(gpu_budget_,free>reserve?free-reserve:0);
        size_t used=0;for(auto& e:entries_){const size_t bytes=e.count*sizeof(float);
            if(bytes<=available-used){e.offset=used/sizeof(float);used+=bytes;}}
        if(!used)return;
        try{device_.allocate(used/sizeof(float),device_id_);}catch(...){
            device_.reset();for(auto& e:entries_)e.offset=missing;cudaGetLastError();return;}
        gpu_bytes_=used;YK_LOGI("[iterative weights] gpu_cache_bytes={} cpu_budget_bytes={} reserve_bytes={}",used,host_budget_,reserve);
    }
    void retain(size_t index,const float* values){
        auto& e=entries_.at(index);if(e.offset==missing||e.gpu_ready)return;
        YK_CUDA_CHECK(cudaMemcpyAsync(slot(e),values,e.count*sizeof(float),cudaMemcpyDeviceToDevice,stream_));e.gpu_ready=true;
    }
    void clear(){
        if(!entries_.empty())YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if(computations_||gpu_hits_||host_hits_)YK_LOGI("[iterative weights] gpu_bytes={} cpu_bytes={} computed={} gpu_hits={} cpu_hits={}",
            gpu_bytes_,host_bytes_,computations_,gpu_hits_,host_hits_);
        device_.reset();entries_.clear();stream_=nullptr;planned_=false;
        gpu_bytes_=host_bytes_=gpu_hits_=host_hits_=computations_=0;
    }
};
}
