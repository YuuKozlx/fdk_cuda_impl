#pragma once
#include "YKCBCT/interface/YkReconstructionTypes.hpp"
#include "common/YkDeviceWorkspace.hpp"
#include "global/YkMacro.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
namespace YK::detail {
inline int iterativeBatchViews(size_t pixels,int views){
    return int(std::max(size_t(1),std::min(size_t(views),std::min(size_t(64),size_t(256)*1024*1024/(pixels*sizeof(float))))));
}
class ProjectionInput {
    ProjectionReadCallback reader_;void* user_;const float* host_;
    size_t pixels_,pending_;int views_,preferred_,mib_,capacity_=0,first_=-1,cached_=0;
    DeviceWorkspaceF32 device_;std::vector<float> host_buffer_;
    cudaStream_t stream_=nullptr;std::string error_;
public:
    ProjectionInput(ProjectionReadCallback reader,void* user,const float* host,size_t pixels,int views,
        int preferred,int mib,size_t pending):reader_(reader),user_(user),host_(host),pixels_(pixels),pending_(pending),views_(views),preferred_(preferred),mib_(mib){}
    ~ProjectionInput(){if(capacity_)cudaStreamSynchronize(stream_);}
    const std::string& error()const{return error_;}
    int capacityViews()const{return capacity_;}
    int preferredViews()const{return preferred_;}
    bool prepare(int count,cudaStream_t stream){
        try{stream_=stream;
            if(!capacity_){
                size_t free=0,total=0;YK_CUDA_CHECK(cudaMemGetInfo(&free,&total));
                const size_t reserve=std::max(size_t(512)*1024*1024,total/5)+pending_;
                const size_t bytes=std::min(size_t(mib_)*1024*1024,free>reserve?free-reserve:0);
                capacity_=int(std::min(size_t(views_),std::min(bytes/(pixels_*sizeof(float)),size_t(std::max(preferred_,count)))));
                if(capacity_<count){capacity_=0;error_="projection batch budget cannot hold one algorithm subset; increase subsets or batch budget";return false;}
                int id=0;YK_CUDA_CHECK(cudaGetDevice(&id));device_.allocate(size_t(capacity_)*pixels_,id);
                host_buffer_.resize(size_t(capacity_)*pixels_);
            }
            return count<=capacity_;
        }catch(const std::exception& e){error_=e.what();device_.reset();capacity_=0;return false;}
    }
    const float* fetch(const std::vector<int>& requested,cudaStream_t stream){
        try{
            stream_=stream;const int count=int(requested.size());
            if(count==0){error_="empty projection subset";return nullptr;}
            for(int view:requested)if(view<0||view>=views_){error_="projection index out of range";return nullptr;}
            bool contiguous=true;for(int i=1;i<count;++i)contiguous=contiguous&&requested[i]==requested[0]+i;
            if(contiguous && first_>=0 && requested[0]>=first_ && requested.back()<first_+cached_)
                return device_.data()+size_t(requested[0]-first_)*pixels_;
            if(!prepare(count,stream))return nullptr;
            if(count>capacity_){error_="algorithm subset exceeds prepared projection batch capacity";return nullptr;}
            // Finish the previous use before overwriting either staging buffer.
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto indices=requested;
            if(contiguous){indices.resize(std::min(capacity_,views_-requested[0]));std::iota(indices.begin(),indices.end(),requested[0]);}
            first_=-1;cached_=0;
            const size_t elements=indices.size()*pixels_;
            if(reader_){if(!reader_(user_,indices.data(),int(indices.size()),host_buffer_.data(),elements)){
                error_="projection reader failed";return nullptr;}}
            else for(size_t i=0;i<indices.size();++i)std::memcpy(host_buffer_.data()+i*pixels_,host_+size_t(indices[i])*pixels_,pixels_*sizeof(float));
            YK_CUDA_CHECK(cudaMemcpyAsync(device_.data(),host_buffer_.data(),elements*sizeof(float),cudaMemcpyHostToDevice,stream_));
            if(contiguous){first_=indices[0];cached_=int(indices.size());}
            return device_.data();
        }catch(const std::exception& e){error_=e.what();return nullptr;}catch(...){error_="projection reader threw an exception";return nullptr;}
    }
};
inline thread_local ProjectionInput* projection_input=nullptr;
class ScopedProjectionInput {
    ProjectionInput* previous_;
public:explicit ScopedProjectionInput(ProjectionInput* input):previous_(projection_input){projection_input=input;}
    ~ScopedProjectionInput(){projection_input=previous_;}
};
inline const float* contiguousProjection(const float* resident,int first,int count,size_t pixels,cudaStream_t stream){
    if(!projection_input)return resident?resident+size_t(first)*pixels:nullptr;
    std::vector<int> indices(count);std::iota(indices.begin(),indices.end(),first);return projection_input->fetch(indices,stream);
}
}
