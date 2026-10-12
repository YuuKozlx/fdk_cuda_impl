#pragma once
#include "common/YkProjectionOperators.hpp"
#include "Reconstruction/Iterative/Common/YkProjectionInput.hpp"
#include "Reconstruction/Iterative/Common/YkIterationWeightCache.hpp"
#include "Reconstruction/Iterative/Common/YkReconstructionMemoryBudget.hpp"
#include "common/YkExecutionProgress.hpp"
#include "Reconstruction/Iterative/Common/kernels/YkIterLaunch.cuh"
namespace YK {
// One mathematical SIRT update over all views. Storage chunks never become
// algorithm subsets: sum every BP contribution before normalization/update.
class SirtBatchBackend {
    SReconstructionParams params_,low_;cudaStream_t stream_=nullptr;int device_=0,max_chunk_=0,cache_chunk_=0;
    bool coarse_=false,prepared_=false,min_=false,max_=false;float lambda_=1,initial_lambda_=1,reduction_=1,eps_=1e-6f,lo_=0,hi_=0;
    int iterations_=0;unsigned updates_=0;size_t pixels_=0,voxels_=0,slice_=0;
    ForwardOperatorAdapter fp_,fp_low_;BackOperatorAdapter bp_;
    DeviceWorkspaceF32 forward_,residual_,row_,column_,bp_sum_,ones_;
    detail::IterationWeightCache weights_;
    SReconstructionParams batch(int first,int count,bool low=false)const{
        auto p=low?low_:params_;p.scan.NAng=count;
        p.scan.angles.assign(params_.scan.angles.begin()+first,params_.scan.angles.begin()+first+count);return p;
    }
public:
    template<class Config> bool init(const SReconstructionParams& params,const Config& cfg,
        const std::vector<SConeProjGeomVec>& geometry,cudaStream_t stream,int device,bool coarse){
        release();params_=params;low_=params;stream_=stream;device_=device;coarse_=coarse;
        pixels_=size_t(params.scan.Nu)*params.scan.Nv;voxels_=size_t(params.volume.Nx)*params.volume.Ny*params.volume.Nz;
        slice_=size_t(params.volume.Nx)*params.volume.Ny;max_chunk_=detail::iterativeBatchViews(pixels_,params.scan.NAng);
        if(!detail::fitsReconstructionBudget((4*size_t(max_chunk_)*pixels_+(coarse?2:4)*voxels_+slice_)*sizeof(float)+size_t(32)*1024*1024,"SIRT batch workspaces"))return false;
        iterations_=cfg.n_iter;lambda_=initial_lambda_=cfg.lambda;reduction_=cfg.lambda_red;eps_=cfg.eps;
        min_=cfg.use_min;max_=cfg.use_max;lo_=cfg.min_constraint;hi_=cfg.max_constraint;
        const size_t projection=size_t(max_chunk_)*pixels_;
        forward_.allocate(projection,device);residual_.allocate(projection,device);row_.allocate(projection,device);
        column_.allocate(coarse?slice_:voxels_,device);bp_sum_.allocate(voxels_,device);ones_.allocate(coarse?8:voxels_,device);
        Iter::fill_ones_launch(ones_,coarse?8:voxels_,stream);
        if(!fp_.init(params,geometry,cfg.fp_task,device,stream)||!bp_.init(params,geometry,cfg.bp_task,device,stream))return false;
        if(coarse){
            low_.volume.Nx=low_.volume.Ny=low_.volume.Nz=2;
            low_.volume.voxelX_mm=params.volume.Nx*params.volume.voxelX_mm*1.1f/2.f;
            low_.volume.voxelY_mm=params.volume.Ny*params.volume.voxelY_mm*1.1f/2.f;
            low_.volume.voxelZ_mm=std::max(params.scan.Nv*params.scan.dv_mm,params.volume.Nz*params.volume.voxelZ_mm)/2.f;
            if(!fp_low_.init(low_,geometry,cfg.fp_task,device,stream))return false;
        }
        prepared_=true;return true;
    }
    bool iterate(const float* resident,float* volume,unsigned iterations){
        if(!prepared_)return false;
        const int chunk=detail::projection_input?std::min(max_chunk_,detail::projection_input->capacityViews()):max_chunk_;
        if(chunk<=0)return false;
        if(cache_chunk_!=chunk){std::vector<size_t> counts{coarse_?slice_:voxels_};
            for(int first=0;first<params_.scan.NAng;first+=chunk)counts.push_back(size_t(std::min(chunk,params_.scan.NAng-first))*pixels_);
            weights_.configure(counts,device_,stream_);cache_chunk_=chunk;
        }
        for(unsigned iteration=0;iteration<iterations;++iteration){
            const float* column=weights_.get(0,column_,[&](float* out){
                const float sx=params_.volume.Nx*params_.volume.voxelX_mm,sy=params_.volume.Ny*params_.volume.voxelY_mm;
                const float norm=std::sqrt(sx*sx+sy*sy),scale=norm>1e-8f?std::max(sx,sy)/norm*.9f:.9f;
                for(int first=0;first<params_.scan.NAng;first+=chunk){
                    const int count=std::min(chunk,params_.scan.NAng-first);auto p=batch(first,count);
                    if(coarse_){p.volume.voxelX_mm*=scale;p.volume.voxelY_mm*=scale;p.volume.voxelZ_mm*=scale;}
                    Iter::fill_ones_launch(residual_,size_t(count)*pixels_,stream_);
                    if(!bp_.run(residual_,p,coarse_?bp_sum_.data():out,stream_,first==0))return false;
                }
                if(coarse_){Iter::mean_z_to_2d_launch(bp_sum_,out,params_.volume.Nx,params_.volume.Ny,params_.volume.Nz,stream_);
                    Iter::threshold_inf_launch(out,slice_,0.f,stream_);}
                return true;
            });
            if(!column)return false;
            int index=1;for(int first=0;first<params_.scan.NAng;first+=chunk,++index){
                const int count=std::min(chunk,params_.scan.NAng-first);const size_t elements=size_t(count)*pixels_;auto p=batch(first,count);
                const float* row=weights_.get(index,row_,[&](float* out){
                    YK_CUDA_CHECK(cudaMemsetAsync(out,0,elements*sizeof(float),stream_));
                    if(!(coarse_?fp_low_.run(ones_,batch(first,count,true),out,stream_):fp_.run(ones_,p,out,stream_)))return false;
                    if(coarse_){const float threshold=std::min({params_.volume.voxelX_mm,params_.volume.voxelY_mm,params_.volume.voxelZ_mm})/2.f;
                        Iter::threshold_inf_launch(out,elements,threshold,stream_);Iter::rcp_launch(out,elements,stream_);}
                    return true;
                });
                const float* measured=detail::contiguousProjection(resident,first,count,pixels_,stream_);if(!row||!measured)return false;
                YK_CUDA_CHECK(cudaMemsetAsync(forward_,0,elements*sizeof(float),stream_));
                if(!fp_.run(volume,p,forward_,stream_))return false;
                Iter::residual_launch(measured,forward_,residual_,elements,stream_);
                if(coarse_)Iter::multiply_launch(residual_,row,elements,stream_);else Iter::divide_launch(residual_,row,eps_,elements,stream_);
                if(!bp_.run(residual_,p,bp_sum_,stream_,first==0))return false;
            }
            if(coarse_)Iter::update_v2d_launch(volume,bp_sum_,column,lambda_,eps_,params_.volume.Nx,params_.volume.Ny,params_.volume.Nz,stream_);
            else Iter::update_launch(volume,bp_sum_,column,lambda_,eps_,voxels_,stream_);
            if(min_)Iter::clamp_min_launch(volume,voxels_,lo_,stream_);if(max_)Iter::clamp_max_launch(volume,voxels_,hi_,stream_);
            weights_.planGpu();weights_.retain(0,column);lambda_*=reduction_;++updates_;
            if(!detail::reportExecutionProgress(int(iteration+1),int(iterations),stream_))return false;
        }return true;
    }
    bool run(const float* input,float* volume){return iterate(input,volume,unsigned(iterations_));}
    void setRelaxation(float value){lambda_=value;}
    void reset(){updates_=0;lambda_=initial_lambda_;}
    unsigned totalIterations()const{return updates_;}
    void release(){if(prepared_)cudaStreamSynchronize(stream_);weights_.clear();
        fp_.release();fp_low_.release();bp_.release();forward_.reset();residual_.reset();row_.reset();column_.reset();bp_sum_.reset();ones_.reset();
        prepared_=false;cache_chunk_=0;updates_=0;stream_=nullptr;}
    ~SirtBatchBackend(){release();}
};
}
