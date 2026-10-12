#pragma once
#include "ProjectionGeometry.cuh"
#include "QuantumSampling.cuh"
namespace yk::spectral::primary_detail {
// Stratify the original broad uniform focus into the normal rectangle and its
// complement. Reuse normal rays for the inner broad-focus contribution.
struct FocalPartition {float outer_fraction;bool stratified;};
__host__ __device__ inline FocalPartition makeFocalPartition(DeviceGeometry g){
    const bool nested=g.focal_epsilon>0.f&&g.defocus_fu>0.f&&g.defocus_fv>0.f&&
        g.normal_fu>=0.f&&g.normal_fv>=0.f&&g.normal_fu<=g.defocus_fu&&g.normal_fv<=g.defocus_fv;
    if(!nested)return {g.focal_epsilon,false};
    const float inner_ratio=(g.normal_fu/g.defocus_fu)*(g.normal_fv/g.defocus_fv);
    return {g.focal_epsilon*(1.f-inner_ratio),true};
}
__host__ __device__ inline float focalPartitionWeight(FocalPartition p,bool outer){
    return outer?p.outer_fraction:1.f-p.outer_fraction;
}
// Four disjoint bands, selected proportional to area. No rejection loop and
// no samples in the interior of the normal focus rectangle.
__host__ __device__ inline float2 outerFocalPoint(DeviceGeometry g,float u,float v){
    const float vertical=(g.defocus_fu-g.normal_fu)/g.defocus_fu;
    const float horizontal=(g.normal_fu/g.defocus_fu)*((g.defocus_fv-g.normal_fv)/g.defocus_fv);
    const float a=u*(vertical+horizontal);
    if(vertical>0.f&&(a<vertical||horizontal<=0.f)){
        const float t=2.f*a/vertical,band=.5f*(g.defocus_fu-g.normal_fu);
        return make_float2(t<1.f?-.5f*g.defocus_fu+t*band:.5f*g.normal_fu+(t-1.f)*band,(v-.5f)*g.defocus_fv);
    }
    const float t=2.f*(a-vertical)/horizontal,band=.5f*(g.defocus_fv-g.normal_fv);
    return make_float2((v-.5f)*g.normal_fu,t<1.f?-.5f*g.defocus_fv+t*band:.5f*g.normal_fv+(t-1.f)*band);
}
static __device__ float2 sampleFocalPoint(DeviceGeometry g,FocalPartition p,bool outer,unsigned base){
    if(outer&&p.stratified){
        const unsigned outer_seed=base^0xd1b54a35U;
        return outerFocalPoint(g,uniform(outer_seed+1),uniform(outer_seed+2));
    }
    return make_float2((uniform(base+1)-.5f)*(outer?g.defocus_fu:g.normal_fu),
        (uniform(base+2)-.5f)*(outer?g.defocus_fv:g.normal_fv));
}
} // namespace yk::spectral::primary_detail
