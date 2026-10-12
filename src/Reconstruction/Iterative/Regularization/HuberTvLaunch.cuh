#pragma once
#include <cuda_runtime_api.h>
namespace YK::Iter {
void huber_tv_gradient_launch(const float*, float*, int, int, int, float, float, float, float, int, cudaStream_t);
void huber_tv_prox_update_launch(float*, const float*, const float*, size_t, float, float, bool, cudaStream_t);
void bounded_prox_update_launch(float*, const float*, const float*, size_t, float, float, float, float, cudaStream_t);
void tv_dual_launch(const float*, float*, int,int,int,float,float,float,int,float,float,cudaStream_t);
void tv_primal_launch(float*,float*,const float*,const float*,int,int,int,float,float,float,int,float,float,float,cudaStream_t);
}
