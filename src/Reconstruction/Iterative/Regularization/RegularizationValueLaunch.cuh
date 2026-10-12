#pragma once
#include <cuda_runtime_api.h>
namespace YK::Iter {
void regularization_value_launch(const float*, int, int, int, float, float, float, float, int, int, double*, cudaStream_t);
}
