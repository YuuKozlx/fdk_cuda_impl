#pragma once
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>


namespace YK {


    namespace Util {
        void crop_u_2d(
            const float* src,
            float* dst,
            int Nu,
            int Nv,
            int paddedN,
            int start_u,
            cudaStream_t stream = 0);
    };

};