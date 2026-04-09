#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>


namespace YK {
    namespace Util {
        struct CudaTimer {
            cudaEvent_t start, stop;
            cudaStream_t stream;
            const char* tag;

            CudaTimer(const char* tag, cudaStream_t stream = 0)
                : tag(tag), stream(stream)
            {
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, stream);
            }

            ~CudaTimer()
            {
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, start, stop);
                printf("[%s] %.3f ms\n", tag, ms);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
        };
    };
};

