#pragma once
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "YkFDKViewTexSlot.hpp"
#include "YkUtil.hpp"
#include "YkGlobals.h"

namespace YK {

    struct ChunkBuffer {
        std::vector<std::unique_ptr<ViewTexSlot>> slots;
        cudaTextureObject_t* d_texObjs = nullptr;
        int capacity = 0;

        void init(int K, int Nu, int Nv) {
            capacity = K;
            slots.resize(K);
            for (auto& s : slots) {
                s = std::make_unique<ViewTexSlot>();
                s->alloc(Nu, Nv);
            }
            YK_CUDA_CHECK(cudaMalloc(&d_texObjs, K * sizeof(cudaTextureObject_t)));
        }

        void uploadTexObjs(int K, cudaStream_t stream) {
            std::vector<cudaTextureObject_t> h(K);
            for (int i = 0; i < K; ++i) h[i] = slots[i]->texObj;
            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_texObjs, h.data(),
                K * sizeof(cudaTextureObject_t),
                cudaMemcpyHostToDevice, stream));
        }

        float* slotPtr(int i) { return slots[i]->d_data; }

        void destroy() {
            slots.clear();
            if (d_texObjs) { cudaFree(d_texObjs); d_texObjs = nullptr; }
        }

        ~ChunkBuffer() { destroy(); }
    };

} // namespace YK