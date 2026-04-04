#pragma once
#include <cuda_runtime.h>
#include "YkUtil.hpp"

namespace YK {

    struct ViewTexSlot {
        float* d_data = nullptr;
        cudaTextureObject_t texObj = 0;
        int Nu = 0, Nv = 0;

        ViewTexSlot() = default;
        ViewTexSlot(const ViewTexSlot&) = delete;
        ViewTexSlot& operator=(const ViewTexSlot&) = delete;

        void alloc(int nu, int nv) {
            Nu = nu; Nv = nv;
            YK_CUDA_CHECK(cudaMalloc(&d_data, (size_t)Nu * Nv * sizeof(float)));
            _buildTex();
        }

        void destroy() {
            if (texObj) { cudaDestroyTextureObject(texObj); texObj = 0; }
            if (d_data) { cudaFree(d_data);                 d_data = nullptr; }
        }

        ~ViewTexSlot() { destroy(); }

    private:
        void _buildTex() {
            cudaResourceDesc res{};
            res.resType = cudaResourceTypePitch2D;
            res.res.pitch2D.devPtr = d_data;
            res.res.pitch2D.desc = cudaCreateChannelDesc<float>();
            res.res.pitch2D.width = Nu;
            res.res.pitch2D.height = Nv;
            res.res.pitch2D.pitchInBytes = (size_t)Nu * sizeof(float);

            cudaTextureDesc tex{};
            tex.addressMode[0] = cudaAddressModeClamp;
            tex.addressMode[1] = cudaAddressModeClamp;
            tex.filterMode = cudaFilterModeLinear;
            tex.readMode = cudaReadModeElementType;
            tex.normalizedCoords = 0;

            YK_CUDA_CHECK(cudaCreateTextureObject(&texObj, &res, &tex, nullptr));
        }
    };

} // namespace YK