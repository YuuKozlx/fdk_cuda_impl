#pragma once
#include <vector>
#include "YkFDKChunkBuffer.hpp"
#include "YkFDKPrecompute.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkGlobals.h"
#include "YkUtil.hpp"
#include "YkVecGeo.hpp"

namespace YK {

    struct FdkGpuContext {
        SConeProjectionVec* d_geo = nullptr;
        SFDKGeoParamPerView* d_gv = nullptr;
        float* d_vol = nullptr;
        float* d_view_in = nullptr;
        float* d_view_pw = nullptr;
        float* d_view_flt = nullptr;
        float* d_padded = nullptr;
        FdkAffineCoeff* d_coeffs = nullptr;

        ChunkBuffer chunk;

        FdkGpuContext() = default;
        FdkGpuContext(const FdkGpuContext&) = delete;
        FdkGpuContext& operator=(const FdkGpuContext&) = delete;

        void init(const SDimensions3D& dims,
            const std::vector<SConeProjectionVec>& h_geo,
            const std::vector<SFDKGeoParamPerView>& h_gv,
            int Kchunk, int paddedN,
            cudaStream_t stream)
        {
            const size_t view_elems = (size_t)dims.iPU * dims.iPV;
            const size_t vol_elems = (size_t)dims.iVX * dims.iVY * dims.iVZ;

            YK_CUDA_CHECK(cudaMalloc(&d_coeffs, dims.iPAng * sizeof(FdkAffineCoeff)));
            YK_CUDA_CHECK(cudaMalloc(&d_geo, dims.iPAng * sizeof(SConeProjectionVec)));
            YK_CUDA_CHECK(cudaMalloc(&d_gv, dims.iPAng * sizeof(SFDKGeoParamPerView)));
            YK_CUDA_CHECK(cudaMalloc(&d_vol, vol_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_view_in, view_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_view_pw, view_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_view_flt, view_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_padded, (size_t)dims.iPV * paddedN * sizeof(float)));

            YK_CUDA_CHECK(cudaMemcpyAsync(d_geo, h_geo.data(),
                dims.iPAng * sizeof(SConeProjectionVec), cudaMemcpyHostToDevice, stream));
            YK_CUDA_CHECK(cudaMemcpyAsync(d_gv, h_gv.data(),
                dims.iPAng * sizeof(SFDKGeoParamPerView), cudaMemcpyHostToDevice, stream));
            YK_CUDA_CHECK(cudaMemsetAsync(d_vol, 0, vol_elems * sizeof(float), stream));

            // geo/gv 上传后立即预计算
            launchPrecomputeCoeffs(d_geo, d_gv, d_coeffs, dims.iPAng, stream);

            chunk.init(Kchunk, dims.iPU, dims.iPV);
        }

        void destroy() {
            chunk.destroy();
            auto freePtr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            freePtr(d_coeffs);
            freePtr(d_geo);
            freePtr(d_gv);
            freePtr(d_vol);
            freePtr(d_view_in);
            freePtr(d_view_pw);
            freePtr(d_view_flt);
            freePtr(d_padded);
        }

        ~FdkGpuContext() { destroy(); }
    };

} // namespace YK