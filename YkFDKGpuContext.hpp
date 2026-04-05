#pragma once
#include <vector>
#include "YkFDKChunkBuffer.hpp"
#include "YkFDKPrecompute.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkGlobals.h"
#include "YkVecGeo.hpp"

namespace YK {

    struct FdkGpuContext {
        SConeProjectionVec* d_geo = nullptr;  // [iPAng]
        SFDKGeoParamPerView* d_gv = nullptr;  // [iPAng]
        FdkAffineCoeff* d_coeffs = nullptr;  // [iPAng]
        float* d_chunk_in = nullptr;  // [Kchunk*iPV*iPU]
        float* d_chunk_pw = nullptr;  // [Kchunk*iPV*iPU]
        float* d_chunk_flt = nullptr;  // [Kchunk*iPV*iPU]

        ChunkBuffer chunk;

        FdkGpuContext() = default;
        FdkGpuContext(const FdkGpuContext&) = delete;
        FdkGpuContext& operator=(const FdkGpuContext&) = delete;

        void init(
            const SDimensions3D& dims,
            const std::vector<SConeProjectionVec>& h_geo,
            const std::vector<SFDKGeoParamPerView>& h_gv,
            int          Kchunk,
            cudaStream_t stream)
        {
            const size_t view_elems = (size_t)dims.iPU * dims.iPV;
            const size_t chunk_elems = (size_t)Kchunk * view_elems;


            // geo / gv
            YK_CUDA_CHECK(cudaMalloc(&d_geo, dims.iPAng * sizeof(SConeProjectionVec)));
            YK_CUDA_CHECK(cudaMalloc(&d_gv, dims.iPAng * sizeof(SFDKGeoParamPerView)));
            YK_CUDA_CHECK(cudaMemcpyAsync(d_geo, h_geo.data(),
                dims.iPAng * sizeof(SConeProjectionVec),
                cudaMemcpyHostToDevice, stream));
            YK_CUDA_CHECK(cudaMemcpyAsync(d_gv, h_gv.data(),
                dims.iPAng * sizeof(SFDKGeoParamPerView),
                cudaMemcpyHostToDevice, stream));

            // 预计算仿射系数（geo/gv 上传后立即触发）
            YK_CUDA_CHECK(cudaMalloc(&d_coeffs, dims.iPAng * sizeof(FdkAffineCoeff)));
            launchPrecomputeCoeffs(d_geo, d_gv, d_coeffs, dims.iPAng, stream);

            // chunk 级投影缓冲
            YK_CUDA_CHECK(cudaMalloc(&d_chunk_in, chunk_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_chunk_pw, chunk_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_chunk_flt, chunk_elems * sizeof(float)));

            chunk.init(Kchunk, dims.iPU, dims.iPV);
        }

        void destroy()
        {
            chunk.destroy();
            auto free_ = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            free_(d_geo);
            free_(d_gv);
            free_(d_coeffs);
            free_(d_chunk_in);
            free_(d_chunk_pw);
            free_(d_chunk_flt);
        }

        ~FdkGpuContext() { destroy(); }
    };

} // namespace YK