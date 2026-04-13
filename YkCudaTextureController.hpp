#pragma once
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <cuda_runtime.h>
#include "YkMem3d.hpp"   // DevicePitchedBuffer3D, DeviceLinearBuffer3D
#include "YkGlobals.h"      // YK_CUDA_CHECK

namespace YK {
    namespace Mem {

        // ============================================================
        // TextureController
        // 管理 cudaTextureObject_t 的创建与销毁
        //
        // 支持类型：
        //   float     投影数据，支持 FilterModeLinear
        //   float2    复数/频域数据，支持 FilterModeLinear
        //   uint8_t   8位原始探测器数据，仅支持 FilterModePoint
        //   uint16_t  12/16位原始探测器数据，仅支持 FilterModePoint
        //   int       标签图/分割mask，仅支持 FilterModePoint
        //
        // 注意：
        //   cudaFilterModeLinear 只支持浮点类型，整数类型传入时构造期抛异常
        //   normalizedCoords 固定为 0（非归一化坐标），与 tex2D(tex, u+0.5, v+0.5) 配套
        //
        // 生命周期：
        //   createTex* 返回的 cudaTextureObject_t 必须通过 destroyTex / destroyTexBatch 释放
        //   TextureController 本身无状态，可以栈上构造，随用随建
        // ============================================================

        class TextureController {
        public:

            // ----------------------------------------------------------------
            // 从线性内存的单个 slice 创建 2D 纹理
            // 适用于 chunk_flt 这类连续分配的投影 buffer
            // slice 指针由调用方计算：basePtr + i * Nu * Nv
            // ----------------------------------------------------------------
            template<typename T>
            cudaTextureObject_t createTex2DLinear(
                T* devPtr, int Nu, int Nv,
                cudaTextureFilterMode  filter = cudaFilterModeLinear,
                cudaTextureAddressMode addr = cudaAddressModeClamp) const
            {
                check_filter_type<T>(filter);

                cudaResourceDesc res{};
                res.resType = cudaResourceTypePitch2D;
                res.res.pitch2D.devPtr = devPtr;
                res.res.pitch2D.desc = cudaCreateChannelDesc<T>();
                res.res.pitch2D.width = Nu;
                res.res.pitch2D.height = Nv;
                res.res.pitch2D.pitchInBytes = Nu * sizeof(T);  // 线性，pitch固定

                return createTexObj_(res, filter, addr);
            }

            // ----------------------------------------------------------------
            // 从线性内存批量创建 2D 纹理（K个视角）
            // ----------------------------------------------------------------
            template<typename T>
            std::vector<cudaTextureObject_t> createTex2DLinearBatch(
                T* basePtr, int Nu, int Nv, int K,
                cudaTextureFilterMode  filter = cudaFilterModeLinear,
                cudaTextureAddressMode addr = cudaAddressModeClamp) const
            {
                check_filter_type<T>(filter);

                std::vector<cudaTextureObject_t> texs(K);
                for (int i = 0; i < K; ++i) {
                    T* slicePtr = basePtr + (size_t)i * Nu * Nv;
                    texs[i] = createTex2DLinear(slicePtr, Nu, Nv, filter, addr);
                }
                return texs;
            }

            // ----------------------------------------------------------------
            // 从 DevicePitchedBuffer3D 的单个 slice 创建 2D 纹理
            // pitch 自动从 buffer 取，slice 偏移自动计算，不会填错
            // buf.nz = K（视角数），buf.nx = Nu，buf.ny = Nv
            // ----------------------------------------------------------------
            template<typename T>
            cudaTextureObject_t createTex2DFromSlice(
                const DevicePitchedBuffer3D<T>& buf, int sliceIdx,
                cudaTextureFilterMode  filter = cudaFilterModeLinear,
                cudaTextureAddressMode addr = cudaAddressModeClamp) const
            {
                check_filter_type<T>(filter);

                if (sliceIdx < 0 || sliceIdx >= buf.shape().nz)
                    throw std::out_of_range("createTex2DFromSlice: sliceIdx out of range");

                // pitched内存slice偏移必须用pitch，不能用nx*sizeof(T)
                const size_t sliceBytes = buf.pitch() * buf.shape().ny;
                auto* slicePtr = reinterpret_cast<T*>(
                    reinterpret_cast<unsigned char*>(buf.data()) + sliceIdx * sliceBytes);

                cudaResourceDesc res{};
                res.resType = cudaResourceTypePitch2D;
                res.res.pitch2D.devPtr = slicePtr;
                res.res.pitch2D.desc = cudaCreateChannelDesc<T>();
                res.res.pitch2D.width = buf.shape().nx;
                res.res.pitch2D.height = buf.shape().ny;
                res.res.pitch2D.pitchInBytes = buf.pitch();     // 自动取，不会填错

                return createTexObj_(res, filter, addr);
            }

            // ----------------------------------------------------------------
            // 从 DevicePitchedBuffer3D 批量创建 2D 纹理（K = buf.nz 个视角）
            // ----------------------------------------------------------------
            template<typename T>
            std::vector<cudaTextureObject_t> createTex2DBatch(
                const DevicePitchedBuffer3D<T>& buf,
                cudaTextureFilterMode  filter = cudaFilterModeLinear,
                cudaTextureAddressMode addr = cudaAddressModeClamp) const
            {
                check_filter_type<T>(filter);

                const int K = buf.shape().nz;
                std::vector<cudaTextureObject_t> texs(K);
                for (int i = 0; i < K; ++i)
                    texs[i] = createTex2DFromSlice(buf, i, filter, addr);
                return texs;
            }

            // ----------------------------------------------------------------
            // 销毁
            // ----------------------------------------------------------------
            void destroyTex(cudaTextureObject_t tex) const
            {
                if (tex != 0)
                    YK_CUDA_CHECK(cudaDestroyTextureObject(tex));
            }

            void destroyTexBatch(std::vector<cudaTextureObject_t>& texs) const
            {
                for (auto& t : texs)
                    destroyTex(t);
                texs.clear();
            }

        private:

            // ----------------------------------------------------------------
            // 公共创建逻辑
            // ----------------------------------------------------------------
            cudaTextureObject_t createTexObj_(
                const cudaResourceDesc& res,
                cudaTextureFilterMode    filter,
                cudaTextureAddressMode   addr) const
            {
                cudaTextureDesc tex{};
                tex.addressMode[0] = addr;
                tex.addressMode[1] = addr;
                tex.filterMode = filter;
                tex.readMode = cudaReadModeElementType;
                tex.normalizedCoords = 0;   // 非归一化坐标，配合 tex2D(tex, u+0.5f, v+0.5f)

                cudaTextureObject_t obj = 0;
                YK_CUDA_CHECK(cudaCreateTextureObject(&obj, &res, &tex, nullptr));
                return obj;
            }

            // ----------------------------------------------------------------
            // 编译期 + 运行时检查：整数类型不支持 FilterModeLinear
            // ----------------------------------------------------------------
            template<typename T>
            static void check_filter_type(cudaTextureFilterMode filter)
            {
                if (filter == cudaFilterModeLinear &&
                    !std::is_floating_point<T>::value)
                {
                    throw std::invalid_argument(
                        "cudaFilterModeLinear requires floating point type (float/float2), "
                        "use cudaFilterModePoint for integer types (uint8/uint16/int)");
                }
            }
        };

    } // namespace Mem
} // namespace YK