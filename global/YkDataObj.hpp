#pragma once

#include <string>
#include <vector>
#include <utility>

#include "YkMem3d.hpp"

namespace YK {
    namespace DataObj {
        using YK::Mem::Shape3D;

        ////////////////////////////////////////////////////////////
        /// DataKind
        ////////////////////////////////////////////////////////////

        enum class DataKind
        {
            Unknown,
            Volume,
            Projection,
            Sinogram
        };

        ////////////////////////////////////////////////////////////
        /// Base Data Object
        ////////////////////////////////////////////////////////////

        class DataObjectBase
        {
        public:

            virtual ~DataObjectBase() = default;

            virtual DataKind kind() const noexcept = 0;

            virtual Shape3D shape() const noexcept = 0;

            virtual std::string name() const
            {
                return "DataObject";
            }
        };

        ////////////////////////////////////////////////////////////
        /// Volume Data
        ///
        /// shape:
        /// nx × ny × nz
        ////////////////////////////////////////////////////////////

        template<typename T, typename Buffer>
        class VolumeData : public DataObjectBase
        {
        public:

            using ValueType = T;
            using BufferType = Buffer;

            // 接受右值引用，直接移动
            explicit VolumeData(Buffer&& buf)
                : buffer_(std::move(buf))
            {
            }

            DataKind kind() const noexcept override
            {
                return DataKind::Volume;
            }

            Shape3D shape() const noexcept override
            {
                return buffer_.shape();
            }

            auto view()
            {
                return buffer_.view();
            }

            auto cview() const
            {
                return buffer_.cview();
            }

            T* data()
            {
                return buffer_.data();
            }

            const T* data() const
            {
                return buffer_.data();
            }

            Buffer& buffer() noexcept
            {
                return buffer_;
            }

            const Buffer& buffer() const noexcept
            {
                return buffer_;
            }

            std::string name() const override
            {
                return "VolumeData";
            }

        private:

            Buffer buffer_;
        };

        ////////////////////////////////////////////////////////////
        /// Projection Data
        ///
        /// shape:
        /// detector_u × detector_v × views
        ////////////////////////////////////////////////////////////

        template<typename T, typename Buffer>
        class ProjectionData : public DataObjectBase
        {
        public:

            using ValueType = T;
            using BufferType = Buffer;

            explicit ProjectionData(Buffer&& buf, int views)
                : buffer_(std::move(buf)),
                views_(views)
            {
            }


            DataKind kind() const noexcept override
            {
                return DataKind::Projection;
            }

            Shape3D shape() const noexcept override
            {
                return buffer_.shape();
            }

            int views() const noexcept
            {
                return views_;
            }

            auto view()
            {
                return buffer_.view();
            }

            auto cview() const
            {
                return buffer_.cview();
            }

            Buffer& buffer() noexcept
            {
                return buffer_;
            }

            const Buffer& buffer() const noexcept
            {
                return buffer_;
            }

            std::string name() const override
            {
                return "ProjectionData";
            }

        private:

            Buffer buffer_;

            int views_ = 0;
        };

        ////////////////////////////////////////////////////////////
        /// Sinogram Data
        ///
        /// shape:
        /// detector_u × views × detector_v
        ////////////////////////////////////////////////////////////

        template<typename T, typename Buffer>
        class SinogramData : public DataObjectBase
        {
        public:

            using ValueType = T;
            using BufferType = Buffer;

            explicit SinogramData(Buffer&& buf)
                : buffer_(std::move(buf))
            {
            }


            DataKind kind() const noexcept override
            {
                return DataKind::Sinogram;
            }

            Shape3D shape() const noexcept override
            {
                return buffer_.shape();
            }

            auto view()
            {
                return buffer_.view();
            }

            auto cview() const
            {
                return buffer_.cview();
            }

            Buffer& buffer() noexcept
            {
                return buffer_;
            }

            const Buffer& buffer() const noexcept
            {
                return buffer_;
            }

            std::string name() const override
            {
                return "SinogramData";
            }

        private:

            Buffer buffer_;
        };


        ////////////////////////////////////////////////////////////
        /// Convenient Type Aliases
        ////////////////////////////////////////////////////////////

        template<typename T>
        using CpuVolume =
            VolumeData<T, Mem::CpuBuffer3D<T>>;

        template<typename T>
        using GpuVolume =
            VolumeData<T, Mem::DeviceBuffer3D<T>>;

        template<typename T>
        using CpuProjection =
            ProjectionData<T, Mem::CpuBuffer3D<T>>;

        template<typename T>
        using GpuProjection =
            ProjectionData<T, Mem::DeviceBuffer3D<T>>;

        template<typename T>
        using CpuSinogram =
            SinogramData<T, Mem::CpuBuffer3D<T>>;

        template<typename T>
        using GpuSinogram =
            SinogramData<T, Mem::DeviceBuffer3D<T>>;

        template<typename T>
        using CpuVolumeBorrowed = VolumeData<T, Mem::CpuBuffer3DBorrowed<T>>;

        template<typename T>
        using CpuProjectionBorrowed = ProjectionData<T, Mem::CpuBuffer3DBorrowed<T>>;

        template<typename T>
        using CpuSinogramBorrowed = SinogramData<T, Mem::CpuBuffer3DBorrowed<T>>;

        // Pinned
        template<typename T>
        using PinnedVolume = VolumeData<T, Mem::HostPinnedBuffer3D<T>>;

        template<typename T>
        using PinnedProjection = ProjectionData<T, Mem::HostPinnedBuffer3D<T>>;

        template<typename T>
        using PinnedSinogram = SinogramData<T, Mem::HostPinnedBuffer3D<T>>;

        // GPU Borrowed
        template<typename T>
        using GpuVolumeBorrowed = VolumeData<T, Mem::DeviceBuffer3DBorrowed<T>>;

        template<typename T>
        using GpuProjectionBorrowed = ProjectionData<T, Mem::DeviceBuffer3DBorrowed<T>>;

        template<typename T>
        using GpuSinogramBorrowed = SinogramData<T, Mem::DeviceBuffer3DBorrowed<T>>;

    } // namespace DataObj
} // namespace YK


namespace YK {
    namespace DataObj {

        class DataController {
        public:
            explicit DataController(Mem::MemoryController& mem) : mem_(mem) {}

            ////////////////////////////////////////////////////////////
            // CPU Volume
            ////////////////////////////////////////////////////////////
            template<typename T>
            CpuVolume<T> createCpuVolume(int nx, int ny, int nz)
            {
                return CpuVolume<T>(mem_.allocateCpu3D<T>(nx, ny, nz));
            }

            template<typename T>
            CpuVolumeBorrowed<T> wrapCpuVolume(T* ptr, int nx, int ny, int nz)
            {
                return CpuVolumeBorrowed<T>(mem_.borrowCpu3D<T>(ptr, nx, ny, nz));
            }

            template<typename T>
            PinnedVolume<T> createPinnedVolume(int nx, int ny, int nz)
            {
                return PinnedVolume<T>(mem_.allocatePinnedCpu3D<T>(nx, ny, nz));
            }

            ////////////////////////////////////////////////////////////
            // GPU Volume
            ////////////////////////////////////////////////////////////
            template<typename T>
            GpuVolume<T> createGpuVolume(int nx, int ny, int nz,
                int device, cudaStream_t stream)
            {
                return GpuVolume<T>(mem_.allocateDevice3D<T>(nx, ny, nz, device));
            }

            template<typename T>
            GpuVolumeBorrowed<T> wrapGpuVolume(T* ptr, int nx, int ny, int nz,
                size_t pitchBytes, int device)
            {
                return GpuVolumeBorrowed<T>(
                    mem_.borrowDevice3D<T>(ptr, nx, ny, nz, pitchBytes, device));
            }

            template<typename T>
            GpuVolumeBorrowed<T> wrapGpuVolumeLinear(T* ptr, int nx, int ny, int nz,
                int device)
            {
                return GpuVolumeBorrowed<T>(
                    mem_.borrowDevice3DLinear<T>(ptr, nx, ny, nz, device));
            }

            ////////////////////////////////////////////////////////////
            // CPU Projection
            ////////////////////////////////////////////////////////////
            template<typename T>
            CpuProjection<T> createCpuProjection(int nu, int nv, int views)
            {
                return CpuProjection<T>(mem_.allocateCpu3D<T>(nu, nv, views), views);
            }

            template<typename T>
            CpuProjectionBorrowed<T> wrapCpuProjection(T* ptr, int nu, int nv, int views)
            {
                return CpuProjectionBorrowed<T>(mem_.borrowCpu3D<T>(ptr, nu, nv, views), views);
            }

            template<typename T>
            PinnedProjection<T> createPinnedProjection(int nu, int nv, int views)
            {
                return PinnedProjection<T>(mem_.allocatePinnedCpu3D<T>(nu, nv, views), views);
            }

            ////////////////////////////////////////////////////////////
            // GPU Projection
            ////////////////////////////////////////////////////////////
            template<typename T>
            GpuProjection<T> createGpuProjection(int nu, int nv, int views,
                int device, cudaStream_t stream)
            {
                return GpuProjection<T>(
                    mem_.allocateDevice3D<T>(nu, nv, views, device), views);
            }

            template<typename T>
            GpuProjectionBorrowed<T> wrapGpuProjection(T* ptr, int nu, int nv, int views,
                size_t pitchBytes, int device)
            {
                return GpuProjectionBorrowed<T>(
                    mem_.borrowDevice3D<T>(ptr, nu, nv, views, pitchBytes, device), views);
            }

            template<typename T>
            GpuProjectionBorrowed<T> wrapGpuProjectionLinear(T* ptr, int nu, int nv,
                int views, int device)
            {
                return GpuProjectionBorrowed<T>(
                    mem_.borrowDevice3DLinear<T>(ptr, nu, nv, views, device), views);
            }

            ////////////////////////////////////////////////////////////
            // CPU Sinogram
            ////////////////////////////////////////////////////////////
            template<typename T>
            CpuSinogram<T> createCpuSinogram(int nu, int views, int nv)
            {
                return CpuSinogram<T>(mem_.allocateCpu3D<T>(nu, views, nv));
            }

            template<typename T>
            CpuSinogramBorrowed<T> wrapCpuSinogram(T* ptr, int nu, int views, int nv)
            {
                return CpuSinogramBorrowed<T>(mem_.borrowCpu3D<T>(ptr, nu, views, nv));
            }

            template<typename T>
            PinnedSinogram<T> createPinnedSinogram(int nu, int views, int nv)
            {
                return PinnedSinogram<T>(mem_.allocatePinnedCpu3D<T>(nu, views, nv));
            }

            ////////////////////////////////////////////////////////////
            // GPU Sinogram
            ////////////////////////////////////////////////////////////
            template<typename T>
            GpuSinogram<T> createGpuSinogram(int nu, int views, int nv,
                int device, cudaStream_t stream)
            {
                return GpuSinogram<T>(
                    mem_.allocateDevice3D<T>(nu, views, nv, device));
            }

            template<typename T>
            GpuSinogramBorrowed<T> wrapGpuSinogram(T* ptr, int nu, int views, int nv,
                size_t pitchBytes, int device)
            {
                return GpuSinogramBorrowed<T>(
                    mem_.borrowDevice3D<T>(ptr, nu, views, nv, pitchBytes, device));
            }

            template<typename T>
            GpuSinogramBorrowed<T> wrapGpuSinogramLinear(T* ptr, int nu, int views,
                int nv, int device)
            {
                return GpuSinogramBorrowed<T>(
                    mem_.borrowDevice3DLinear<T>(ptr, nu, views, nv, device));
            }

        private:
            Mem::MemoryController& mem_;
        };

    } // namespace DataObj
} // namespace YK