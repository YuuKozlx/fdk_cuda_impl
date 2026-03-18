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

    } // namespace DataObj
} // namespace YK


namespace YK {
    namespace DataObj {

        class DataController
        {
        public:
            explicit DataController(Mem::MemoryController& mem)
                : mem_(mem) {
            }

            ////////////////////////////////////////////////////////////
            /// ------------------- CPU Volume -----------------------
            ////////////////////////////////////////////////////////////

            template<typename T>
            CpuVolume<T> createCpuVolume(int nx, int ny, int nz)
            {
                auto buf = mem_.allocateCpu3D<T>(nx, ny, nz);
                return CpuVolume<T>(std::move(buf));
            }

            template<typename T>
            CpuVolumeBorrowed<T> wrapCpuVolume(T* externalPtr, int nx, int ny, int nz)
            {
                Mem::CpuBuffer3DBorrowed<T> buf(externalPtr, nx, ny, nz); // borrowed
                return CpuVolumeBorrowed<T>(std::move(buf));
            }

            ////////////////////////////////////////////////////////////
            /// ------------------- GPU Volume -----------------------
            ////////////////////////////////////////////////////////////

            template<typename T>
            GpuVolume<T> createGpuVolume(int nx, int ny, int nz, int device = 0)
            {
                auto buf = mem_.template allocateDevice3D<T>(nx, ny, nz, device);
                return GpuVolume<T>(std::move(buf));
            }

            ////////////////////////////////////////////////////////////
            /// ------------------- CPU Projection -------------------
            ////////////////////////////////////////////////////////////

            template<typename T>
            CpuProjection<T> createCpuProjection(int nu, int nv, int views)
            {
                auto buf = mem_.allocateCpu3D<T>(nu, nv, views);
                return CpuProjection<T>(std::move(buf), views);
            }

            template<typename T>
            CpuProjectionBorrowed<T> wrapCpuProjection(T* externalPtr, int nu, int nv, int views)
            {
                YK::Mem::CpuBuffer3DBorrowed<T> buf(externalPtr, nu, nv, views); // borrowed
                return CpuProjectionBorrowed<T>(std::move(buf), views);
            }

            ////////////////////////////////////////////////////////////
            /// ------------------- GPU Projection -------------------
            ////////////////////////////////////////////////////////////

            template<typename T>
            GpuProjection<T> createGpuProjection(int nu, int nv, int views, int device = 0)
            {
                auto buf = mem_.template allocateDevice3D<T>(nu, nv, views, device);
                return GpuProjection<T>(std::move(buf), views);
            }

            ////////////////////////////////////////////////////////////
            /// ------------------- CPU Sinogram ---------------------
            ////////////////////////////////////////////////////////////

            template<typename T>
            CpuSinogram<T> createCpuSinogram(int nu, int views, int nv)
            {
                auto buf = mem_.allocateCpu3D<T>(nu, views, nv);
                return CpuSinogram<T>(std::move(buf));
            }

            template<typename T>
            CpuSinogramBorrowed<T> wrapCpuSinogram(T* externalPtr, int nu, int views, int nv)
            {
                YK::Mem::CpuBuffer3DBorrowed<T> buf(externalPtr, nu, views, nv); // borrowed
                return CpuSinogramBorrowed<T>(std::move(buf));
            }

            ////////////////////////////////////////////////////////////
            /// ------------------- GPU Sinogram ---------------------
            ////////////////////////////////////////////////////////////

            template<typename T>
            GpuSinogram<T> createGpuSinogram(int nu, int views, int nv, int device = 0)
            {
                auto buf = mem_.template allocateDevice3D<T>(nu, views, nv, device);
                return GpuSinogram<T>(std::move(buf));
            }

        private:
            YK::Mem::MemoryController& mem_;
        };

    } // namespace DataObj
} // namespace YK