#pragma once

#include <vector>

#include "common/cuda/operators/YkOperatorKernelTypes.cuh"
#include "global/YkMem3d.hpp"

namespace YK::CudaOp {

// Flat FP/BP 共用的几何缓存。worldGeometry 是调用方提供的唯一几何真源；
// voxelGeometry 只是在指定体积坐标系下生成的 kernel 派生缓存，不能被修改
// 后再作为外部几何传播。缓存对象的生命周期同时约束 host/device 两份数据。
class ProjectionGeometryCache {
public:
    void init(const std::vector<SConeProjGeomVec>& world_geometry,
        const SVolGeom& volume_geometry, int device_id = 0)
    {
        prepare(world_geometry, volume_geometry, device_id);
    }

    void prepare(const std::vector<SConeProjGeomVec>& world_geometry,
        const SVolGeom& volume_geometry, int device_id = 0)
    {
        world_geometry_ = world_geometry;
        voxel_geometry_ = normalizeToVoxelBatch(world_geometry_, volume_geometry);
        volume_geometry_ = volume_geometry;

        Mem::PodDataController memory;
        device_world_geometry_ = memory.allocateAndUpload(
            world_geometry_, device_id);
        device_voxel_geometry_ = memory.allocateAndUpload(
            voxel_geometry_, device_id);
    }

    SConeProjGeomVec* deviceWorldGeometry() const
    { return device_world_geometry_.data(); }

    SConeProjGeomVec* deviceVoxelGeometry() const
    { return device_voxel_geometry_.data(); }

    const SConeProjGeomVec* hostWorldGeometry() const
    { return world_geometry_.data(); }

    const SConeProjGeomVec* hostVoxelGeometry() const
    { return voxel_geometry_.data(); }

    const std::vector<SConeProjGeomVec>& worldGeometry() const
    { return world_geometry_; }

    const std::vector<SConeProjGeomVec>& voxelGeometry() const
    { return voxel_geometry_; }

    const SVolGeom& volumeGeometry() const { return volume_geometry_; }

    // 临时兼容旧调用点；新代码应使用上面的完整语义名称。
    SConeProjGeomVec* d_views() const { return deviceWorldGeometry(); }
    SConeProjGeomVec* d_views_world() const { return deviceWorldGeometry(); }
    SConeProjGeomVec* d_views_vox() const { return deviceVoxelGeometry(); }
    const SConeProjGeomVec* h_views() const { return hostWorldGeometry(); }
    const SConeProjGeomVec* h_views_world() const { return hostWorldGeometry(); }
    const SConeProjGeomVec* h_views_vox() const { return hostVoxelGeometry(); }
    const std::vector<SConeProjGeomVec>& h_views_vec() const
    { return worldGeometry(); }
    const std::vector<SConeProjGeomVec>& h_views_world_vec() const
    { return worldGeometry(); }
    const std::vector<SConeProjGeomVec>& h_views_vox_vec() const
    { return voxelGeometry(); }
    const SVolGeom& h_volgeom() const { return volumeGeometry(); }

private:
    std::vector<SConeProjGeomVec> world_geometry_;
    std::vector<SConeProjGeomVec> voxel_geometry_;
    Mem::DeviceLinearBuffer<SConeProjGeomVec> device_world_geometry_;
    Mem::DeviceLinearBuffer<SConeProjGeomVec> device_voxel_geometry_;
    SVolGeom volume_geometry_{};
};

} // namespace YK::CudaOp
