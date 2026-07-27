#pragma once

#include <vector>

#include "Heli/YkHelicalGeo.hpp"
#include "Heli/YkHeliCTParams.h"
#include "Iter/YkParallelPwlsReconstructor.hpp"

namespace YK::Helical {

// 兼容旧接口。通用实现现位于 Iter::ParallelPwlsReconstructor；此包装仅
// 保留 SHeliCTParam 便捷入口和既有 IcdConfig/IcdReconstructor 名称。
using IcdConfig = Iter::ParallelPwlsConfig;

class IcdReconstructor : public Iter::ParallelPwlsReconstructor {
public:
    using Iter::ParallelPwlsReconstructor::prepare;

    bool prepare(const SHeliCTParam& params, const IcdConfig& config,
        cudaStream_t stream, int device_id = 0)
    {
        std::vector<SConeProjGeomVec> geometry;
        build_helical_vec_geometry(geometry, params);
        return Iter::ParallelPwlsReconstructor::prepare(
            toCbctParams_(params), geometry, config, stream, device_id);
    }

private:
    static SCBCTParams toCbctParams_(const SHeliCTParam& h)
    {
        SCBCTParams p{};
        p.angle_list = h.angle_list;
        p.iPU = h.iPU; p.iPV = h.iPV;
        p.iPAng = static_cast<int>(h.angle_list.size());
        p.iPAngTotal = p.iPAng;
        p.du_mm = h.du_mm; p.dv_mm = h.dv_mm;
        p.offsetU_mm = h.offsetU_mm; p.offsetV_mm = h.offsetV_mm;
        p.tiltu_angle_rad = h.tiltu_angle_rad;
        p.tiltn_angle_rad = h.tiltn_angle_rad;
        p.tiltv_angle_rad = h.tiltv_angle_rad;
        p.SID = h.SID; p.SDD = h.SDD;
        p.iVX = h.iVX; p.iVY = h.iVY; p.iVZ = h.iVZ;
        p.vox_x_mm = h.vox_x_mm; p.vox_y_mm = h.vox_y_mm;
        p.vox_z_mm = h.vox_z_mm;
        p.vol_offset_x_mm = h.vol_offset_x_mm;
        p.vol_offset_y_mm = h.vol_offset_y_mm;
        p.vol_offset_z_mm = h.vol_offset_z_mm;
        p.scan_start_angle_rad = h.angle_list.empty() ? 0.f : h.angle_list.front();
        p.scan_range_rad = h.angle_list.size() < 2 ? 0.f :
            h.angle_list.back() - h.angle_list.front();
        return p;
    }
};

} // namespace YK::Helical
