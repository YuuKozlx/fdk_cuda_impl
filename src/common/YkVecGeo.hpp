#pragma once

#include <string>
#include <vector>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"
#include "../global/YkLog.h"
#include "../global/YkMacro.hpp"
#include "../util/YkVecOperation.hpp"
#ifndef __CUDACC__
#include <util/fmt/format.h>
#endif

// 内部几何类型与诊断格式化工具。几何构造必须显式包含对应 builder，
// kernel 和算法数据结构不通过本文件间接依赖前端构造逻辑。
namespace YK {

#ifndef __CUDACC__
inline std::string fmt_f4(const float4& value)
{
    return fmt::format("[{:.4f}, {:.4f}, {:.4f}, {:.4f}]",
        value.x, value.y, value.z, value.w);
}

inline std::string fmt_f3(const float3& value)
{
    return fmt::format("[{:.4f}, {:.4f}, {:.4f}]", value.x, value.y, value.z);
}

inline void print_proj_geom(const SConeProjGeomVec& geometry)
{
    YK_LOGI(
        "ProjGeom:\n"
        "  src   = {}\n"
        "  detS  = {}\n"
        "  detU  = {}\n"
        "  detV  = {}\n"
        "  angle = {:.4f}",
        fmt_f4(geometry.src), fmt_f4(geometry.detS),
        fmt_f4(geometry.detU), fmt_f4(geometry.detV), geometry.angle.x);
}

inline void print_proj_geom(const std::vector<SConeProjGeomVec>& geometry)
{
    for (const auto& view : geometry) print_proj_geom(view);
}
#endif

} // namespace YK
