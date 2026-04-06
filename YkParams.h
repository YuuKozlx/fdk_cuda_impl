#pragma once
#include <vector>
#include "YkGlobals.h"


struct SCBCTParams {
    // projection params
    std::vector<float> angle_list; // in radians
    int iPU; // number of detectors in the U direction
    int iPV; // number of detectors in the V direction
    int iPAng; // number of projection angles
    float du_mm = 1.0f;; // detector pixel size in U direction in mm
    float dv_mm = 1.0f;; // detector pixel size in V direction in mm
    float offsetU_mm = 0.0f; // detector offset in U direction in mm
    float offsetV_mm = 0.0f; // detector offset in V direction in mm
    float skew_angle_rad = 0.0f; // detector skew angle in radians (探测器平面绕中心射线的旋转角，右手规则，正值表示逆时针旋转)
    float slant_angle_rad = 0.0f; // detector slant angle in radians (探测器平面绕水平轴的旋转角，右手规则，正值表示前倾)
    float tilt_angle_rad = 0.0f; // detector tilt angle in radians (探测器平面绕垂直轴的旋转角，右手规则，正值表示左倾)


    // geometry params
    float SID; // source-to-isocenter distance in mm 
    float SDD; // source-to-detector distance in mm


    // recon volume params
    int iVX; // number of voxels in the X direction
    int iVY; // number of voxels in the Y direction
    int iVZ; // number of voxels in the Z direction
    float vox_xy_mm = 1.0f; // voxel size in mm
    float vox_z_mm = 1.0f;; // voxel size in Z direction in mm
    float vol_offset_x_mm = 0.0f; // volume center offset in X direction in mm (relative to isocenter)
    float vol_offset_y_mm = 0.0f; // volume center offset in Y direction in mm (relative to isocenter)
    float vol_offset_z_mm = 0.0f; // volume center offset in Z direction in mm (relative to isocenter)
    SFilterKernelDesc desc = SFilterKernelDesc(EFilterKernel::RamLak); // filter kernel description
};

