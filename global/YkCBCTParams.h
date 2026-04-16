#pragma once
#include <vector>
#include "YkGlobals.h"

#define PI 3.14159265358979323846f

struct SCBCTParams {
    // projection params
    std::vector<float> angle_list; // in radians
    int iPU; // number of detectors in the U direction
    int iPV; // number of detectors in the V direction
    int iPAng; // number of projection angles
    int iPAngTotal; // total number of angles in the full scan (e.g. 360 for full scan, 180 for short scan)

    float du_mm = 1.0f;; // detector pixel size in U direction in mm
    float dv_mm = 1.0f;; // detector pixel size in V direction in mm
    float offsetU_mm = 0.0f; // detector offset in U direction in mm
    float offsetV_mm = 0.0f; // detector offset in V direction in mm
    float tiltn_angle_rad = 0.0f; // detector skew angle in radians (探测器平面绕中心射线的旋转角，右手规则，正值表示逆时针旋转)
    float tiltu_angle_rad = 0.0f; // detector slant angle in radians (探测器平面绕水平轴的旋转角，右手规则，正值表示前倾)
    float tiltv_angle_rad = 0.0f; // detector tilt angle in radians (探测器平面绕垂直轴的旋转角，右手规则，正值表示左倾)



    //Scan params
    float scan_range_rad = 2 * PI; // total scan range in radians (e.g. 2*PI for full scan, PI for short scan)
    float scan_start_angle_rad = 0.0f; // start angle of the scan in radians (e.g. 0 for full scan, -PI/2 for short scan)
    bool bShortScan = false; // whether it's a short scan (if true, Parker weighting will be applied)

    // geometry params
    float SID; // source-to-isocenter distance in mm 
    float SDD; // source-to-detector distance in mm


    // recon volume params
    int iVX; // number of voxels in the X direction
    int iVY; // number of voxels in the Y direction
    int iVZ; // number of voxels in the Z direction
    float vox_x_mm = 1.0f; // voxel size in mm
    float vox_y_mm = 1.0;
    float vox_z_mm = 1.0f;; // voxel size in Z direction in mm
    float vol_offset_x_mm = 0.0f; // volume center offset in X direction in mm (relative to isocenter)
    float vol_offset_y_mm = 0.0f; // volume center offset in Y direction in mm (relative to isocenter)
    float vol_offset_z_mm = 0.0f; // volume center offset in Z direction in mm (relative to isocenter)
    YK::SFilterKernelDesc desc = YK::SFilterKernelDesc(YK::EFilterKernel::RamLak); // filter kernel description
};

