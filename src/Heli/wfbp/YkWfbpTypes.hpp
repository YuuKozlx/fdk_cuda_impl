#pragma once

#include <cuda_runtime.h>

#include "Filter/YkCreateFilterKernel.cuh"
#include "Heli/YkHeliCTParams.h"
#include "global/YkGlobals.h"

namespace YK { namespace Helical { namespace Wfbp {

enum class EInputDetector {
    FlatPanel,
    EquiangularArc
};

// FreeCT_wFBP adaptation configuration. This implementation supports no FFS
// and uniformly spaced source angles. Flat input is resampled to an internal
// equiangular arc before entering the original FreeCT-style rebinning path.
struct Config {
    EInputDetector input_detector = EInputDetector::FlatPanel;
    int channel_oversampling = 2;
    float redundancy_flat = 0.6f;
    float angle_tolerance = 1e-3f;
    // Required for EquiangularArc input. FlatPanel derives the virtual arc
    // increment from du_mm and SDD.
    float arc_channel_angle_step_rad = 0.f;
    // Principal-ray channel of an arc input. A negative value selects the
    // geometric center channel. This is deliberately independent of flat du.
    float arc_principal_channel = -1.f;
    SFilterKernelDesc filter = SFilterKernelDesc::RamLak();
    SKernelLaunchPolicy launch = {};
};

struct Geometry {
    int input_channels = 0;
    int output_channels = 0;
    int rows = 0;
    int views = 0;
    int views_per_turn = 0;
    float first_angle = 0.f;
    float angle_step = 0.f;
    float central_channel = 0.f;
    float parallel_center = 0.f;
    float fan_angle_step = 0.f;
    float parallel_spacing = 0.f;
    float cone_half_angle = 0.f;
    float sid = 0.f;
    float sdd = 0.f;
    float pitch = 0.f;
    float start_z = 0.f;
};

} } } // namespace YK::Helical::Wfbp
