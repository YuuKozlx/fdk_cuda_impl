#pragma once
#include <stdexcept>

namespace yk::spectral::primary_detail {
struct PrimaryLaunchLayout {
    int threads_per_pixel;
    int threads_per_block;
};

// Deliberately coarse policy, evaluated on the host. No runtime autotuning or
// image-size/GPU-specific lookup tables. Material split follows storage capacity.
inline PrimaryLaunchLayout selectPrimaryLaunchLayout(int materials,int energies,int rays) {
    if(materials<1||materials>64||energies<1||energies>256||rays<1)
        throw std::invalid_argument("invalid primary launch dimensions");
    if(energies>128)return {32,128};
    if(rays<=2||(materials<=32&&rays<=4))return {4,128};
    if(rays<=8)return {8,128};
    if(materials<=32&&rays>16)return {32,128};
    return {16,materials>32?64:128};
}
}
