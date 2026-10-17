#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
namespace yk::visualization {
inline double attenuationToHu(double mu_mm_inv,double water_mm_inv){
    if(!std::isfinite(water_mm_inv)||water_mm_inv<=0)throw std::invalid_argument("invalid CT water reference");
    return 1000.0*(mu_mm_inv/water_mm_inv-1.0);
}
struct FloatVolume {
    std::array<int,3> dimensions{};
    std::array<double,3> spacing{1,1,1};
    std::vector<float> values; // [z][y][x], x fastest, original reconstruction grid.
    std::array<double,3> origin{}; // World position of voxel (0,0,0), mm.
    std::array<double,9> direction{1,0,0,0,1,0,0,0,1}; // Row-major grid axes -> world.
};
struct FloatSlice {
    int width=0,height=0;
    std::array<double,2> spacing{1,1};
    std::vector<float> values;
};
inline void validateFloatVolume(const FloatVolume& volume){
    std::size_t count=1;
    for(int i=0;i<3;++i){if(volume.dimensions[i]<=0 || !std::isfinite(volume.spacing[i]) || volume.spacing[i]<=0 ||
        count>std::numeric_limits<std::size_t>::max()/std::size_t(volume.dimensions[i]))throw std::runtime_error("invalid reconstruction grid");
        count*=volume.dimensions[i];}
    if(volume.values.size()!=count)throw std::runtime_error("reconstruction volume size mismatch");
    for(double v:volume.origin)if(!std::isfinite(v))throw std::runtime_error("nonfinite reconstruction origin");
    for(double v:volume.direction)if(!std::isfinite(v))throw std::runtime_error("nonfinite reconstruction direction");
    for(int a=0;a<3;++a)for(int b=0;b<3;++b){double dot=0;
        for(int r=0;r<3;++r)dot+=volume.direction[r*3+a]*volume.direction[r*3+b];
        if(std::abs(dot-(a==b?1.:0.))>1e-5)throw std::runtime_error("reconstruction direction must be orthonormal");}
    for(float value:volume.values)if(!std::isfinite(value))throw std::runtime_error("nonfinite reconstruction value");
}
// Axis is the fixed coordinate: 2=XY axial, 1=XZ coronal, 0=YZ sagittal.
inline FloatSlice extractFloatSlice(const FloatVolume& volume,int axis,int index){
    if(axis<0||axis>2||index<0||index>=volume.dimensions[axis])throw std::out_of_range("invalid reconstruction slice");
    const int nx=volume.dimensions[0],ny=volume.dimensions[1];
    FloatSlice slice;slice.width=axis==0?ny:nx;slice.height=axis==2?ny:volume.dimensions[2];
    slice.spacing={volume.spacing[axis==0?1:0],volume.spacing[axis==2?1:2]};
    slice.values.resize(std::size_t(slice.width)*slice.height);
    for(int v=0;v<slice.height;++v)for(int u=0;u<slice.width;++u){
        const int x=axis==0?index:u,y=axis==1?index:axis==0?u:v,z=axis==2?index:v;
        slice.values[std::size_t(v)*slice.width+u]=volume.values[(std::size_t(z)*ny+y)*nx+x];}
    return slice;
}
inline std::array<double,2> automaticFloatWindow(const FloatVolume& volume){
    validateFloatVolume(volume);
    const auto bounds=std::minmax_element(volume.values.begin(),volume.values.end());
    const double low=*bounds.first,high=*bounds.second;
    if(high<=low)return {low,low+1};
    std::array<std::size_t,2048> histogram{};
    for(float value:volume.values){const int bin=std::clamp(int((value-low)/(high-low)*2047),0,2047);++histogram[bin];}
    auto percentile=[&](double quantile){const auto target=std::size_t(std::ceil(quantile*volume.values.size()));std::size_t count=0;
        for(std::size_t i=0;i<histogram.size();++i){count+=histogram[i];if(count>=target)return low+(high-low)*i/2047.;}return high;};
    double black=percentile(.005),white=percentile(.995);
    if(white<=black)return {low,high};return {black,white};
}
}
