#include "IntensityOps.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace yk::spectral::postprocess::intensity_detail {
void validateLocation(Location l)
{
    if (l != Location::Host && l != Location::Device)
        throw std::invalid_argument("invalid intensity stage location");
#ifndef YK_INTENSITY_CUDA
    if (l == Location::Device)
        throw std::runtime_error("intensity CUDA backend not built");
#endif
}
void validateCoefficients(HuLinearCoefficients c)
{
    if (!std::isfinite(c.scale) || c.scale <= 0 || !std::isfinite(c.offset))
        throw std::invalid_argument("linear HU coefficients require positive finite scale and finite offset");
}
void validateTargets(AirWaterTargets t)
{
    if (!std::isfinite(t.air_hu) || !std::isfinite(t.water_hu) || t.water_hu <= t.air_hu || !std::isfinite(t.minimum_measured_gap_hu) || t.minimum_measured_gap_hu <= 0)
        throw std::invalid_argument("invalid air/water calibration targets");
}
void validateStorage(const Volume& v, Location l)
{
    validateLocation(l);
    size_t n = 1;
    for (int d : v.grid.dimensions) {
        if (d <= 0 || n > std::numeric_limits<size_t>::max() / sizeof(float) / size_t(d))
            throw std::invalid_argument("invalid intensity volume dimensions");
        n *= size_t(d);
    }
    if (v.location != l || v.elements != n || !(l == Location::Host ? v.host_data : v.device_data))
        throw std::invalid_argument("invalid intensity volume storage");
}
void validateRoi(const CalibrationRoi& r, const Grid& g)
{
    for (int a = 0; a < 3; ++a)
        if (r.begin[a] < 0 || r.begin[a] >= r.end[a] || r.end[a] > g.dimensions[a])
            throw std::invalid_argument("calibration ROI is empty or outside the current volume grid");
}
size_t roiCount(const CalibrationRoi& r) { return size_t(r.end[0] - r.begin[0]) * size_t(r.end[1] - r.begin[1]) * size_t(r.end[2] - r.begin[2]); }
void applyLinear(Volume& v, Location l, HuLinearCoefficients c)
{
    validateStorage(v, l);
    validateCoefficients(c);
#ifdef YK_INTENSITY_CUDA
    if (l == Location::Device) {
        cudaLinear(v.device_data, v.elements, c);
        return;
    }
#endif
    // Check the entire output before in-place mutation, including overflow.
    for (size_t i = 0; i < v.elements; ++i) {
        double out = c.scale * double(v.host_data[i]) + c.offset;
        if (!std::isfinite(v.host_data[i]) || !std::isfinite(out) || std::abs(out) > std::numeric_limits<float>::max())
            throw std::runtime_error("nonfinite or overflowing intensity conversion");
    }
    for (size_t i = 0; i < v.elements; ++i)
        v.host_data[i] = float(c.scale * double(v.host_data[i]) + c.offset);
}
double meanRoi(const Volume& v, Location l, const CalibrationRoi& roi)
{
    validateStorage(v, l);
    validateRoi(roi, v.grid);
#ifdef YK_INTENSITY_CUDA
    if (l == Location::Device)
        return cudaMeanRoi(v.device_data, v.grid, roi);
#endif
    double sum = 0;
    const auto& n = v.grid.dimensions;
    for (int z = roi.begin[2]; z < roi.end[2]; ++z)
        for (int y = roi.begin[1]; y < roi.end[1]; ++y)
            for (int x = roi.begin[0]; x < roi.end[0]; ++x) {
                const float value = v.host_data[(size_t(z) * n[1] + y) * n[0] + x];
                if (!std::isfinite(value))
                    throw std::runtime_error("nonfinite calibration ROI value");
                sum += value;
            }
    return sum / double(roiCount(roi));
}
void classify(Volume& v, Location l, const HuThresholdSegmentation& s)
{
    validateStorage(v, l);
#ifdef YK_INTENSITY_CUDA
    if (l == Location::Device) {
        cudaClassify(v.device_data, v.elements, s);
        return;
    }
#endif
    for (size_t i = 0; i < v.elements; ++i)
        if (!std::isfinite(v.host_data[i]))
            throw std::runtime_error("nonfinite HU segmentation input");
    for (size_t i = 0; i < v.elements; ++i) {
        const size_t k = size_t(std::upper_bound(s.thresholds_hu.begin(), s.thresholds_hu.end(), double(v.host_data[i])) - s.thresholds_hu.begin());
        v.host_data[i] = float(s.classes[k].value);
    }
}
}
