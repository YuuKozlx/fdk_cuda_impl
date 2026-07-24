#pragma once

#include <algorithm>
#include <cufft.h>

#include "Filter/YkConv.hpp"
#include "Filter/YkFFT.hpp"
#include "Filter/YkCreateFilterKernel.cuh"
#include "Heli/wfbp/YkWfbpTypes.hpp"
#include "Heli/wfbp/kernels/YkWfbpLaunch.cuh"
#include "global/YkMem3d.hpp"

namespace YK { namespace Helical { namespace Wfbp {

class FlatToEquiangularArcProcessor {
public:
    bool prepare(const Geometry& geometry, float flat_du_mm, const Config& config)
    {
        geometry_ = geometry;
        flat_du_mm_ = flat_du_mm;
        policy_ = config.launch;
        return geometry_.views > 0 && geometry_.rows > 0 &&
            geometry_.input_channels > 1 && flat_du_mm_ > 0.f;
    }

    bool apply(const float* flat, float* arc, cudaStream_t stream) const
    {
        if (!flat || !arc || flat_du_mm_ <= 0.f) return false;
        detail::launch_flat_to_equiangular_arc(flat, arc, geometry_,
            flat_du_mm_, policy_, stream);
        return true;
    }

private:
    Geometry geometry_{};
    float flat_du_mm_ = 0.f;
    SKernelLaunchPolicy policy_{};
};

class RebinProcessor {
public:
    bool prepare(const Geometry& geometry, const Config& config)
    {
        geometry_ = geometry;
        policy_ = config.launch;
        return geometry_.views > 0 && geometry_.rows > 0 &&
            geometry_.input_channels > 1 && geometry_.output_channels > 1;
    }

    bool apply(const float* input, float* output, cudaStream_t stream) const
    {
        if (!input || !output || geometry_.views <= 0) return false;
        detail::launch_rebin(input, output, geometry_, policy_, stream);
        return true;
    }

private:
    Geometry geometry_{};
    SKernelLaunchPolicy policy_{};
};

class FilterProcessor {
public:
    ~FilterProcessor() { release(); }
    FilterProcessor() = default;
    FilterProcessor(const FilterProcessor&) = delete;
    FilterProcessor& operator=(const FilterProcessor&) = delete;

    bool prepare(const Geometry& geometry, const Config& config,
        cudaStream_t stream, int device_id)
    {
        release();
        geometry_ = geometry;
        policy_ = config.launch;
        stream_ = stream;
        padded_channels_ = 1;
        while (padded_channels_ < 2 * geometry_.output_channels)
            padded_channels_ <<= 1;
        complex_channels_ = padded_channels_ / 2 + 1;
        const int batch = geometry_.views * geometry_.rows;
        d_padded_ = memory_.allocateDevice3D<float>(padded_channels_,
            geometry_.rows, geometry_.views, device_id);
        d_complex_ = memory_.allocateDevice3D<cufftComplex>(complex_channels_,
            geometry_.rows, geometry_.views, device_id);
        d_weights_ = memory_.allocateDevice3D<float>(complex_channels_, 1, 1, device_id);
        if (!fft_.init(padded_channels_, batch, stream_)) return false;
        filter_builder_.prepare(padded_channels_, stream_);
        filter_builder_.build_weights(d_weights_.data(), config.filter,
            geometry_.parallel_spacing, true);
        prepared_ = true;
        return true;
    }

    bool apply(const float* input, float* output)
    {
        if (!prepared_ || !input || !output) return false;
        detail::launch_pad(input, d_padded_.data(), geometry_.output_channels,
            padded_channels_, geometry_.rows, geometry_.views, policy_, stream_);
        fft_.fft(d_padded_.data(), d_complex_.data());
        Filter::launch_pointwise_mul(d_complex_.data(), d_weights_.data(),
            complex_channels_, geometry_.views * geometry_.rows, stream_);
        fft_.ifft(d_complex_.data(), d_padded_.data());
        detail::launch_crop(d_padded_.data(), output, geometry_.output_channels,
            padded_channels_, geometry_.rows, geometry_.views, policy_, stream_);
        return true;
    }

    void release()
    {
        filter_builder_.release();
        fft_.release(false);
        d_padded_ = {};
        d_complex_ = {};
        d_weights_ = {};
        padded_channels_ = complex_channels_ = 0;
        prepared_ = false;
    }

private:
    Geometry geometry_{};
    SKernelLaunchPolicy policy_{};
    cudaStream_t stream_ = nullptr;
    int padded_channels_ = 0;
    int complex_channels_ = 0;
    bool prepared_ = false;
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> d_padded_{};
    Mem::DeviceLinearBuffer3D<cufftComplex> d_complex_{};
    Mem::DeviceLinearBuffer3D<float> d_weights_{};
    CudaFFT fft_{};
    Filter::CreateFilterKernelFromFFT filter_builder_{};
};

class BackProjectProcessor {
public:
    bool prepare(const SHeliCTParam& params, const Geometry& geometry,
        const Config& config)
    {
        params_ = params;
        geometry_ = geometry;
        config_ = config;
        return params_.iVX > 0 && params_.iVY > 0 && params_.iVZ > 0;
    }

    bool apply(const float* filtered, float* volume, cudaStream_t stream) const
    {
        if (!filtered || !volume) return false;
        detail::launch_backproject(filtered, volume,
            params_.iVX, params_.iVY, params_.iVZ,
            params_.vox_x_mm, params_.vox_y_mm, params_.vox_z_mm,
            params_.vol_offset_x_mm, params_.vol_offset_y_mm,
            params_.vol_offset_z_mm, geometry_, config_.redundancy_flat,
            config_.launch, stream);
        return true;
    }

private:
    SHeliCTParam params_{};
    Geometry geometry_{};
    Config config_{};
};

} } } // namespace YK::Helical::Wfbp
