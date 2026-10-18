#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace yk::spectral::primary_detail {
// Legacy one-frame air files broadcast to all views. New files use
// [view][row][column], matched to the projection sequence.
class AirReferenceReader {
public:
    AirReferenceReader(const std::filesystem::path& path, size_t pixels, int views)
        : pixels_(pixels), views_(views), stream_(path, std::ios::binary) {
        if (!stream_ || pixels == 0 || views <= 0)
            throw std::runtime_error("invalid air reference file or dimensions");
        stream_.seekg(0,std::ios::end);
        const auto bytes = static_cast<std::uint64_t>(stream_.tellg());
        per_view_ = bytes == pixels_ * sizeof(float) * static_cast<size_t>(views_);
        if (!per_view_ && bytes != pixels_ * sizeof(float))
            throw std::runtime_error("air reference must contain one frame or exactly all projection views");
    }
    void read(int view, std::vector<float>& air) {
        if (view < 0 || view >= views_) throw std::runtime_error("air reference view out of range");
        const size_t index = per_view_ ? static_cast<size_t>(view) : 0;
        stream_.clear();
        stream_.seekg(static_cast<std::streamoff>(index*pixels_*sizeof(float)),std::ios::beg);
        air.resize(pixels_);
        stream_.read(reinterpret_cast<char*>(air.data()),
            static_cast<std::streamsize>(pixels_*sizeof(float)));
        if (!stream_) throw std::runtime_error("cannot read air reference frame");
        for(float value:air)
            if (!std::isfinite(value) || value<=0.f)
                throw std::runtime_error("air reference contains non-positive or non-finite values");
    }
private:
    size_t pixels_;
    int views_;
    bool per_view_=false;
    std::ifstream stream_;
};
} // namespace yk::spectral::primary_detail
