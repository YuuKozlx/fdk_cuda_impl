#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "global/YkFdkKernelTypes.hpp"

namespace YK::NonStandardFpBp {

enum class EProjectionBatchLocation { Host, Device };

// data starts at first_view, not at the beginning of the acquisition.
// Capacity is measured in float elements on the acquired detector grid.
struct NonStandardFdkProjectionBatch {
    const float* data = nullptr;
    int first_view = 0;
    int view_count = 0;
    EProjectionBatchLocation location = EProjectionBatchLocation::Device;
    size_t element_count = 0;
};

// Shared validation only: the pipelines own CUDA workspaces and synchronize
// each accepted batch before advancing received_, so input may be reused.
class NonStandardFdkStreamingState {
public:
    bool begin(float* volume) {
        if (!volume || active_) return false;
        volume_ = volume;
        received_ = 0;
        active_ = true;
        failed_ = false;
        return true;
    }

    bool accepts(const NonStandardFdkProjectionBatch& batch,
        int total_views, size_t view_elements) const {
        if (!active_ || failed_ || !batch.data || batch.view_count <= 0 ||
            batch.first_view != received_ || received_ > total_views ||
            batch.view_count > total_views - received_ || view_elements == 0 ||
            (batch.location != EProjectionBatchLocation::Host &&
             batch.location != EProjectionBatchLocation::Device)) return false;
        const size_t count = static_cast<size_t>(batch.view_count);
        const size_t limit = std::numeric_limits<size_t>::max() / sizeof(float);
        return count <= limit / view_elements &&
            batch.element_count >= count * view_elements;
    }

    void commit(int count) { received_ += count; }
    void fail() { failed_ = true; }
    bool canComplete(int total_views) const {
        return active_ && !failed_ && received_ == total_views;
    }
    void complete() { active_ = false; volume_ = nullptr; }
    void reset() { *this = NonStandardFdkStreamingState{}; }
    bool active() const { return active_; }
    int received() const { return received_; }
    float* volume() const { return volume_; }

private:
    float* volume_ = nullptr;
    int received_ = 0;
    bool active_ = false;
    bool failed_ = false;
};

// Keep the original prepared chunk's shared filter spacing even when an
// incoming batch cuts that chunk. Otherwise perspective maps with varying du
// would change reconstruction scale merely because input was packetized.
inline const SFDKGeoParamPerView* nonStandardStreamingFilterGeometry(
    const std::vector<SFDKGeoParamPerView>& derived, int first_view, int count,
    int chunk_size, std::vector<SFDKGeoParamPerView>& scratch)
{
    scratch.assign(derived.begin() + first_view,
        derived.begin() + first_view + count);
    scratch.front().du_mm = derived[first_view - first_view % chunk_size].du_mm;
    return scratch.data();
}

} // namespace YK::NonStandardFpBp
