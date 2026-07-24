#pragma once

#include <cstddef>

#include "global/YkMem3d.hpp"

namespace YK {

// RAII workspace for algorithms that expose raw linear float pointers to CUDA
// kernels.  Allocation still goes through MemoryController, so ownership and
// release are consistent with the rest of the project.
class DeviceWorkspaceF32 {
public:
    void allocate(size_t elements, int device_id)
    {
        buffer_ = memory_.allocateDevice3D<float>(static_cast<int>(elements), 1, 1, device_id);
    }
    void reset() noexcept { buffer_ = Mem::DeviceLinearBuffer3D<float>{}; }
    float* data() const noexcept { return buffer_.data(); }
    explicit operator bool() const noexcept { return data() != nullptr; }
    operator float* () const noexcept { return data(); }
private:
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> buffer_{};
};

} // namespace YK
