#pragma once
#include "FloatVolume.hpp"
#include <memory>

namespace yk::visualization {
enum class MprInterpolation { Nearest,
    Linear,
    Cubic };
struct MprSettings {
    std::array<int, 3> indices {};
    std::array<double, 3> rotation_deg {}; // Rz * Ry * Rx, about the reconstruction grid axes.
    double offset_mm = 0; // Translate oblique plane along its normal from the linked cursor.
    MprInterpolation interpolation = MprInterpolation::Linear;
    // Row-major common MPR rotation. rotation_deg is its X/Y/Z editor representation.
    std::array<double, 9> reference_axes { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    std::array<double, 3> reference_roll_deg {};
};
struct MprPlane {
    std::array<double, 3> center {}, u { 1, 0, 0 }, v { 0, 1, 0 }, normal { 0, 0, 1 };
    std::array<double, 2> start {}, spacing { 1, 1 };
    int width = 0, height = 0;
};
// Plane coordinates are physical mm in the grid frame. The original grid is never resampled.
MprPlane makeMprPlane(const FloatVolume&, const MprSettings&, int panel);
std::array<double, 3> mprWorldPosition(const FloatVolume&, const std::array<int, 3>&);
std::array<int, 3> mprPick(const FloatVolume&, const MprPlane&, double x, double y);
// Rotate the other linked planes about the active plane's normal, keeping its image frame fixed.
void rotateMprReference(MprSettings&, int active_panel, double angle_rad);
void setMprRotation(MprSettings&, const std::array<double, 3>& degrees);
bool mprReferenceIsOrthogonal(const MprSettings&);
// Borrows the owner's FloatVolume storage, avoiding a second full-volume buffer.
// Clear before changing or releasing that volume. Copies only the requested 2D output.
class MprReslicer {
public:
    MprReslicer();
    ~MprReslicer();
    void clear();
    void setInput(const FloatVolume&);
    FloatSlice slice(const MprPlane&, MprInterpolation);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
struct CtWindowPreset {
    const char* name;
    double width, level;
};
inline constexpr std::array<CtWindowPreset, 6> ctWindowPresets { { { "Brain", 80, 40 }, { "Soft tissue", 400, 40 }, { "Bone", 2000, 500 },
    { "Lung", 1500, -600 }, { "Abdomen", 350, 50 }, { "Subdural", 200, 80 } } };
}
