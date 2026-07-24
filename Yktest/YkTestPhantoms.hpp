#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"

// Synthetic phantoms used by the refactored pipeline tests.  They deliberately
// stay host-side and deterministic: a test does not depend on a local raw file
// and callers can upload the returned [z][y][x] array through any API.
namespace YK::TestPhantom {

constexpr float kPi = 3.14159265358979323846f;

inline size_t index(const SCBCTParams& p, int x, int y, int z)
{
    return (static_cast<size_t>(z) * p.iVY + y) * p.iVX + x;
}

inline void voxelCenterMm(const SCBCTParams& p, int x, int y, int z,
    float& px, float& py, float& pz)
{
    px = (x + 0.5f - 0.5f * p.iVX) * p.vox_x_mm;
    py = (y + 0.5f - 0.5f * p.iVY) * p.vox_y_mm;
    pz = (z + 0.5f - 0.5f * p.iVZ) * p.vox_z_mm;
}

// A compact analytic phantom for FP/BP smoke tests: water-equivalent cylinder,
// one high-contrast bead and one air cavity.  The three structures exercise
// both smooth boundaries and isolated high-frequency content.
inline std::vector<float> makeBasic(const SCBCTParams& p)
{
    std::vector<float> volume(static_cast<size_t>(p.iVX) * p.iVY * p.iVZ, 0.f);
    const float radius = 0.32f * std::min(p.iVX * p.vox_x_mm, p.iVY * p.vox_y_mm);
    const float bead_radius = std::max(p.vox_x_mm, p.vox_y_mm) * 2.0f;
    for (int z = 0; z < p.iVZ; ++z) for (int y = 0; y < p.iVY; ++y) for (int x = 0; x < p.iVX; ++x) {
        float px, py, pz;
        voxelCenterMm(p, x, y, z, px, py, pz);
        float value = px * px + py * py <= radius * radius ? 0.020f : 0.f;
        if ((px - radius * 0.35f) * (px - radius * 0.35f) +
            (py + radius * 0.20f) * (py + radius * 0.20f) + pz * pz <= bead_radius * bead_radius)
            value = 0.080f;
        if ((px + radius * 0.30f) * (px + radius * 0.30f) +
            (py - radius * 0.18f) * (py - radius * 0.18f) <= bead_radius * bead_radius)
            value = 0.f;
        volume[index(p, x, y, z)] = value;
    }
    return volume;
}

// A deliberately small "Catphan-like" QA phantom, not a clinical Catphan
// specification.  It keeps the useful module concepts for regressions:
//   lower z: uniformity; middle z: material inserts; upper z: low contrast
//   disks plus alternating line bars.  Values are linear attenuation (1/mm).
inline std::vector<float> makeCatphanLike(const SCBCTParams& p)
{
    std::vector<float> volume(static_cast<size_t>(p.iVX) * p.iVY * p.iVZ, 0.f);
    const float body_radius = 0.40f * std::min(p.iVX * p.vox_x_mm, p.iVY * p.vox_y_mm);
    const float insert_radius = std::max(1.5f * p.vox_x_mm, body_radius * 0.10f);
    const float ring_radius = body_radius * 0.57f;
    const float z_material = -0.13f * p.iVZ * p.vox_z_mm;
    const float z_low_contrast = 0.08f * p.iVZ * p.vox_z_mm;
    const float z_resolution = 0.27f * p.iVZ * p.vox_z_mm;
    const float module_half = std::max(2.f * p.vox_z_mm, 0.09f * p.iVZ * p.vox_z_mm);
    constexpr float material_mu[] = { 0.000f, 0.012f, 0.028f, 0.055f, 0.085f, 0.040f };

    for (int z = 0; z < p.iVZ; ++z) for (int y = 0; y < p.iVY; ++y) for (int x = 0; x < p.iVX; ++x) {
        float px, py, pz;
        voxelCenterMm(p, x, y, z, px, py, pz);
        const float r2 = px * px + py * py;
        if (r2 > body_radius * body_radius) continue;
        float value = 0.020f;

        if (std::fabs(pz - z_material) <= module_half) {
            for (int i = 0; i < 6; ++i) {
                const float angle = 2.f * kPi * i / 6.f;
                const float cx = ring_radius * std::cos(angle);
                const float cy = ring_radius * std::sin(angle);
                if ((px - cx) * (px - cx) + (py - cy) * (py - cy) <= insert_radius * insert_radius) {
                    value = material_mu[i];
                    break;
                }
            }
        } else if (std::fabs(pz - z_low_contrast) <= module_half) {
            for (int i = 0; i < 5; ++i) {
                const float angle = 2.f * kPi * i / 5.f;
                const float cx = ring_radius * 0.62f * std::cos(angle);
                const float cy = ring_radius * 0.62f * std::sin(angle);
                const float radius = insert_radius * (1.25f - 0.12f * i);
                if ((px - cx) * (px - cx) + (py - cy) * (py - cy) <= radius * radius) {
                    value = 0.022f;
                    break;
                }
            }
        } else if (std::fabs(pz - z_resolution) <= module_half &&
            std::fabs(px) < body_radius * 0.55f && std::fabs(py) < body_radius * 0.24f) {
            // Alternating 2-voxel bars are robust on the intentionally small
            // smoke-test volume and expose interpolation/geometry mistakes.
            const int stripe = static_cast<int>((px + body_radius * 0.55f) / (2.f * p.vox_x_mm));
            value = (stripe & 1) ? 0.050f : 0.020f;
        }
        volume[index(p, x, y, z)] = value;
    }
    return volume;
}

} // namespace YK::TestPhantom
