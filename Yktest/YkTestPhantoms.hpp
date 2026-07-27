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

// 三轴方向箭头模体。形状规则与 gen_arrow_phantom_v2 一致：每根箭头由
// 实心圆柱箭杆和实心圆锥箭头组成，分别指向 +X/+Y/+Z，尺寸按 X<Y<Z
// 递增。MATLAB 原版的 uint8 标签 2/4/6 在这里映射为 0.02/0.04/0.06
// mm^-1，使模体可直接用于 CT 正投影，同时仍可由灰度唯一判断方向。
inline std::vector<float> makeArrowDirections(const SCBCTParams& p,
    bool stamp_labels = false)
{
    std::vector<float> volume(static_cast<size_t>(p.iVX) * p.iVY * p.iVZ, 0.f);
    const int min_dim = std::min({ p.iVX, p.iVY, p.iVZ });
    const auto clamp_int = [](int value, int low, int high) {
        return std::max(low, std::min(high, value));
    };
    const int radius = clamp_int(static_cast<int>(std::lround(0.020 * min_dim)),
        2, 12);
    const int spacing = clamp_int(static_cast<int>(std::lround(4.0 * radius)),
        2 * radius + 1, std::max(2 * radius + 1,
            static_cast<int>(std::lround(0.06 * min_dim))));
    const int shaft_base = clamp_int(static_cast<int>(std::lround(
        0.30 * min_dim / spacing)), 4, 15);
    const int layer_base = clamp_int(static_cast<int>(std::lround(
        0.12 * min_dim / spacing)), 2, 5);
    const int spread_base = clamp_int(static_cast<int>(std::lround(
        0.07 * min_dim)), 2 * radius,
        std::max(2 * radius, static_cast<int>(std::lround(0.12 * min_dim))));
    constexpr float gradient[3] = { 0.5f, 1.0f, 1.3f };
    constexpr float attenuation[3] = { 0.02f, 0.04f, 0.06f };
    const float center[3] = {
        0.5f * (p.iVX - 1), 0.5f * (p.iVY - 1), 0.5f * (p.iVZ - 1)
    };

    for (int axis = 0; axis < 3; ++axis) {
        const int shaft_count = std::max(3, static_cast<int>(std::lround(
            shaft_base * gradient[axis])));
        const int cone_layers = std::max(2, static_cast<int>(std::lround(
            layer_base * gradient[axis])));
        const int cone_radius = std::max(2 * radius,
            static_cast<int>(std::lround(spread_base * gradient[axis])));
        const float shaft_start = center[axis] -
            static_cast<float>(shaft_count / 2 * spacing);
        const float shaft_end = shaft_start + (shaft_count - 1) * spacing;
        const float cone_base = shaft_end + spacing;
        const float cone_tip = cone_base + cone_layers * spacing;

        for (int z = 0; z < p.iVZ; ++z) {
            for (int y = 0; y < p.iVY; ++y) {
                for (int x = 0; x < p.iVX; ++x) {
                    const float coordinate[3] = {
                        static_cast<float>(x), static_cast<float>(y),
                        static_cast<float>(z)
                    };
                    float radial2 = 0.f;
                    for (int d = 0; d < 3; ++d) {
                        if (d == axis) continue;
                        const float delta = coordinate[d] - center[d];
                        radial2 += delta * delta;
                    }
                    const float axial = coordinate[axis];
                    const bool in_shaft = axial >= shaft_start &&
                        axial <= shaft_end && radial2 <= radius * radius;
                    bool in_cone = false;
                    if (axial >= cone_base && axial <= cone_tip) {
                        const float t = (axial - cone_base) /
                            std::max(cone_tip - cone_base, 1e-6f);
                        const float radius_at_t = cone_radius * (1.f - t);
                        in_cone = radial2 <= radius_at_t * radius_at_t;
                    }
                    if (in_shaft || in_cone)
                        volume[index(p, x, y, z)] = attenuation[axis];
                }
            }
        }
    }

    if (stamp_labels) {
        // 等价迁移 stamp_axis_labels：六个边界面均写中心主标签，四边再写
        // 邻轴方向。从各面约定视线方向观察时，所有文字均正向可读。
        constexpr const char* plus[7] = {
            "00000", "00100", "00100", "11111", "00100", "00100", "00000"
        };
        constexpr const char* minus[7] = {
            "00000", "00000", "00000", "11111", "00000", "00000", "00000"
        };
        constexpr const char* glyph_x[7] = {
            "10101", "10001", "01010", "00100", "01010", "10001", "10001"
        };
        constexpr const char* glyph_y[7] = {
            "10001", "10001", "01010", "00100", "00100", "00100", "00100"
        };
        constexpr const char* glyph_z[7] = {
            "11111", "00001", "00010", "00100", "01000", "10000", "11111"
        };
        const auto glyph = [&](char character) {
            if (character == '+') return plus;
            if (character == '-') return minus;
            if (character == 'X') return glyph_x;
            if (character == 'Y') return glyph_y;
            return glyph_z;
        };

        struct Face {
            const char* name;
            int depth_axis;
            int depth_index;
            int h[3];
            int v[3];
            int inward;
        };
        const int dimensions[3] = { p.iVX, p.iVY, p.iVZ };
        const Face faces[] = {
            { "+X", 0, p.iVX - 1, { 0, 1, 0 }, { 0, 0,-1 }, -1 },
            { "-X", 0, 0,          { 0,-1, 0 }, { 0, 0,-1 },  1 },
            { "+Y", 1, p.iVY - 1, {-1, 0, 0 }, { 0, 0,-1 }, -1 },
            { "-Y", 1, 0,          { 1, 0, 0 }, { 0, 0,-1 },  1 },
            { "+Z", 2, p.iVZ - 1, { 1, 0, 0 }, { 0,-1, 0 }, -1 },
            { "-Z", 2, 0,          { 1, 0, 0 }, { 0, 1, 0 },  1 }
        };
        constexpr int char_width = 5;
        constexpr int char_height = 7;
        constexpr int char_gap = 2;
        constexpr int margin = 4;
        constexpr int thickness = 4;
        const int scale_max = std::max(1, static_cast<int>(std::floor(
            min_dim * 0.30 / std::max(2 * char_width + char_gap, char_height))));
        const int font_scale = std::min(5, scale_max);
        const int small_scale = std::min(std::max(1, font_scale - 1), scale_max);
        constexpr float label_mu = 0.01f;

        const auto axis_size = [&](const int direction[3]) {
            for (int d = 0; d < 3; ++d)
                if (direction[d] != 0) return dimensions[d];
            return 0;
        };
        const auto edge_label = [](const int direction[3], int sign,
            char result[3]) {
            int dimension = 0;
            while (dimension < 3 && direction[dimension] == 0) ++dimension;
            result[0] = sign * direction[dimension] > 0 ? '+' : '-';
            result[1] = "XYZ"[dimension];
            result[2] = '\0';
        };
        const auto rasterize = [&](const Face& face, const char* text,
            int h_start, int v_start, int scale) {
            int origin[3] = {};
            for (int d = 0; d < 3; ++d) {
                if (face.h[d] > 0) origin[d] = h_start;
                else if (face.h[d] < 0) origin[d] = dimensions[d] - 1 - h_start;
                if (face.v[d] > 0) origin[d] = v_start;
                else if (face.v[d] < 0) origin[d] = dimensions[d] - 1 - v_start;
            }
            for (int depth = 0; depth < thickness; ++depth) {
                const int depth_index = face.depth_index + face.inward * depth;
                for (int character = 0; text[character] != '\0'; ++character) {
                    const char* const* bitmap = glyph(text[character]);
                    const int character_offset = character *
                        (char_width + char_gap) * scale;
                    for (int row = 0; row < char_height; ++row) {
                        for (int column = 0; column < char_width; ++column) {
                            if (bitmap[row][column] != '1') continue;
                            for (int dr = 0; dr < scale; ++dr) {
                                for (int dc = 0; dc < scale; ++dc) {
                                    const int h_offset = character_offset +
                                        column * scale + dc;
                                    const int v_offset = row * scale + dr;
                                    int coordinate[3];
                                    for (int d = 0; d < 3; ++d)
                                        coordinate[d] = origin[d] +
                                            h_offset * face.h[d] +
                                            v_offset * face.v[d];
                                    coordinate[face.depth_axis] = depth_index;
                                    if (coordinate[0] >= 0 && coordinate[0] < p.iVX &&
                                        coordinate[1] >= 0 && coordinate[1] < p.iVY &&
                                        coordinate[2] >= 0 && coordinate[2] < p.iVZ)
                                        volume[index(p, coordinate[0], coordinate[1],
                                            coordinate[2])] = label_mu;
                                }
                            }
                        }
                    }
                }
            }
        };

        for (const auto& face : faces) {
            const int h_size = axis_size(face.h);
            const int v_size = axis_size(face.v);
            const int main_width = (2 * char_width + char_gap) * font_scale;
            rasterize(face, face.name,
                std::max(margin, (h_size - main_width) / 2),
                std::max(margin, (v_size - char_height * font_scale) / 2),
                font_scale);

            char right[3], left[3], top[3], bottom[3];
            edge_label(face.h, 1, right);
            edge_label(face.h, -1, left);
            edge_label(face.v, -1, top);
            edge_label(face.v, 1, bottom);
            const int edge_width = (2 * char_width + char_gap) * small_scale;
            const int edge_height = char_height * small_scale;
            rasterize(face, right, std::max(margin, h_size - edge_width - margin),
                std::max(margin, (v_size - edge_height) / 2), small_scale);
            rasterize(face, left, margin,
                std::max(margin, (v_size - edge_height) / 2), small_scale);
            rasterize(face, top, std::max(margin, (h_size - edge_width) / 2),
                margin, small_scale);
            rasterize(face, bottom, std::max(margin, (h_size - edge_width) / 2),
                std::max(margin, v_size - edge_height - margin), small_scale);
        }
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
