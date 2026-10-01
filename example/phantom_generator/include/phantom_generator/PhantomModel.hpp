#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace phantom_generator {

struct Material {
    int label = 0;
    std::string name;
    double density = 0.0;
    std::string formula;
    std::string preset;
    std::string description;
};

struct Insert {
    int label = 0;
    double diameter_mm = 0.0;
    double contrast_fraction = 0.0;
    double center_x_mm = 0.0;
    double center_y_mm = 0.0;
    double thickness_mm = 0.0;
    double center_z_mm = 0.0;
};

struct Phantom {
    int nx = 256, ny = 256, nz = 128;
    double voxel_mm = 1.0;
    std::string type;
    std::string variant;
    std::vector<std::uint8_t> labels;
    std::vector<Material> materials;
    std::vector<Insert> inserts;
    double target_diameter_mm = 0.0;
    double target_thickness_mm = 0.0;
    double target_tangent = 0.0;
    double bead_spacing_mm = 0.0;
    double support_diameter_mm = 0.0;
    double support_length_mm = 0.0;
    double bead_ring_diameter_mm = 0.0;
    double lower_bead_ring_diameter_mm = 0.0;
    double bead_ring_spacing_mm = 0.0;
    double lower_ring_phase_deg = 0.0;
    double marker_bead_diameter_mm = 0.0;
    double marker_bead_offset_mm = 0.0;
    double marker_cylinder_height_mm = 0.0;
    bool cylindrical_ring_markers = false;
    bool spiral_beads = false;
    int beads_per_ring = 0;

    std::size_t index(int x, int y, int z) const {
        return (static_cast<std::size_t>(z) * ny + y) * nx + x;
    }
    double x(int i) const { return (i - (nx - 1) * 0.5) * voxel_mm; }
    double y(int i) const { return (i - (ny - 1) * 0.5) * voxel_mm; }
    double z(int i) const { return (i - (nz - 1) * 0.5) * voxel_mm; }
};

} // namespace phantom_generator
