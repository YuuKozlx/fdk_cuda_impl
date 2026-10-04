#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "phantom_generator/PhantomModel.hpp"
#include "phantom_generator/GeneratorDispatch.hpp"
#include "phantom_generator/PhantomOutput.hpp"

namespace {

using phantom_generator::Insert;
using phantom_generator::Material;
using phantom_generator::Phantom;

bool ellipse(double x, double y, double cx, double cy,
    double rx, double ry, double angle_deg = 0.0)
{
    const double a = angle_deg * std::acos(-1.0) / 180.0;
    const double ca = std::cos(a), sa = std::sin(a);
    const double dx = x - cx, dy = y - cy;
    const double u = ca * dx + sa * dy;
    const double v = -sa * dx + ca * dy;
    return u * u / (rx * rx) + v * v / (ry * ry) <= 1.0;
}

void fillWaterShell(Phantom& p, bool elliptical)
{
    const double rx = p.nx * p.voxel_mm * 0.40;
    const double ry = p.ny * p.voxel_mm * (elliptical ? 0.32 : 0.40);
    const double shell = std::max(2.0, 3.0 * p.voxel_mm);
    for (int z = 0; z < p.nz; ++z) for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x) {
            const bool outer = ellipse(p.x(x), p.y(y), 0, 0, rx, ry);
            const bool inner = ellipse(p.x(x), p.y(y), 0, 0, rx - shell, ry - shell);
            p.labels[p.index(x, y, z)] = inner ? 1 : outer ? 2 : 0;
        }
    p.materials = {{1, "water", 1.0, "H2O", {}, {}},
                   {2, "pmma", 1.18, "C5H8O2", {}, {}}};
    p.variant = elliptical ? "elliptical PMMA shell water phantom" :
        "circular PMMA shell water phantom";
}

void makeWaterCylinder200mm(Phantom& p)
{
    constexpr double outer_radius = 100.0;
    constexpr double shell = 5.0;
    const double inner_radius = outer_radius - shell;
    for (int z = 0; z < p.nz; ++z) for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x) {
            const double r2 = p.x(x) * p.x(x) + p.y(y) * p.y(y);
            p.labels[p.index(x, y, z)] = r2 <= inner_radius * inner_radius ? 1 :
                r2 <= outer_radius * outer_radius ? 2 : 0;
        }
    p.materials = {{1, "water", 1.0, "H2O", {}, {}},
                   {2, "pmma", 1.18, "C5H8O2", {}, {}}};
    p.target_diameter_mm = 2.0 * outer_radius;
    p.variant = "200 mm water cylinder with 5 mm PMMA shell";
}

void requireTargetResolution(const Phantom& p, double feature_mm)
{
    if (p.voxel_mm > feature_mm)
        throw std::runtime_error("voxel_mm must be <= " + std::to_string(feature_mm) +
            " mm for this image-quality target");
}

void makeWire(Phantom& p, bool slanted)
{
    constexpr double diameter = 0.05;
    constexpr double tangent = 0.42;
    requireTargetResolution(p, diameter * 0.5);
    const double radius = diameter * 0.5;
    for (int z = 0; z < p.nz; ++z) for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x) {
            double distance_squared = p.x(x) * p.x(x) + p.y(y) * p.y(y);
            if (slanted) {
                // Infinite line through the origin with direction (1, 0, tan(theta)).
                const double residual = p.z(z) - tangent * p.x(x);
                distance_squared = p.y(y) * p.y(y) +
                    residual * residual / (1.0 + tangent * tangent);
            }
            if (distance_squared <= radius * radius)
                p.labels[p.index(x, y, z)] = 3;
        }
    p.materials = {{3, "tungsten", 19.25, "W", {},
        slanted ? "slanted tungsten wire" : "axial tungsten wire"}};
    p.target_diameter_mm = diameter;
    p.target_tangent = slanted ? tangent : 0.0;
    p.variant = slanted ? "0.05 mm tungsten wire, z = 0.42 * x" :
        "0.05 mm axial tungsten wire";
}

void makeTungstenBeadLine(Phantom& p, double bead_diameter_mm,
    double bead_spacing_mm)
{
    constexpr double support_radius_mm = 15.0;
    constexpr double support_length_mm = 120.0;
    constexpr std::uint8_t kTungstenLabel = 100;
    const double bead_radius_mm = bead_diameter_mm * 0.5;
    if (p.voxel_mm > bead_diameter_mm / 8.0)
        throw std::runtime_error("bead line requires voxel_mm <= diameter/8");
    if (p.nx * p.voxel_mm * 0.5 < support_radius_mm ||
        p.ny * p.voxel_mm * 0.5 < support_radius_mm ||
        p.nz * p.voxel_mm * 0.5 < support_length_mm * 0.5)
        throw std::runtime_error("bead line requires a 30 mm x 80 mm support field");

    const int bead_count = static_cast<int>(
        std::floor((support_length_mm - 2.0 * bead_radius_mm) /
            bead_spacing_mm)) + 1;
    const double first_z = -0.5 * (bead_count - 1) * bead_spacing_mm;
    for (int z = 0; z < p.nz; ++z) for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x) {
            const double px = p.x(x), py = p.y(y), pz = p.z(z);
            if (px * px + py * py > support_radius_mm * support_radius_mm) continue;
            p.labels[p.index(x, y, z)] = 1;
            const double slot = std::round((pz - first_z) / bead_spacing_mm);
            const double nearest_z = first_z + std::clamp(slot, 0.0,
                static_cast<double>(bead_count - 1)) * bead_spacing_mm;
            const double dz = pz - nearest_z;
            if (px * px + py * py + dz * dz <= bead_radius_mm * bead_radius_mm)
                p.labels[p.index(x, y, z)] = kTungstenLabel;
        }
    p.materials = {{1, "pmma_support", 1.18, "C5H8O2", {},
        "external cylindrical PMMA support"},
        {kTungstenLabel, "tungsten", 19.25, "W", {}, "linear tungsten beads"}};
    p.target_diameter_mm = bead_diameter_mm;
    p.bead_spacing_mm = bead_spacing_mm;
    p.support_diameter_mm = 2.0 * support_radius_mm;
    p.support_length_mm = support_length_mm;
    p.variant = "single straight row of tungsten spheres, diameter " +
        std::to_string(bead_diameter_mm) + " mm, center spacing " +
        std::to_string(bead_spacing_mm) + " mm, in 30 mm diameter x 120 mm PMMA cylinder";
    for (int i = 0; i < bead_count; ++i)
        p.inserts.push_back({kTungstenLabel, bead_diameter_mm, 0.0,
            0.0, 0.0, 0.0, first_z + i * bead_spacing_mm});
}

void makeTungstenBeadDoubleRing(Phantom& p, double ring_diameter_mm,
    double ring_spacing_mm, double bead_diameter_mm,
    double support_diameter_mm, double support_length_mm, int beads_per_ring,
    bool cylindrical_markers = false, bool replace_markers = false)
{
    constexpr std::uint8_t kPmmaLabel = 1;
    constexpr std::uint8_t kTungstenLabel = 100;
    const double support_radius = support_diameter_mm * 0.5;
    const double ring_radius = ring_diameter_mm * 0.5;
    const double bead_radius = bead_diameter_mm * 0.5;
    const double marker_diameter = replace_markers ? bead_diameter_mm : 5.0;
    const double marker_height = replace_markers ? 2.0 * bead_diameter_mm : 5.0;
    const double marker_gap = 10.0;
    if (beads_per_ring < 1)
        throw std::runtime_error("beads_per_ring must be positive");
    if (ring_radius + bead_radius > support_radius ||
        ring_spacing_mm * 0.5 + (cylindrical_markers
            ? (replace_markers ? marker_height * 0.5 :
                marker_gap + marker_height * 0.5) : bead_radius) >
            support_length_mm * 0.5)
        throw std::runtime_error("double bead rings do not fit inside support cylinder");
    if (p.nx * p.voxel_mm < support_diameter_mm ||
        p.ny * p.voxel_mm < support_diameter_mm ||
        p.nz * p.voxel_mm < support_length_mm)
        throw std::runtime_error("volume is smaller than double-ring support cylinder");
    if (p.voxel_mm > bead_diameter_mm / 8.0)
        throw std::runtime_error("double bead rings require voxel_mm <= bead diameter/8");

    for (int z = 0; z < p.nz; ++z) {
        if (std::abs(p.z(z)) > support_length_mm * 0.5) continue;
        for (int y = 0; y < p.ny; ++y) for (int x = 0; x < p.nx; ++x)
            if (p.x(x) * p.x(x) + p.y(y) * p.y(y) <=
                support_radius * support_radius)
                p.labels[p.index(x, y, z)] = kPmmaLabel;
    }

    for (const double center_z : {-ring_spacing_mm * 0.5,
            ring_spacing_mm * 0.5}) {
        for (int bead = 0; bead < beads_per_ring; ++bead) {
            const double angle = 2.0 * std::acos(-1.0) * bead /
                beads_per_ring;
            const double center_x = ring_radius * std::cos(angle);
            const double center_y = ring_radius * std::sin(angle);
            const int xmin = std::max(0, static_cast<int>(std::floor(
                (center_x - bead_radius) / p.voxel_mm + (p.nx - 1) * 0.5)));
            const int xmax = std::min(p.nx - 1, static_cast<int>(std::ceil(
                (center_x + bead_radius) / p.voxel_mm + (p.nx - 1) * 0.5)));
            const int ymin = std::max(0, static_cast<int>(std::floor(
                (center_y - bead_radius) / p.voxel_mm + (p.ny - 1) * 0.5)));
            const int ymax = std::min(p.ny - 1, static_cast<int>(std::ceil(
                (center_y + bead_radius) / p.voxel_mm + (p.ny - 1) * 0.5)));
            const bool is_cylinder = cylindrical_markers && replace_markers && bead == 0;
            const double half_height = is_cylinder
                ? marker_height * 0.5 : bead_radius;
            const int zmin = std::max(0, static_cast<int>(std::floor(
                (center_z - half_height) / p.voxel_mm + (p.nz - 1) * 0.5)));
            const int zmax = std::min(p.nz - 1, static_cast<int>(std::ceil(
                (center_z + half_height) / p.voxel_mm + (p.nz - 1) * 0.5)));
            for (int z = zmin; z <= zmax; ++z)
                for (int y = ymin; y <= ymax; ++y)
                    for (int x = xmin; x <= xmax; ++x) {
                        const double dx = p.x(x) - center_x;
                        const double dy = p.y(y) - center_y;
                        const double dz = p.z(z) - center_z;
                        const bool inside = is_cylinder
                            ? dx * dx + dy * dy <= bead_radius * bead_radius &&
                                std::abs(dz) <= half_height
                            : dx * dx + dy * dy + dz * dz <=
                                bead_radius * bead_radius;
                        if (inside)
                            p.labels[p.index(x, y, z)] = kTungstenLabel;
                    }
            p.inserts.push_back({kTungstenLabel, bead_diameter_mm, 0.0,
                center_x, center_y, is_cylinder ? marker_height : 0.0,
                center_z});
        }
    }
    p.materials = {{kPmmaLabel, "pmma_support", 1.18, "C5H8O2", {},
        "cylindrical PMMA support"},
        {kTungstenLabel, "tungsten", 19.25, "W", {},
        "two aligned rings of tungsten spheres"}};
    p.target_diameter_mm = bead_diameter_mm;
    p.support_diameter_mm = support_diameter_mm;
    p.support_length_mm = support_length_mm;
    p.bead_ring_diameter_mm = ring_diameter_mm;
    p.bead_ring_spacing_mm = ring_spacing_mm;
    p.beads_per_ring = beads_per_ring;
    if (cylindrical_markers && !replace_markers) {
        const double marker_radius = marker_diameter * 0.5;
        for (const double center_z : {-ring_spacing_mm * 0.5 - marker_gap,
                ring_spacing_mm * 0.5 + marker_gap}) {
            const int zmin = std::max(0, static_cast<int>(std::floor(
                (center_z - marker_height * 0.5) / p.voxel_mm +
                (p.nz - 1) * 0.5)));
            const int zmax = std::min(p.nz - 1, static_cast<int>(std::ceil(
                (center_z + marker_height * 0.5) / p.voxel_mm +
                (p.nz - 1) * 0.5)));
            const int xmin = std::max(0, static_cast<int>(std::floor(
                (ring_radius - marker_radius) / p.voxel_mm +
                (p.nx - 1) * 0.5)));
            const int xmax = std::min(p.nx - 1, static_cast<int>(std::ceil(
                (ring_radius + marker_radius) / p.voxel_mm +
                (p.nx - 1) * 0.5)));
            for (int z = zmin; z <= zmax; ++z)
                for (int y = 0; y < p.ny; ++y)
                    for (int x = xmin; x <= xmax; ++x) {
                        const double dx = p.x(x) - ring_radius;
                        const double dz = p.z(z) - center_z;
                        if (dx * dx + p.y(y) * p.y(y) <= marker_radius * marker_radius &&
                            std::abs(dz) <= marker_height * 0.5)
                            p.labels[p.index(x, y, z)] = kTungstenLabel;
                    }
            p.inserts.push_back({kTungstenLabel, marker_diameter, 0.0,
                ring_radius, 0.0, marker_height, center_z});
        }
        p.marker_bead_offset_mm = marker_gap;
    }
    p.cylindrical_ring_markers = cylindrical_markers;
    p.marker_cylinder_height_mm = cylindrical_markers ? marker_height : 0.0;
    p.variant = "two aligned " + std::to_string(beads_per_ring) +
        "-bead tungsten rings in a PMMA cylinder" +
        (cylindrical_markers ? ", with one axial cylinder per ring" : "");
}

void paintSphere(Phantom& p, double center_x, double center_y,
    double center_z, double diameter_mm, std::uint8_t label)
{
    const double radius = diameter_mm * 0.5;
    const int xmin = std::max(0, static_cast<int>(std::floor(
        (center_x - radius) / p.voxel_mm + (p.nx - 1) * 0.5)));
    const int xmax = std::min(p.nx - 1, static_cast<int>(std::ceil(
        (center_x + radius) / p.voxel_mm + (p.nx - 1) * 0.5)));
    const int ymin = std::max(0, static_cast<int>(std::floor(
        (center_y - radius) / p.voxel_mm + (p.ny - 1) * 0.5)));
    const int ymax = std::min(p.ny - 1, static_cast<int>(std::ceil(
        (center_y + radius) / p.voxel_mm + (p.ny - 1) * 0.5)));
    const int zmin = std::max(0, static_cast<int>(std::floor(
        (center_z - radius) / p.voxel_mm + (p.nz - 1) * 0.5)));
    const int zmax = std::min(p.nz - 1, static_cast<int>(std::ceil(
        (center_z + radius) / p.voxel_mm + (p.nz - 1) * 0.5)));
    for (int z = zmin; z <= zmax; ++z)
        for (int y = ymin; y <= ymax; ++y)
            for (int x = xmin; x <= xmax; ++x) {
                const double dx = p.x(x) - center_x;
                const double dy = p.y(y) - center_y;
                const double dz = p.z(z) - center_z;
                if (dx * dx + dy * dy + dz * dz <= radius * radius)
                    p.labels[p.index(x, y, z)] = label;
            }
}

void makeTungstenBeadDoubleRingMarker(Phantom& p,
    double upper_ring_diameter_mm, double lower_ring_diameter_mm,
    double ring_spacing_mm, double bead_diameter_mm,
    double lower_phase_deg, double marker_diameter_mm,
    double marker_offset_mm, double support_diameter_mm,
    double support_length_mm, int beads_per_ring,
    double secondary_marker_phase_deg = 0.0,
    double secondary_marker_offset_mm = 0.0)
{
    constexpr std::uint8_t kPmmaLabel = 1;
    constexpr std::uint8_t kTungstenLabel = 100;
    const double support_radius = support_diameter_mm * 0.5;
    const double upper_radius = upper_ring_diameter_mm * 0.5;
    const double lower_radius = lower_ring_diameter_mm * 0.5;
    const double bead_radius = bead_diameter_mm * 0.5;
    const double marker_radius = marker_diameter_mm * 0.5;
    const double lower_z = -ring_spacing_mm * 0.5;
    const double upper_z = ring_spacing_mm * 0.5;
    const double marker_z = upper_z + marker_offset_mm;
    if (beads_per_ring < 1)
        throw std::runtime_error("beads_per_ring must be positive");
    if (std::max(upper_radius, lower_radius) +
            std::max(bead_radius, marker_radius) > support_radius ||
        marker_z + marker_radius > support_length_mm * 0.5 ||
        lower_z - bead_radius < -support_length_mm * 0.5)
        throw std::runtime_error("marked double bead rings do not fit inside support cylinder");
    if (p.nx * p.voxel_mm < support_diameter_mm ||
        p.ny * p.voxel_mm < support_diameter_mm ||
        p.nz * p.voxel_mm < support_length_mm)
        throw std::runtime_error("volume is smaller than marked double-ring support cylinder");
    if (p.voxel_mm > bead_diameter_mm / 8.0)
        throw std::runtime_error("marked double bead rings require voxel_mm <= bead diameter/8");

    for (int z = 0; z < p.nz; ++z) {
        if (std::abs(p.z(z)) > support_length_mm * 0.5) continue;
        for (int y = 0; y < p.ny; ++y) for (int x = 0; x < p.nx; ++x)
            if (p.x(x) * p.x(x) + p.y(y) * p.y(y) <=
                support_radius * support_radius)
                p.labels[p.index(x, y, z)] = kPmmaLabel;
    }

    const double pi = std::acos(-1.0);
    const double lower_phase = lower_phase_deg * pi / 180.0;
    for (int bead = 0; bead < beads_per_ring; ++bead) {
        const double upper_angle = 2.0 * pi * bead / beads_per_ring;
        const double upper_x = upper_radius * std::cos(upper_angle);
        const double upper_y = upper_radius * std::sin(upper_angle);
        paintSphere(p, upper_x, upper_y, upper_z, bead_diameter_mm,
            kTungstenLabel);
        p.inserts.push_back({kTungstenLabel, bead_diameter_mm, 0.0,
            upper_x, upper_y, 0.0, upper_z});

        const double lower_angle = upper_angle + lower_phase;
        const double lower_x = lower_radius * std::cos(lower_angle);
        const double lower_y = lower_radius * std::sin(lower_angle);
        paintSphere(p, lower_x, lower_y, lower_z, bead_diameter_mm,
            kTungstenLabel);
        p.inserts.push_back({kTungstenLabel, bead_diameter_mm, 0.0,
            lower_x, lower_y, 0.0, lower_z});
    }

    // The marker is axially above upper-ring bead 0 and shares its X/Y.
    paintSphere(p, upper_radius, 0.0, marker_z, marker_diameter_mm,
        kTungstenLabel);
    p.inserts.push_back({kTungstenLabel, marker_diameter_mm, 0.0,
        upper_radius, 0.0, 0.0, marker_z});
    if (secondary_marker_phase_deg != 0.0) {
        const double marker_angle = secondary_marker_phase_deg * pi / 180.0;
        const double marker_x = upper_radius * std::cos(marker_angle);
        const double marker_y = upper_radius * std::sin(marker_angle);
        const double secondary_marker_z = upper_z + secondary_marker_offset_mm;
        paintSphere(p, marker_x, marker_y, secondary_marker_z, marker_diameter_mm,
            kTungstenLabel);
        p.inserts.push_back({kTungstenLabel, marker_diameter_mm, 0.0,
            marker_x, marker_y, 0.0, secondary_marker_z});
    }
    p.materials = {{kPmmaLabel, "pmma_support", 1.18, "C5H8O2", {},
        "cylindrical PMMA support"},
        {kTungstenLabel, "tungsten", 19.25, "W", {},
        "asymmetric double rings with one axial marker sphere"}};
    p.target_diameter_mm = bead_diameter_mm;
    p.support_diameter_mm = support_diameter_mm;
    p.support_length_mm = support_length_mm;
    p.bead_ring_diameter_mm = upper_ring_diameter_mm;
    p.lower_bead_ring_diameter_mm = lower_ring_diameter_mm;
    p.bead_ring_spacing_mm = ring_spacing_mm;
    p.lower_ring_phase_deg = lower_phase_deg;
    p.marker_bead_diameter_mm = marker_diameter_mm;
    p.marker_bead_offset_mm = marker_offset_mm;
    p.secondary_marker_phase_deg = secondary_marker_phase_deg;
    p.secondary_marker_offset_mm = secondary_marker_offset_mm;
    p.beads_per_ring = beads_per_ring;
    p.variant = "asymmetric tungsten bead rings with axial marker sphere";
}

void makeTungstenBeadSpiralMarker(Phantom& p, double ring_diameter_mm,
    double layer_spacing_mm, double phase_step_deg, double bead_diameter_mm,
    double marker_diameter_mm, double marker_below_mm, int bead_count)
{
    constexpr std::uint8_t kPmmaLabel = 1;
    constexpr std::uint8_t kTungstenLabel = 100;
    if (bead_count < 2)
        throw std::runtime_error("spiral bead count must be at least two");
    const double pi = std::acos(-1.0);
    const double radius = ring_diameter_mm * 0.5;
    const double bead_radius = bead_diameter_mm * 0.5;
    const double marker_radius = marker_diameter_mm * 0.5;
    const double first_z = -0.5 * (bead_count - 1) * layer_spacing_mm;
    const double marker_z = first_z - marker_below_mm;
    const double extent_z = std::max(std::abs(first_z) + bead_radius,
        std::abs(marker_z) + marker_radius);
    const double support_radius = radius + marker_radius + 2.0;
    const double support_length = 2.0 * (extent_z + 2.0);
    if (p.nx * p.voxel_mm < 2.0 * support_radius ||
        p.ny * p.voxel_mm < 2.0 * support_radius ||
        p.nz * p.voxel_mm < support_length)
        throw std::runtime_error("volume is smaller than spiral bead support");
    if (p.voxel_mm > bead_diameter_mm / 8.0)
        throw std::runtime_error("spiral bead phantom requires voxel_mm <= bead diameter/8");

    for (int z = 0; z < p.nz; ++z) {
        if (std::abs(p.z(z)) > support_length * 0.5) continue;
        for (int y = 0; y < p.ny; ++y) for (int x = 0; x < p.nx; ++x)
            if (p.x(x) * p.x(x) + p.y(y) * p.y(y) <=
                support_radius * support_radius)
                p.labels[p.index(x, y, z)] = kPmmaLabel;
    }

    for (int bead = 0; bead < bead_count; ++bead) {
        const double angle = bead * phase_step_deg * pi / 180.0;
        const double center_x = radius * std::cos(angle);
        const double center_y = radius * std::sin(angle);
        const double center_z = first_z + bead * layer_spacing_mm;
        paintSphere(p, center_x, center_y, center_z, bead_diameter_mm,
            kTungstenLabel);
        p.inserts.push_back({kTungstenLabel, bead_diameter_mm, 0.0,
            center_x, center_y, 0.0, center_z});
    }

    // The marker is directly below the first (lowest) spiral bead.
    paintSphere(p, radius, 0.0, marker_z, marker_diameter_mm,
        kTungstenLabel);
    p.inserts.push_back({kTungstenLabel, marker_diameter_mm, 0.0,
        radius, 0.0, 0.0, marker_z});
    p.materials = {{kPmmaLabel, "pmma_support", 1.18, "C5H8O2", {},
        "cylindrical PMMA support"},
        {kTungstenLabel, "tungsten", 19.25, "W", {},
        "helical tungsten bead line with axial marker sphere"}};
    p.target_diameter_mm = bead_diameter_mm;
    p.support_diameter_mm = 2.0 * support_radius;
    p.support_length_mm = support_length;
    p.bead_ring_diameter_mm = ring_diameter_mm;
    p.bead_ring_spacing_mm = layer_spacing_mm;
    p.lower_ring_phase_deg = phase_step_deg;
    p.marker_bead_diameter_mm = marker_diameter_mm;
    p.marker_bead_offset_mm = marker_below_mm;
    p.beads_per_ring = bead_count;
    p.spiral_beads = true;
    p.variant = "helical tungsten beads with 30-degree phase step and axial marker";
}

void makeGoldFoil(Phantom& p)
{
    constexpr double thickness = 0.05;
    requireTargetResolution(p, thickness * 0.5);
    const double radius = std::min(p.nx, p.ny) * p.voxel_mm * 0.35;
    for (int z = 0; z < p.nz; ++z) for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x) {
            const bool in_plane = p.x(x) * p.x(x) + p.y(y) * p.y(y) <= radius * radius;
            if (in_plane && std::abs(p.z(z)) <= thickness * 0.5)
                p.labels[p.index(x, y, z)] = 3;
        }
    p.materials = {{3, "gold", 19.32, "Au", {}, "z-MTF gold foil"}};
    p.target_thickness_mm = thickness;
    p.target_diameter_mm = radius * 2.0;
    p.variant = "0.05 mm gold foil perpendicular to z axis";
}

void makeLowContrast(Phantom& p)
{
    constexpr double thickness = 5.0;
    requireTargetResolution(p, 0.25);
    if (p.nz * p.voxel_mm < thickness)
        throw std::runtime_error("low_contrast z extent must be at least 5 mm");
    const double body = std::min(p.nx, p.ny) * p.voxel_mm * 0.49;
    for (int z = 0; z < p.nz; ++z) for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x)
            if (p.x(x) * p.x(x) + p.y(y) * p.y(y) <= body * body)
                p.labels[p.index(x, y, z)] = 1;
    p.materials.push_back({1, "water", 1.0, "H2O", {}, {}});
    const int diameters[] = {15, 9, 8, 7, 6, 5, 3};
    const double contrasts[] = {0.003, 0.005, 0.01};
    constexpr double gap = 0.25;
    std::vector<std::pair<double, double>> centers;
    for (int row = 0; row < 3; ++row) for (int column = 0; column < 7; ++column) {
        const int label = row + 2;
        const double radius = diameters[column] * 0.5;
        const double arm_phase = row * 2.0 * std::acos(-1.0) / 3.0;
        const double angle = arm_phase + column * 0.55;
        const double spiral_radius = 26.0 - column * 3.5;
        const double cx = spiral_radius * std::cos(angle);
        const double cy = spiral_radius * std::sin(angle);
        if (spiral_radius + radius > body)
            throw std::runtime_error("low-contrast spiral exceeds phantom body");
        for (std::size_t i = 0; i < centers.size(); ++i) {
            const double dx = cx - centers[i].first;
            const double dy = cy - centers[i].second;
            const double previous_radius = diameters[i % 7] * 0.5;
            if (dx * dx + dy * dy <
                (radius + previous_radius + gap) *
                (radius + previous_radius + gap))
                throw std::runtime_error("low-contrast spiral inserts overlap at " +
                    std::to_string(row) + "," + std::to_string(column) +
                    " with " + std::to_string(i));
        }
        centers.emplace_back(cx, cy);
        for (int z = 0; z < p.nz; ++z) for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x) {
                const double dx = p.x(x) - cx, dy = p.y(y) - cy;
                if (dx * dx + dy * dy <= radius * radius &&
                    std::abs(p.z(z)) <= thickness * 0.5)
                    p.labels[p.index(x, y, z)] = static_cast<std::uint8_t>(label);
            }
        const double contrast = contrasts[row];
        if (column == 0)
            p.materials.push_back({label,
                "water_minus_" + std::to_string(contrast * 100.0) + "pct",
                1.0 - contrast, "H2O", {}, "low-density contrast material"});
        p.inserts.push_back({label, static_cast<double>(diameters[column]),
            contrast, cx, cy, thickness});
    }
    p.variant = "21 water-equivalent low-contrast inserts";
}

void makeCatphan(Phantom& p)
{
    const double body = std::min(p.nx, p.ny) * p.voxel_mm * 0.45;
    for (int z = 0; z < p.nz; ++z) for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x)
            if (p.x(x) * p.x(x) + p.y(y) * p.y(y) <= body * body)
                p.labels[p.index(x, y, z)] = 1;
    p.materials = {{1, "pmma", 1.18, "C5H8O2", {}, {}}};
    const Material inserts[] = {
        {2, "air", .001225, "N2", {}, {}}, {3, "teflon", 2.2, "C2F4", {}, {}},
        {4, "delrin", 1.42, "CH2O", {}, {}}, {5, "acrylic", 1.18, "C5H8O2", {}, {}},
        {6, "polystyrene", 1.05, "C8H8", {}, {}}, {7, "ldpe", .92, "C2H4", {}, {}},
        {8, "water", 1.0, "H2O", {}, {}}};
    const double ring = body * 0.58, diameter = std::min(12.0, body * 0.22);
    for (int i = 0; i < 7; ++i) {
        const double angle = 2.0 * std::acos(-1.0) * i / 7.0;
        const double cx = ring * std::cos(angle), cy = ring * std::sin(angle);
        for (int z = 0; z < p.nz; ++z) for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x) {
                const double dx = p.x(x) - cx, dy = p.y(y) - cy;
                if (dx * dx + dy * dy <= diameter * diameter * 0.25)
                    p.labels[p.index(x, y, z)] = static_cast<std::uint8_t>(inserts[i].label);
            }
        p.materials.push_back(inserts[i]);
        p.inserts.push_back({inserts[i].label, diameter, 0.0, cx, cy});
    }
    p.variant = "multi-material Catphan-like phantom";
}

void makeSheppLogan(Phantom& p)
{
    struct E { int label; double cx, cy, cz, rx, ry, rz, angle; };
    const E ellipsoids[] = {
        {1, 0, 0, 0, .69, .92, .90, 0}, {2, 0, -.0184, 0, .6624, .874, .88, 0},
        {3, .22, 0, 0, .11, .31, .22, -18}, {3, -.22, 0, 0, .16, .41, .28, 18},
        {4, 0, .35, -.15, .21, .25, .25, 0}, {2, 0, .10, .25, .046, .046, .05, 0}};
    for (const auto& e : ellipsoids) for (int z = 0; z < p.nz; ++z) {
        const double zn = (z - (p.nz - 1) * .5) / (p.nz * .5);
        const double q = 1.0 - (zn - e.cz) * (zn - e.cz) / (e.rz * e.rz);
        if (q <= 0) continue;
        for (int y = 0; y < p.ny; ++y) for (int x = 0; x < p.nx; ++x) {
            const double xn = (x - (p.nx - 1) * .5) / (p.nx * .5);
            const double yn = (y - (p.ny - 1) * .5) / (p.ny * .5);
            if (ellipse(xn, yn, e.cx, e.cy, e.rx * std::sqrt(q),
                    e.ry * std::sqrt(q), e.angle))
                p.labels[p.index(x, y, z)] = static_cast<std::uint8_t>(e.label);
        }
    }
    p.materials = {
        {1, "soft_tissue", 1.0, {}, "tissue.icru_soft_tissue_adult", {}},
        {2, "brain", 1.04, {}, "tissue.icru_brain_adult", {}},
        {3, "bone", 1.61, {}, "tissue.ncat_skull", {}},
        {4, "water", 1.0, "H2O", {}, {}}};
    p.variant = "material-label 3D Shepp-Logan";
}

int integer(const char* text, const char* name) {
    const int value = std::stoi(text);
    if (value <= 0) throw std::runtime_error(std::string(name) + " must be positive");
    return value;
}

phantom_generator::GeneratorRegistry makeRegistry()
{
    using Arguments = std::vector<std::string>;
    phantom_generator::GeneratorRegistry r;
    r.add("shepp_logan", [](Phantom& p, const Arguments&) { makeSheppLogan(p); });
    r.add("tungsten_wire", [](Phantom& p, const Arguments&) { makeWire(p, false); });
    r.add("tungsten_wire_slanted", [](Phantom& p, const Arguments&) { makeWire(p, true); });
    r.add("gold_foil", [](Phantom& p, const Arguments&) { makeGoldFoil(p); });
    r.add("water_cylinder", [](Phantom& p, const Arguments&) { fillWaterShell(p, false); });
    r.add("water_cylinder_200mm", [](Phantom& p, const Arguments&) { makeWaterCylinder200mm(p); });
    r.add("water_ellipse", [](Phantom& p, const Arguments&) { fillWaterShell(p, true); });
    r.add("catphan", [](Phantom& p, const Arguments&) { makeCatphan(p); });
    r.add("low_contrast", [](Phantom& p, const Arguments&) { makeLowContrast(p); });
    r.add("tungsten_bead_line", [](Phantom& p, const Arguments& a) {
        if (a.size() != 2)
            throw std::runtime_error("tungsten_bead_line requires bead diameter and center spacing");
        makeTungstenBeadLine(p, std::stod(a[0]), std::stod(a[1]));
    });
    const auto ring = [](bool marker, bool replace) { return [=](Phantom& p, const Arguments& a) {
        if (a.size() != 5 && a.size() != 6) throw std::runtime_error("double-ring requires five dimensions and optional bead count");
        makeTungstenBeadDoubleRing(p, std::stod(a[0]), std::stod(a[1]), std::stod(a[2]), std::stod(a[3]), std::stod(a[4]), a.size() == 6 ? integer(a[5].c_str(), "beads_per_ring") : 6, marker, replace);
    }; };
    r.add("tungsten_bead_double_ring", ring(false, false));
    r.add("tungsten_bead_double_ring_cylinder_marker", ring(true, false));
    r.add("tungsten_bead_double_ring_cylinder_replace", ring(true, true));
    r.add("tungsten_bead_double_ring_marker", [](Phantom& p, const Arguments& a) {
        if (a.size() != 10 && a.size() != 12) throw std::runtime_error("marked double-ring requires ten parameters and optional secondary marker phase/height");
        makeTungstenBeadDoubleRingMarker(p, std::stod(a[0]), std::stod(a[1]), std::stod(a[2]), std::stod(a[3]), std::stod(a[4]), std::stod(a[5]), std::stod(a[6]), std::stod(a[7]), std::stod(a[8]), integer(a[9].c_str(), "beads_per_ring"), a.size() == 12 ? std::stod(a[10]) : 0.0, a.size() == 12 ? std::stod(a[11]) : 0.0);
    });
    r.add("tungsten_bead_spiral_marker", [](Phantom& p, const Arguments& a) {
        if (a.size() != 7) throw std::runtime_error("spiral requires seven parameters");
        makeTungstenBeadSpiralMarker(p, 2.0 * std::stod(a[0]), std::stod(a[1]), std::stod(a[2]), std::stod(a[3]), std::stod(a[4]), std::stod(a[5]), integer(a[6].c_str(), "bead_count"));
    });
    return r;
}

void printHelp(const std::string& type = {})
{
    if (type.empty()) {
        std::cout
            << "usage:\n"
            << "  phantom_generator --help [type]\n"
            << "  phantom_generator <type> <output.raw> <nx> <ny> <nz> <voxel_mm> [type arguments]\n\n"
            << "types:\n"
            << "  shepp_logan\n  tungsten_wire\n  tungsten_wire_slanted\n  gold_foil\n"
            << "  water_cylinder\n  water_cylinder_200mm\n  water_ellipse\n"
            << "  catphan\n  low_contrast\n  tungsten_bead_line\n"
            << "  tungsten_bead_double_ring\n"
            << "  tungsten_bead_double_ring_cylinder_marker\n"
            << "  tungsten_bead_double_ring_cylinder_replace\n"
            << "  tungsten_bead_double_ring_marker\n"
            << "  tungsten_bead_spiral_marker\n\n"
            << "Use --help <type> for type-specific arguments. All lengths are in mm.\n";
        return;
    }

    if (!makeRegistry().contains(type))
        throw std::invalid_argument("unknown phantom type: " + type);
    std::cout << "type: " << type << "\ncommon arguments:\n"
              << "  output.raw nx ny nz voxel_mm\n";
    if (type == "tungsten_bead_line")
        std::cout << "type arguments:\n  bead_diameter_mm center_spacing_mm\n"
                  << "example:\n  phantom_generator tungsten_bead_line beads.raw 256 256 960 0.125 3 10\n";
    else if (type == "tungsten_bead_double_ring" ||
             type == "tungsten_bead_double_ring_cylinder_marker" ||
             type == "tungsten_bead_double_ring_cylinder_replace")
        std::cout << "type arguments:\n"
                  << "  ring_diameter_mm ring_spacing_mm bead_diameter_mm\n"
                  << "  support_diameter_mm support_length_mm [beads_per_ring]\n";
    else if (type == "tungsten_bead_double_ring_marker")
        std::cout << "type arguments:\n"
                  << "  upper_diameter_mm lower_diameter_mm ring_spacing_mm bead_diameter_mm\n"
                  << "  lower_phase_deg marker_diameter_mm marker_above_mm\n"
                  << "  support_diameter_mm support_length_mm beads_per_ring\n"
                  << "  [secondary_marker_phase_deg secondary_marker_above_mm]\n";
    else if (type == "tungsten_bead_spiral_marker")
        std::cout << "type arguments:\n"
                  << "  spiral_radius_mm layer_spacing_mm phase_step_deg bead_diameter_mm\n"
                  << "  marker_diameter_mm marker_below_mm bead_count\n";
    else
        std::cout << "type arguments: none\n";
}

}

int main(int argc, char** argv)
{
    try {
        if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            printHelp(argc >= 3 ? argv[2] : "");
            return 0;
        }
        if (argc < 7) {
            printHelp();
            return 2;
        }
        Phantom p;
        p.type = argv[1];
        p.nx = integer(argv[3], "nx"); p.ny = integer(argv[4], "ny");
        p.nz = integer(argv[5], "nz"); p.voxel_mm = std::stod(argv[6]);
        if (!(p.voxel_mm > 0)) throw std::runtime_error("voxel_mm must be positive");
        p.labels.assign(static_cast<std::size_t>(p.nx) * p.ny * p.nz, 0);
        std::vector<std::string> arguments;
        for (int i = 7; i < argc; ++i) arguments.emplace_back(argv[i]);
        makeRegistry().generate(p.type, p, arguments);
        phantom_generator::PhantomOutputWriter::write(p, argv[2]);
        std::cout << "output=" << argv[2] << " size=" << p.nx << 'x' << p.ny
                  << 'x' << p.nz << " materials=" << p.materials.size() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "phantom_generator: " << e.what() << '\n';
        return 1;
    }
}
