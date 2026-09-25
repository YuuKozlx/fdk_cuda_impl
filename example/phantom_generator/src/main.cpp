#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Material {
    int label;
    std::string name;
    double density;
    std::string formula;
    std::string preset;
    std::string description;
};

struct Insert {
    int label;
    double diameter_mm;
    double contrast_fraction;
    double center_x_mm;
    double center_y_mm;
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
    double bead_ring_spacing_mm = 0.0;
    int beads_per_ring = 0;

    std::size_t index(int x, int y, int z) const {
        return (static_cast<std::size_t>(z) * ny + y) * nx + x;
    }
    double x(int i) const { return (i - (nx - 1) * 0.5) * voxel_mm; }
    double y(int i) const { return (i - (ny - 1) * 0.5) * voxel_mm; }
    double z(int i) const { return (i - (nz - 1) * 0.5) * voxel_mm; }
};

std::string tomlString(std::string value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') escaped.push_back('\\');
        escaped.push_back(ch);
    }
    return escaped;
}

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
    double support_diameter_mm, double support_length_mm, int beads_per_ring)
{
    constexpr std::uint8_t kPmmaLabel = 1;
    constexpr std::uint8_t kTungstenLabel = 100;
    const double support_radius = support_diameter_mm * 0.5;
    const double ring_radius = ring_diameter_mm * 0.5;
    const double bead_radius = bead_diameter_mm * 0.5;
    if (beads_per_ring < 1)
        throw std::runtime_error("beads_per_ring must be positive");
    if (ring_radius + bead_radius > support_radius ||
        ring_spacing_mm * 0.5 + bead_radius > support_length_mm * 0.5)
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
            const int zmin = std::max(0, static_cast<int>(std::floor(
                (center_z - bead_radius) / p.voxel_mm + (p.nz - 1) * 0.5)));
            const int zmax = std::min(p.nz - 1, static_cast<int>(std::ceil(
                (center_z + bead_radius) / p.voxel_mm + (p.nz - 1) * 0.5)));
            for (int z = zmin; z <= zmax; ++z)
                for (int y = ymin; y <= ymax; ++y)
                    for (int x = xmin; x <= xmax; ++x) {
                        const double dx = p.x(x) - center_x;
                        const double dy = p.y(y) - center_y;
                        const double dz = p.z(z) - center_z;
                        if (dx * dx + dy * dy + dz * dz <= bead_radius * bead_radius)
                            p.labels[p.index(x, y, z)] = kTungstenLabel;
                    }
            p.inserts.push_back({kTungstenLabel, bead_diameter_mm, 0.0,
                center_x, center_y, 0.0, center_z});
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
    p.variant = "two aligned " + std::to_string(beads_per_ring) +
        "-bead tungsten rings in a PMMA cylinder";
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

void writeOutputs(const Phantom& p, const std::filesystem::path& output)
{
    for (const auto& material : p.materials) {
        if (material.label <= 0 || material.label > 255)
            throw std::runtime_error("material label must be in [1, 255]");
        if (std::find(p.labels.begin(), p.labels.end(),
                static_cast<std::uint8_t>(material.label)) == p.labels.end())
            throw std::runtime_error("material label is absent from volume: " +
                std::to_string(material.label));
    }
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    std::ofstream raw(output, std::ios::binary);
    raw.write(reinterpret_cast<const char*>(p.labels.data()),
        static_cast<std::streamsize>(p.labels.size()));
    if (!raw) throw std::runtime_error("failed to write label RAW");
    auto metadata = output; metadata += ".toml";
    std::ofstream toml(metadata);
    toml << "[phantom]\n"
         << "type = \"" << p.type << "\"\n"
         << "variant = \"" << p.variant << "\"\n"
         << "label_volume = \"" << tomlString(output.filename().generic_string()) << "\"\n"
         << "columns = " << p.nx << "\nrows = " << p.ny
         << "\nslices = " << p.nz << "\nvoxel_x_mm = " << p.voxel_mm
         << "\nvoxel_y_mm = " << p.voxel_mm
         << "\nvoxel_z_mm = " << p.voxel_mm
         << "\ndata_type = \"uint8\"\nlayout = \"slice_row_column\"\n\n";
    if (p.target_diameter_mm > 0.0 || p.target_thickness_mm > 0.0) {
        toml << "[quality_target]\n";
        if (p.target_diameter_mm > 0.0)
            toml << "diameter_mm = " << p.target_diameter_mm << '\n';
        if (p.target_thickness_mm > 0.0)
            toml << "thickness_mm = " << p.target_thickness_mm << '\n';
        if (p.target_tangent > 0.0)
            toml << "tangent = " << p.target_tangent << '\n';
        toml << '\n';
    }
    if (p.bead_spacing_mm > 0.0) {
        toml << "[bead_line]\n"
             << "diameter_mm = " << p.target_diameter_mm << '\n'
             << "center_spacing_mm = " << p.bead_spacing_mm << '\n'
             << "support_diameter_mm = " << p.support_diameter_mm << '\n'
             << "support_length_mm = " << p.support_length_mm << "\n\n";
    }
    if (p.bead_ring_diameter_mm > 0.0) {
        toml << "[bead_rings]\n"
             << "ring_count = 2\nbeads_per_ring = " << p.beads_per_ring << '\n'
             << "ring_diameter_mm = " << p.bead_ring_diameter_mm << '\n'
             << "ring_center_spacing_mm = " << p.bead_ring_spacing_mm << '\n'
             << "bead_diameter_mm = " << p.target_diameter_mm << '\n'
             << "support_diameter_mm = " << p.support_diameter_mm << '\n'
             << "support_length_mm = " << p.support_length_mm << "\n\n";
    }
    for (const auto& insert : p.inserts) {
        toml << "[[phantom.inserts]]\nlabel = " << insert.label
             << "\ndiameter_mm = " << insert.diameter_mm
             << "\ncontrast_fraction = " << insert.contrast_fraction
             << "\ncenter_mm = [" << insert.center_x_mm << ", "
             << insert.center_y_mm << "]\n";
        if (insert.thickness_mm > 0.0)
            toml << "thickness_mm = " << insert.thickness_mm << '\n';
        if (insert.center_z_mm != 0.0)
            toml << "center_z_mm = " << insert.center_z_mm << '\n';
        toml << '\n';
    }
    for (const auto& m : p.materials) {
        toml << "[[projection.materials]]\nlabel = " << m.label
             << "\nname = \"" << m.name << "\"\ndensity_g_cm3 = " << m.density << '\n';
        if (!m.preset.empty()) toml << "preset = \"" << m.preset << "\"\n";
        else toml << "formula = \"" << m.formula << "\"\n";
        if (!m.description.empty()) toml << "# " << m.description << '\n';
        toml << '\n';
    }
    if (!toml) throw std::runtime_error("failed to write phantom TOML");
}

int integer(const char* text, const char* name) {
    const int value = std::stoi(text);
    if (value <= 0) throw std::runtime_error(std::string(name) + " must be positive");
    return value;
}

}

int main(int argc, char** argv)
{
    try {
        if (argc != 7 && argc != 12 && argc != 13) {
            std::cerr << "usage: phantom_generator <type> <output.raw> <nx> <ny> <nz> <voxel_mm>\n"
                << "types: shepp_logan tungsten_wire tungsten_wire_slanted gold_foil water_cylinder water_cylinder_200mm "
                   "water_ellipse catphan low_contrast tungsten_bead_line_2mm_5mm "
                   "tungsten_bead_line_3mm_10mm tungsten_bead_double_ring\n"
                << "double-ring extra args: ring_diameter_mm ring_spacing_mm bead_diameter_mm support_diameter_mm support_length_mm [beads_per_ring]\n";
            return 2;
        }
        Phantom p;
        p.type = argv[1];
        p.nx = integer(argv[3], "nx"); p.ny = integer(argv[4], "ny");
        p.nz = integer(argv[5], "nz"); p.voxel_mm = std::stod(argv[6]);
        if (!(p.voxel_mm > 0)) throw std::runtime_error("voxel_mm must be positive");
        p.labels.assign(static_cast<std::size_t>(p.nx) * p.ny * p.nz, 0);
        if (p.type == "shepp_logan") makeSheppLogan(p);
        else if (p.type == "tungsten_wire") makeWire(p, false);
        else if (p.type == "tungsten_wire_slanted") makeWire(p, true);
        else if (p.type == "gold_foil") makeGoldFoil(p);
        else if (p.type == "water_cylinder") fillWaterShell(p, false);
        else if (p.type == "water_cylinder_200mm") makeWaterCylinder200mm(p);
        else if (p.type == "water_ellipse") fillWaterShell(p, true);
        else if (p.type == "catphan") makeCatphan(p);
        else if (p.type == "low_contrast") makeLowContrast(p);
        else if (p.type == "tungsten_bead_line_2mm_5mm")
            makeTungstenBeadLine(p, 2.0, 5.0);
        else if (p.type == "tungsten_bead_line_3mm_10mm")
            makeTungstenBeadLine(p, 3.0, 10.0);
        else if (p.type == "tungsten_bead_double_ring") {
            if (argc != 12 && argc != 13) throw std::runtime_error(
                "tungsten_bead_double_ring requires five dimensions and optional bead count");
            makeTungstenBeadDoubleRing(p, std::stod(argv[7]),
                std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10]),
                std::stod(argv[11]), argc == 13 ? integer(argv[12],
                    "beads_per_ring") : 6);
        }
        else throw std::runtime_error("unknown phantom type: " + p.type);
        writeOutputs(p, argv[2]);
        std::cout << "output=" << argv[2] << " size=" << p.nx << 'x' << p.ny
                  << 'x' << p.nz << " materials=" << p.materials.size() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "phantom_generator: " << e.what() << '\n';
        return 1;
    }
}
