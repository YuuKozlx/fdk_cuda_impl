#include "phantom_generator/PhantomOutput.hpp"

#include <fstream>
#include <stdexcept>

namespace phantom_generator {
namespace {

std::string tomlString(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') escaped.push_back('\\');
        escaped.push_back(ch);
    }
    return escaped;
}

}

void PhantomOutputWriter::validate(const Phantom& phantom)
{
    if (phantom.nx <= 0 || phantom.ny <= 0 || phantom.nz <= 0)
        throw std::runtime_error("phantom dimensions must be positive");
    const auto expected = static_cast<std::size_t>(phantom.nx) * phantom.ny * phantom.nz;
    if (phantom.labels.size() != expected)
        throw std::runtime_error("label volume size does not match phantom dimensions");
    for (const auto& material : phantom.materials) {
        if (material.label < 1 || material.label > 255)
            throw std::runtime_error("material label must be in [1, 255]");
    }
}

void PhantomOutputWriter::writeRaw(const Phantom& phantom,
    const std::filesystem::path& output)
{
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    std::ofstream raw(output, std::ios::binary);
    raw.write(reinterpret_cast<const char*>(phantom.labels.data()),
        static_cast<std::streamsize>(phantom.labels.size()));
    if (!raw) throw std::runtime_error("failed to write label RAW");
}

void PhantomOutputWriter::writeDescription(const Phantom& p,
    const std::filesystem::path& output)
{
    auto metadata = output;
    metadata += ".toml";
    std::ofstream toml(metadata);
    toml << "[phantom]\n"
         << "type = \"" << tomlString(p.type) << "\"\n"
         << "variant = \"" << tomlString(p.variant) << "\"\n"
         << "label_volume = \"" << tomlString(output.filename().generic_string()) << "\"\n"
         << "columns = " << p.nx << "\nrows = " << p.ny << "\nslices = " << p.nz
         << "\nvoxel_x_mm = " << p.voxel_mm << "\nvoxel_y_mm = " << p.voxel_mm
         << "\nvoxel_z_mm = " << p.voxel_mm << "\ndata_type = \"uint8\"\nlayout = \"slice_row_column\"\n\n";
    if (p.target_diameter_mm > 0.0 || p.target_thickness_mm > 0.0) {
        toml << "[quality_target]\n";
        if (p.target_diameter_mm > 0.0) toml << "diameter_mm = " << p.target_diameter_mm << '\n';
        if (p.target_thickness_mm > 0.0) toml << "thickness_mm = " << p.target_thickness_mm << '\n';
        if (p.target_tangent > 0.0) toml << "tangent = " << p.target_tangent << '\n';
        toml << '\n';
    }
    if (p.bead_spacing_mm > 0.0) {
        toml << "[bead_line]\ndiameter_mm = " << p.target_diameter_mm
             << "\ncenter_spacing_mm = " << p.bead_spacing_mm
             << "\nsupport_diameter_mm = " << p.support_diameter_mm
             << "\nsupport_length_mm = " << p.support_length_mm << "\n\n";
    }
    if (p.bead_ring_diameter_mm > 0.0) {
        if (p.spiral_beads) {
            toml << "[bead_spiral]\nbead_count = " << p.beads_per_ring
                 << "\nspiral_radius_mm = " << p.bead_ring_diameter_mm * 0.5
                 << "\nlayer_spacing_mm = " << p.bead_ring_spacing_mm
                 << "\nphase_step_deg = " << p.lower_ring_phase_deg
                 << "\nbead_diameter_mm = " << p.target_diameter_mm
                 << "\nmarker_bead_diameter_mm = " << p.marker_bead_diameter_mm
                 << "\nmarker_below_lowest_bead_mm = " << p.marker_bead_offset_mm
                 << "\nsupport_diameter_mm = " << p.support_diameter_mm
                 << "\nsupport_length_mm = " << p.support_length_mm << "\n\n";
        } else {
            toml << "[bead_rings]\nring_count = 2\nbeads_per_ring = " << p.beads_per_ring
                 << "\nupper_ring_diameter_mm = " << p.bead_ring_diameter_mm;
            if (p.lower_bead_ring_diameter_mm > 0.0)
                toml << "\nlower_ring_diameter_mm = " << p.lower_bead_ring_diameter_mm
                     << "\nlower_ring_phase_deg = " << p.lower_ring_phase_deg;
            else toml << "\nring_diameter_mm = " << p.bead_ring_diameter_mm;
            toml << "\nring_center_spacing_mm = " << p.bead_ring_spacing_mm
                 << "\nbead_diameter_mm = " << p.target_diameter_mm
                 << "\nsupport_diameter_mm = " << p.support_diameter_mm
                 << "\nsupport_length_mm = " << p.support_length_mm << '\n';
            if (p.marker_bead_diameter_mm > 0.0)
                toml << "marker_bead_diameter_mm = " << p.marker_bead_diameter_mm
                     << "\nmarker_above_upper_ring_mm = " << p.marker_bead_offset_mm << '\n';
            if (p.cylindrical_ring_markers)
                toml << "marker_shape = \"cylinder\"\nmarker_phase_deg = 0\n"
                     << "marker_cylinder_diameter_mm = "
                     << (p.marker_cylinder_height_mm > p.target_diameter_mm ? 5.0 : p.target_diameter_mm)
                     << "\nmarker_cylinder_height_mm = " << p.marker_cylinder_height_mm
                     << "\nmarker_center_offset_mm = " << p.marker_bead_offset_mm << '\n';
            toml << '\n';
        }
    }
    for (const auto& insert : p.inserts) {
        toml << "[[phantom.inserts]]\nlabel = " << insert.label
             << "\ndiameter_mm = " << insert.diameter_mm
             << "\ncontrast_fraction = " << insert.contrast_fraction
             << "\ncenter_mm = [" << insert.center_x_mm << ", " << insert.center_y_mm << "]\n";
        if (insert.thickness_mm > 0.0) toml << "thickness_mm = " << insert.thickness_mm << '\n';
        if (insert.center_z_mm != 0.0) toml << "center_z_mm = " << insert.center_z_mm << '\n';
        toml << '\n';
    }
    for (const auto& material : p.materials) {
        toml << "[[projection.materials]]\nlabel = " << material.label
             << "\nname = \"" << tomlString(material.name) << "\"\ndensity_g_cm3 = "
             << material.density << '\n';
        if (!material.preset.empty()) toml << "preset = \"" << tomlString(material.preset) << "\"\n";
        else toml << "formula = \"" << tomlString(material.formula) << "\"\n";
        if (!material.description.empty()) toml << "# " << material.description << '\n';
        toml << '\n';
    }
    if (!toml) throw std::runtime_error("failed to write phantom TOML");
}

void PhantomOutputWriter::write(const Phantom& phantom, const std::filesystem::path& output)
{
    validate(phantom);
    writeRaw(phantom, output);
    writeDescription(phantom, output);
}

} // namespace phantom_generator
