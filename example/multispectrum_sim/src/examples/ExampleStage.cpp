#include "examples/ExampleStage.hpp"
#include <algorithm>
#include <cmath>
namespace yk::spectral::examples {
namespace {
    constexpr double pi = 3.14159265358979323846;
    enum class Model { WaterCircle,
        WaterHelix,
        Line,
        Asymmetric,
        CylinderMarker,
        Spiral,
        Sync6,
        Sync12 };
    MaterialSpec material(int label, const char* name, const char* formula, double density)
    {
        MaterialSpec m;
        m.label = static_cast<std::uint8_t>(label);
        m.name = name;
        m.formula = formula;
        m.density_g_cm3 = density;
        return m;
    }
    void paint(ExampleResult& result, const Landmark& shape)
    {
        const auto& g = result.config.geometry.parameters;
        const int dims[] = { g.volume_x, g.volume_y, g.volume_z };
        const double step = g.voxel_x_mm;
        int lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
            double extent = (a == 2 && shape.shape == Shape::CylinderZ) ? shape.half_height_mm : shape.radius_mm;
            lo[a] = std::max(0, int(std::floor((shape.center_mm[a] - extent) / step + (dims[a] - 1) * .5)));
            hi[a] = std::min(dims[a] - 1, int(std::ceil((shape.center_mm[a] + extent) / step + (dims[a] - 1) * .5)));
        }
        for (int z = lo[2]; z <= hi[2]; ++z)
            for (int y = lo[1]; y <= hi[1]; ++y)
                for (int x = lo[0]; x <= hi[0]; ++x) {
                    const double dx = (x - (dims[0] - 1) * .5) * step - shape.center_mm[0], dy = (y - (dims[1] - 1) * .5) * step - shape.center_mm[1], dz = (z - (dims[2] - 1) * .5) * step - shape.center_mm[2];
                    const bool inside = shape.shape == Shape::CylinderZ ? dx * dx + dy * dy <= shape.radius_mm * shape.radius_mm && std::abs(dz) <= shape.half_height_mm : dx * dx + dy * dy + dz * dz <= shape.radius_mm * shape.radius_mm;
                    if (inside)
                        result.labels[(size_t(z) * dims[1] + y) * dims[0] + x] = shape.label;
                }
    }
    class GeometryExample final : public IExampleStage {
        pipeline::PluginInfo metadata_;
        Model model_;

    public:
        GeometryExample(Model model, std::string id, std::string name, std::string description)
            : model_(model)
        {
            metadata_.id = std::move(id);
            metadata_.name = std::move(name);
            metadata_.description = std::move(description);
            metadata_.version = "1.0";
            metadata_.kind = pipeline::PluginKind::ExampleStage;
            metadata_.category = "builtin_examples";
            metadata_.tags = { "in_memory", "generated", "polychromatic" };
        }
        const pipeline::PluginInfo& info() const noexcept override { return metadata_; }
        ScanPreset scanPreset() const override
        {
            if (model_ == Model::WaterHelix)
                return { "Water / helical / 2 turns", GeometryKind::FlatHelical, 720, 360, 1024, 256, 709, 1143, .4, .8, 40, -40, "wfbp" };
            if (model_ == Model::WaterCircle)
                return { "Water / circular / 360 degrees", GeometryKind::FlatCbct, 360, 360, 1024, 256, 709, 1143, .4, 1.6, 0, 0, "fdk" };
            return { "Geometry calibration / circular / 360 degrees", GeometryKind::FlatCbct, 360, 360, 1024, 256, 440, 770, .417, 1.668, 0, 0, "fdk" };
        }
        std::vector<Parameter> parameters() const override
        {
            const bool water = model_ == Model::WaterCircle || model_ == Model::WaterHelix;
            const bool line = model_ == Model::Line, spiral = model_ == Model::Spiral;
            const bool sync = model_ == Model::Sync6 || model_ == Model::Sync12;
            const bool cylinder = model_ == Model::CylinderMarker;
            std::vector<Parameter> p;
            auto add = [&](const char* id, const char* name, const char* group, double value, double lo, double hi, bool integer = false) { p.push_back({ id, name, group, value, lo, hi, integer }); };
            add("support_diameter", "Cylinder diameter (mm)", "Body", water ? 200 : line ? 30
                    : spiral                                                             ? 109
                    : sync                                                               ? 110
                    : cylinder                                                           ? 108
                                                                                         : 106,
                1, 500);
            add("support_height", "Cylinder height (mm)", "Body", water ? 90 : line ? 120
                    : spiral                                                        ? 144
                    : cylinder                                                      ? 130
                                                                                    : 170,
                1, 500);
            if (!water) {
                add("bead_diameter", "Bead diameter (mm)", "Beads", 3, .1, 20);
                add("count", line ? "Bead count" : spiral ? "Bead count"
                                                          : "Beads per ring",
                    "Beads", line ? 12 : spiral                                    ? 24
                        : (model_ == Model::Asymmetric || model_ == Model::Sync12) ? 12
                                                                                   : 6,
                    1, 256, true);
                if (line || spiral)
                    add("spacing", "Axial spacing (mm)", "Beads", line ? 10 : 5, .1, 100);
                if (!line)
                    add("radius", "Upper ring / spiral radius (mm)", "Beads", 50, 1, 200);
                if (spiral)
                    add("angle_step", "Angular step (deg)", "Beads", 30, -360, 360);
                if (!line && !spiral) {
                    add("lower_radius", "Lower ring radius (mm)", "Beads", model_ == Model::Asymmetric ? 40 : 50, 1, 200);
                    add("separation", "Ring separation (mm)", "Beads", 100, 1, 400);
                    add("phase", "Lower ring phase (deg)", "Beads", model_ == Model::Asymmetric ? 15 : 0, -360, 360);
                }
                if (!line) {
                    add("marker_diameter", "Marker diameter (mm)", "Markers", 5, .1, 30);
                    add("marker_z", "Marker Z / cylinder |Z| (mm)", "Markers", spiral ? -67.5 : cylinder ? 60
                                                                                                         : 80,
                        -250, 250);
                    if (cylinder)
                        add("marker_height", "Marker height (mm)", "Markers", 5, .1, 40);
                    if (sync) {
                        add("marker2_z", "Second marker Z (mm)", "Markers", 70, -250, 250);
                        add("marker2_angle", "Second marker angle (deg)", "Markers", model_ == Model::Sync12 ? 15 : 30, -360, 360);
                    }
                }
            }
            add("voxel", "Voxel spacing (mm)", "Grid", water ? 2 : line ? .125
                                                                        : .2,
                .05, 5);
            add("nx", "Grid X", "Grid", water ? 128 : line ? 256
                    : spiral                               ? 560
                    : sync                                 ? 550
                                                           : 540,
                8, 2048, true);
            add("ny", "Grid Y", "Grid", water ? 128 : line ? 256
                    : spiral                               ? 560
                    : sync                                 ? 550
                                                           : 540,
                8, 2048, true);
            add("nz", "Grid Z", "Grid", water ? 64 : line ? 960
                    : spiral                              ? 760
                                                          : 900,
                8, 2048, true);
            return p;
        }
        ExampleResult generate(const ExampleRequest& request) const override
        {
            if (request.output_directory.empty() || request.views < 0 || request.views > 10000)
                throw std::invalid_argument("invalid example output directory/views");
            std::map<std::string, double> values;
            for (const auto& p : parameters()) {
                const auto found = request.parameters.find(p.id);
                double v = found == request.parameters.end() ? p.value : found->second;
                if (!std::isfinite(v) || v < p.minimum || v > p.maximum || (p.integer && std::floor(v) != v))
                    throw std::invalid_argument("Invalid phantom parameter: " + p.name);
                values[p.id] = v;
            }
            for (const auto& item : request.parameters)
                if (!values.count(item.first))
                    throw std::invalid_argument("Unknown phantom parameter: " + item.first);
            auto v = [&](const char* id) { return values.at(id); };
            ExampleResult result;
            auto& c = result.config;
            auto& g = c.geometry.parameters;
            const bool water = model_ == Model::WaterCircle || model_ == Model::WaterHelix;
            const auto preset = scanPreset();
            const auto directory = std::filesystem::absolute(request.output_directory).lexically_normal();
            c.projection.output_file = directory / "projection.raw";
            c.projection.energy_output_file = directory / "energy.raw";
            c.reconstruction.input_projection_file = c.projection.output_file;
            c.reconstruction.output_volume_file = directory / "volume.raw";
            c.reconstruction.slice_prefix = directory / "slice";
            c.workflow.mode = WorkflowMode::Project;
            c.geometry.kind = preset.kind;
            c.projection.spectrum.source = "generated";
            c.projection.spectrum.model = "spekpy-v1";
            c.projection.spectrum.kvp = 120;
            c.projection.spectrum.bin_width_keV = 2;
            SpectrumFilterConfig filter;
            filter.material = "Al";
            filter.thickness_mm = 2;
            c.projection.spectrum.filters = { filter };
            c.projection.quantum_noise.enabled = false;
            c.projection.sampling.samples_per_pixel = 1;
            c.projection.scatter.enabled = false;
            c.reconstruction.analytic.pipeline = preset.reconstruction;
            c.reconstruction.analytic.filter = "hann";
            g.views = request.views ? request.views : preset.views;
            g.views_per_turn = preset.kind == GeometryKind::FlatHelical ? preset.views_per_turn : g.views;
            g.sid_mm = preset.sid_mm;
            g.sdd_mm = preset.sdd_mm;
            g.detector_u = preset.detector_u;
            g.detector_v = preset.detector_v;
            g.pixel_u_mm = preset.pixel_u_mm;
            g.pixel_v_mm = preset.pixel_v_mm;
            g.pitch_mm_per_turn = preset.pitch_mm_per_turn;
            g.start_z_mm = preset.start_z_mm;
            g.volume_x = g.volume_y = water ? 128 : 540;
            g.volume_z = water ? 64 : 900;
            g.voxel_x_mm = g.voxel_y_mm = g.voxel_z_mm = water ? 2 : .2;
            result.support_radius_mm = water ? 100 : 53;
            result.support_half_height_mm = water ? 45 : 85;
            if (water)
                c.projection.materials = { material(1, "water", "H2O", 1) };
            else {
                c.projection.materials = { material(1, "pmma_support", "C5H8O2", 1.18), material(100, "tungsten", "W", 19.25) };
                g.offset_u_mm = 2.085;
                g.offset_v_mm = 4.17;
                g.tilt_u_rad = model_ == Model::Line ? 0 : pi / 180;
                g.tilt_v_rad = model_ == Model::Line ? 0 : 2 * pi / 180;
                g.tilt_n_rad = 3 * pi / 180;
                g.source_offset_y_mm = 10;
                g.source_offset_z_mm = 5;
                g.phantom_offset_x_mm = model_ == Model::Line ? 35 : 10;
                g.phantom_offset_y_mm = model_ == Model::Line ? 28 : 15;
                g.phantom_offset_z_mm = 20;
                g.phantom_rotation_x_rad = pi / 180;
                g.phantom_rotation_y_rad = 2 * pi / 180;
                g.phantom_rotation_z_rad = 3 * pi / 180;
            }
            g.volume_x = int(v("nx"));
            g.volume_y = int(v("ny"));
            g.volume_z = int(v("nz"));
            g.voxel_x_mm = g.voxel_y_mm = g.voxel_z_mm = v("voxel");
            result.support_radius_mm = v("support_diameter") / 2;
            result.support_half_height_mm = v("support_height") / 2;
            auto sphere = [&](double x, double y, double z, double r) { result.landmarks.push_back({ Shape::Sphere, { x, y, z }, r, 0, 100 }); };
            if (!water) {
                const int count = int(v("count"));
                const double r = v("bead_diameter") / 2;
                if (model_ == Model::Line) {
                    for (int i = 0; i < count; ++i)
                        sphere(0, 0, (i - (count - 1) * .5) * v("spacing"), r);
                } else if (model_ == Model::Spiral) {
                    for (int i = 0; i < count; ++i) {
                        double a = i * v("angle_step") * pi / 180;
                        sphere(v("radius") * std::cos(a), v("radius") * std::sin(a), (i - (count - 1) * .5) * v("spacing"), r);
                    }
                    sphere(v("radius"), 0, v("marker_z"), v("marker_diameter") / 2);
                } else {
                    for (int i = 0; i < count; ++i) {
                        double a = i * 2 * pi / count, b = a + v("phase") * pi / 180;
                        sphere(v("radius") * std::cos(a), v("radius") * std::sin(a), v("separation") / 2, r);
                        sphere(v("lower_radius") * std::cos(b), v("lower_radius") * std::sin(b), -v("separation") / 2, r);
                    }
                    if (model_ == Model::CylinderMarker) {
                        for (double sign : { -1., 1. })
                            result.landmarks.push_back({ Shape::CylinderZ, { v("radius"), 0, sign * v("marker_z") }, v("marker_diameter") / 2, v("marker_height") / 2, 100 });
                    } else {
                        sphere(v("radius"), 0, v("marker_z"), v("marker_diameter") / 2);
                        if (model_ == Model::Sync6 || model_ == Model::Sync12) {
                            double a = v("marker2_angle") * pi / 180;
                            sphere(v("radius") * std::cos(a), v("radius") * std::sin(a), v("marker2_z"), v("marker_diameter") / 2);
                        }
                    }
                }
            }
            const double extent[] = { g.volume_x * v("voxel") / 2, g.volume_y * v("voxel") / 2, g.volume_z * v("voxel") / 2 };
            if (result.support_radius_mm > std::min(extent[0], extent[1]) + 1e-8 || result.support_half_height_mm > extent[2] + 1e-8)
                throw std::invalid_argument("Body exceeds grid. Increase grid dimensions or voxel spacing.");
            for (const auto& l : result.landmarks)
                for (int axis = 0; axis < 3; ++axis) {
                    double radius = axis == 2 && l.shape == Shape::CylinderZ ? l.half_height_mm : l.radius_mm;
                    if (std::abs(l.center_mm[axis]) + radius > extent[axis] + 1e-8)
                        throw std::invalid_argument("A marker/bead exceeds the volume grid.");
                }
            if (size_t(g.volume_x) * g.volume_y * g.volume_z > size_t(512) * 1024 * 1024)
                throw std::invalid_argument("Example label volume exceeds 512 MiB; reduce the grid.");
            // Reduce only the reconstruction grid; retain the source labels and physical extent.
            g.reconstruction_volume_x = (g.volume_x + 1) / 2;
            g.reconstruction_volume_y = (g.volume_y + 1) / 2;
            g.reconstruction_volume_z = (g.volume_z + 1) / 2;
            g.reconstruction_voxel_x_mm = g.volume_x * g.voxel_x_mm / g.reconstruction_volume_x;
            g.reconstruction_voxel_y_mm = g.volume_y * g.voxel_y_mm / g.reconstruction_volume_y;
            g.reconstruction_voxel_z_mm = g.volume_z * g.voxel_z_mm / g.reconstruction_volume_z;
            result.labels.resize(size_t(g.volume_x) * g.volume_y * g.volume_z, 0);
            paint(result, { Shape::CylinderZ, { 0, 0, 0 }, result.support_radius_mm, result.support_half_height_mm, 1 });
            for (const auto& landmark : result.landmarks)
                paint(result, landmark);
            c.reconstruction.geometry_override.parameters = g;
            return result;
        }
    };
}
ExampleRegistry builtinExamples()
{
    ExampleRegistry r;
    auto add = [&](Model model, const char* id, const char* name, const char* description) {GeometryExample sample(model,id,name,description);r.add(sample.info(),[=]{return std::make_unique<GeometryExample>(model,id,name,description);}); };
    add(Model::WaterCircle, "water-circle", "Water 20 cm / circular", "120 kVp, SID 709 / SDD 1143; in-memory water cylinder.");
    add(Model::WaterHelix, "water-helix", "Water 20 cm / helical", "Two turns, 360 views per turn, 40 mm per turn; WFBP setup.");
    add(Model::Line, "bead-line", "Beads / straight line", "3 mm beads, 10 mm spacing; PMMA support 30 x 120 mm; 0.125 mm voxels.");
    add(Model::Asymmetric, "bead-double-ring", "Beads / asymmetric double ring", "100/80 mm rings, 100 mm separation, 15 degree phase; 12 beads per ring plus 5 mm marker.");
    add(Model::CylinderMarker, "bead-cylinder-marker", "Beads / cylinder markers", "Aligned six-bead rings plus two separate 5 x 5 mm cylinders, offset 10 mm.");
    add(Model::Spiral, "bead-spiral", "Beads / spiral", "24 beads, radius 50 mm, axial spacing 5 mm, phase step 30 degrees; lower marker.");
    add(Model::Sync6, "bead-sync-6", "Beads / aligned rings (6)", "Six beads per ring and two marker spheres; 0.2 mm voxels.");
    add(Model::Sync12, "bead-sync-12", "Beads / aligned rings (12)", "Twelve beads per ring and two marker spheres; 0.2 mm voxels.");
    return r;
}
}
