#include "MultiSpecFpStage.hpp"
#include "../src/primary/CudaProjectionBackend.hpp"
#include "../src/primary/ProjectionGeometry.cuh"
#include "SpectralProjectionPipeline.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLPolyDataReader.h>
using namespace yk::spectral;
namespace fs = std::filesystem;
namespace {
void require(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}
void near(double a, double b, double tolerance, const char* message) { require(std::abs(a - b) < tolerance, message); }
struct Attenuation final : IMassAttenuationProvider {
    bool query(const MaterialSpec& m, const std::vector<double>& e, std::vector<double>& values, std::string&) const override
    {
        values.assign(e.size(), m.label == 1 ? .15 : 1.2);
        return true;
    }
};
std::vector<char> bytes(const fs::path& path)
{
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "missing projection output");
    return { std::istreambuf_iterator<char>(f), {} };
}
std::vector<std::uint8_t> phantom()
{
    std::vector<std::uint8_t> labels(32 * 28 * 24);
    for (int z = 0; z < 24; ++z)
        for (int y = 0; y < 28; ++y)
            for (int x = 0; x < 32; ++x) {
                const double a = (x - 15.5) / 12, b = (y - 13.5) / 10, c = (z - 11.5) / 8;
                auto& label = labels[(z * 28 + y) * 32 + x];
                if (a * a + b * b + c * c < 1)
                    label = 1;
                if ((x - 19) * (x - 19) + (y - 11) * (y - 11) + (z - 12) * (z - 12) < 9)
                    label = 2;
            }
    return labels;
}
void verifySnapshot(const fs::path& dir, const std::vector<std::uint8_t>& labels)
{
    vtkNew<vtkXMLImageDataReader> reader;
    reader->SetFileName((dir / "labels.vti").string().c_str());
    reader->Update();
    auto* volume = reader->GetOutput();
    const auto* dims = volume->GetDimensions();
    require(dims[0] == 32 && dims[1] == 28 && dims[2] == 24, "volume was downsampled");
    require(volume->GetScalarType() == VTK_UNSIGNED_CHAR && std::memcmp(volume->GetScalarPointer(), labels.data(), labels.size()) == 0, "material labels changed");
    reader->SetFileName((dir / "texture.vti").string().c_str());
    reader->Update();
    dims = reader->GetOutput()->GetDimensions();
    require(dims[0] == 32 && dims[1] == 22, "display projection was not downsampled");
    vtkNew<vtkXMLPolyDataReader> mesh;
    mesh->SetFileName((dir / "detector.vtp").string().c_str());
    mesh->Update();
    auto* uv = mesh->GetOutput()->GetPointData()->GetTCoords();
    require(uv, "texture coordinates missing");
    const int columns = int(mesh->GetOutput()->GetNumberOfPoints() / 2);
    near(uv->GetComponent(0, 0), 0, 1e-8, "U-min texture edge");
    near(uv->GetComponent(columns - 1, 0), 1, 1e-8, "U-max texture edge");
    near(uv->GetComponent(0, 1), 0, 1e-8, "V-min texture edge");
    near(uv->GetComponent(columns, 1), 1, 1e-8, "V-max texture edge");
    require(fs::file_size(dir / "live.png") > 10000, "empty live render");
    mesh->SetFileName((dir / "source-trajectory.vtp").string().c_str());
    mesh->Update();
    require(mesh->GetOutput()->GetNumberOfPoints() == 8, "source trajectory incomplete");
    mesh->SetFileName((dir / "detector-trajectory.vtp").string().c_str());
    mesh->Update();
    require(mesh->GetOutput()->GetNumberOfPoints() == 8, "detector trajectory incomplete");
    std::ifstream meta(dir / "scene.txt");
    std::string field;
    int view;
    double source[3], center[3], ignored;
    meta >> field >> view >> field >> source[0] >> source[1] >> source[2] >> field;
    for (int i = 0; i < 16; ++i)
        meta >> ignored;
    meta >> field >> center[0] >> center[1] >> center[2];
    require(bool(meta), "missing source/beam pose");
    vtkNew<vtkXMLPolyDataReader> panel;
    panel->SetFileName((dir / "detector.vtp").string().c_str());
    panel->Update();
    mesh->SetFileName((dir / "beam.vtp").string().c_str());
    mesh->Update();
    require(mesh->GetOutput()->GetNumberOfPoints() == panel->GetOutput()->GetNumberOfPoints() + 1, "beam does not cover panel boundary");
    double apex[3], edge[3], corner[3];
    mesh->GetOutput()->GetPoint(0, apex);
    mesh->GetOutput()->GetPoint(1, edge);
    panel->GetOutput()->GetPoint(0, corner);
    for (int i = 0; i < 3; ++i) {
        near(apex[i], source[i], 1e-8, "beam apex moved from focal point");
        near(edge[i], corner[i], 1e-8, "beam missed detector edge");
    }
    double ray[3];
    double length = 0;
    for (int i = 0; i < 3; ++i) {
        ray[i] = center[i] - source[i];
        length += ray[i] * ray[i];
    }
    length = std::sqrt(length);
    for (double& x : ray)
        x /= length;
    mesh->SetFileName((dir / "source-body.vtp").string().c_str());
    mesh->Update();
    double far = 0, front = -1e10;
    for (vtkIdType i = 0; i < mesh->GetOutput()->GetNumberOfPoints(); ++i) {
        double point[3], along = 0;
        mesh->GetOutput()->GetPoint(i, point);
        for (int j = 0; j < 3; ++j)
            along += (point[j] - source[j]) * ray[j];
        far = std::min(far, along);
        front = std::max(front, along);
    }
    near(front, 0, 3e-5, "cylinder front face moved from focal point");
    near(far, -4, 3e-5, "cylinder orientation/length incorrect");
}
void verifyGeometry(const SimulationConfig& c, const std::vector<std::uint8_t>& labels, const SpectralTransmissionModel& model)
{
    using namespace primary_detail;
    CudaProjectionBackend backend;
    backend.prepare(c, labels, model);
    for (int view : { 0, 3, 7 }) {
        const auto frame = backend.geometryForDisplay(view);
        require(frame.view == view && frame.detector.size() >= 4, "display frame contract");
        auto g = makeDeviceGeometry(c, 2, 2);
        // Independently validate the published object pose against inverse ray coordinates.
        const double local[] = { 2, -3, 4 };
        float3 world {};
        float* components = &world.x;
        for (int r = 0; r < 3; ++r) {
            components[r] = float(frame.object_pose[r * 4 + 3]);
            for (int k = 0; k < 3; ++k)
                components[r] += float(frame.object_pose[r * 4 + k] * local[k]);
        }
        const auto inverse = phantomInverse(g, world, false);
        near(inverse.x, local[0], 2e-5, "object X pose order");
        near(inverse.y, local[1], 2e-5, "object Y pose order");
        near(inverse.z, local[2], 2e-5, "object Z pose order");
        if (c.geometry.kind != GeometryKind::PlanarSourceOrbit && c.geometry.kind != GeometryKind::NikonTilted) {
            const double a = g.start + g.rotation_direction * 2 * 3.14159265358979323846 * view / g.views_per_turn;
            near(frame.source[0], (g.sid + g.source_x) * std::sin(a) + g.source_y * std::cos(a), 5e-5, "actual source X offset");
            near(frame.source[1], -(g.sid + g.source_x) * std::cos(a) + g.source_y * std::sin(a), 5e-5, "actual source Y offset");
            near(frame.source[2], g.start_z + g.pitch * (a - g.start) / (2 * 3.14159265358979323846) + g.source_z, 2e-5, "actual source Z offset");
        }
    }
}
void writeDemoInput(const fs::path& dir, const std::vector<std::uint8_t>& labels)
{
    std::ofstream raw(dir / "phantom.raw", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(labels.data()), labels.size());
    std::ofstream spectrum(dir / "spectrum.spec");
    spectrum << "Energy[keV] N[photons]\n40 100\n80 50\n";
    for (bool enabled : { false, true }) {
        std::ofstream config(dir / (enabled ? "live.toml" : "headless.toml"));
        config << R"(schema_version=2
[workflow]
mode="project"
[visualization]
)" << "enabled="
               << (enabled ? "true" : "false") << R"(
texture_max_size=32
max_fps=30
projection="signal"
hold_open=false
offscreen=true
snapshot_directory="app-display"
[projection]
label_volume="phantom.raw"
spectrum_file="spectrum.spec"
)" << "output_file=\""
               << (enabled ? "app-live.raw" : "app-headless.raw") << "\"\n"
               << R"([projection.geometry_flux]
enabled=true
[projection.quantum_noise]
enabled=false
[projection.sampling]
samples_per_pixel=2
spatial_seed=12345
[[projection.materials]]
label=1
name="water"
formula="H2O"
density_g_cm3=1
[[projection.materials]]
label=2
name="aluminum"
formula="Al"
density_g_cm3=2.7
[geometry]
kind="cyl_helical"
views=8
views_per_turn=8
detector_u=96
detector_v=64
pixel_u_mm=1
pixel_v_mm=1
sid_mm=100
sdd_mm=200
pitch_mm_per_turn=6
start_z_mm=-3
source_offset_x_mm=2
source_offset_y_mm=-1
source_offset_z_mm=3
offset_u_mm=4
offset_n_mm=2
offset_v_mm=-3
tilt_u_rad=0.08
tilt_v_rad=-0.05
tilt_n_rad=0.04
volume_x=32
volume_y=28
volume_z=24
voxel_x_mm=0.8
voxel_y_mm=1.1
voxel_z_mm=1.3
phantom_offset_x_mm=3
phantom_offset_y_mm=-2
phantom_offset_z_mm=1
phantom_rotation_x_rad=0.15
phantom_rotation_y_rad=-0.1
phantom_rotation_z_rad=0.2
)";
    }
}
}
int main(int argc, char** argv)
{
    try {
        require(argc >= 2, "output directory required");
        const fs::path output(argv[1]);
        fs::create_directories(output);
        const bool interactive = argc >= 3 && std::string(argv[2]) == "--interactive";
        SimulationConfig c;
        auto& g = c.geometry.parameters;
        g.volume_x = 32;
        g.volume_y = 28;
        g.volume_z = 24;
        g.voxel_x_mm = .8;
        g.voxel_y_mm = 1.1;
        g.voxel_z_mm = 1.3;
        g.views = 8;
        g.views_per_turn = 8;
        g.detector_u = 96;
        g.detector_v = 64;
        g.pixel_u_mm = g.pixel_v_mm = 1;
        g.sid_mm = 100;
        g.sdd_mm = 200;
        g.start_z_mm = -3;
        g.pitch_mm_per_turn = 6;
        g.source_offset_x_mm = 2;
        g.source_offset_y_mm = -1;
        g.source_offset_z_mm = 3;
        g.offset_u_mm = 4;
        g.offset_n_mm = 2;
        g.offset_v_mm = -3;
        g.tilt_u_rad = .08;
        g.tilt_v_rad = -.05;
        g.tilt_n_rad = .04;
        g.phantom_offset_x_mm = 3;
        g.phantom_offset_y_mm = -2;
        g.phantom_offset_z_mm = 1;
        g.phantom_rotation_x_rad = .15;
        g.phantom_rotation_y_rad = -.1;
        g.phantom_rotation_z_rad = .2;
        g.planar_detector_y_mm = 100;
        g.source_orbit_radius_mm = 8;
        g.nikon_tilt_rad = .3;
        c.projection.quantum_noise.enabled = false;
        c.projection.sampling.samples_per_pixel = 2;
        MaterialSpec water;
        water.label = 1;
        water.name = "water";
        water.density_g_cm3 = 1;
        MaterialSpec inclusion;
        inclusion.label = 2;
        inclusion.name = "inclusion";
        inclusion.density_g_cm3 = 2;
        c.projection.materials = { water, inclusion };
        Attenuation attenuation;
        SpectralTransmissionModel model(c.projection.materials, { { 40, 100, 1 }, { 80, 50, 1 } }, attenuation, {});
        const auto labels = phantom();
        writeDemoInput(output, labels);
        const GeometryKind kinds[] = { GeometryKind::FlatCbct, GeometryKind::FlatHelical, GeometryKind::CylCbct, GeometryKind::CylHelical, GeometryKind::PlanarSourceOrbit, GeometryKind::NikonTilted };
        for (int kind = 0; kind < 6; ++kind) {
            c.geometry.kind = kinds[kind];
            verifyGeometry(c, labels, model);
            const auto dir = output / std::to_string(kind);
            fs::create_directories(dir);
            std::string error;
            c.visualization.enabled = false;
            SpectralProjectionPipeline pipeline(primary_detail::makeCudaProjectionSourceRegistry().create("projection.source.multi_spec_fp"));
            require(pipeline.run(c, labels, model, dir / "disabled.raw", error), error.c_str());
            c.visualization.enabled = true;
            c.visualization.texture_max_size = 32;
            c.visualization.max_fps = 120;
            c.visualization.offscreen = !interactive;
            c.visualization.hold_open = interactive && kind == 3;
            c.visualization.snapshot_directory = dir / "display";
            c.visualization.projection = kind == 1 ? "primary" : kind == 2 ? "log"
                                                                           : "signal";
            require(pipeline.run(c, labels, model, dir / "enabled.raw", error), error.c_str());
            require(bytes(dir / "disabled.raw") == bytes(dir / "enabled.raw"), "display modified full-resolution computed projection");
            verifySnapshot(c.visualization.snapshot_directory, labels);
            std::cout << "live geometry " << kind << ": PASS\n";
            if (interactive && kind == 3)
                break;
        }
        // Asymmetric synthetic ramp proves averaging/orientation independently of ray output.
        yk::visualization::LiveView view({ 2, 1, true, false, (output / "ramp").string() });
        yk::visualization::LabelVolume volume;
        volume.dimensions = { 32, 28, 24 };
        volume.spacing = { .8, 1.1, 1.3 };
        volume.labels = labels;
        volume.materials = { { 1, "water", { .8, .9, 1, .15 } } };
        view.setVolume(volume);
        yk::visualization::FrameGeometry frame;
        frame.source = { 0, -100, 0 };
        frame.detector_center = { 0, 100, 0 };
        frame.detector_axes = { yk::visualization::Point { 1, 0, 0 }, yk::visualization::Point { 0, 0, 1 }, yk::visualization::Point { 0, -1, 0 } };
        frame.detector = { { -2, 100, -1 }, { 2, 100, -1 }, { -2, 100, 1 }, { 2, 100, 1 } };
        view.setParameters({ { "Object X (mm)", 0, -10, 10 } });
        view.updateProjections(frame, 4, 2, { 0, 1, 2, 3, 4, 5, 6, 7 }, { 10, 20, 30, 40, 50, 60, 70, 80 }, { .1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f }, "orientation ramp");
        frame.view = 1;
        frame.source[0] = 10;
        frame.detector_center[0] += 5;
        for (auto& point : frame.detector)
            point[0] += 5;
        frame.object_pose[3] = 2;
        view.updateProjections(frame, 4, 2, { 7, 6, 5, 4, 3, 2, 1, 0 }, { 80, 70, 60, 50, 40, 30, 20, 10 }, { .8f, .7f, .6f, .5f, .4f, .3f, .2f, .1f }, "second frame");
        require(view.cachedFrameCount() == 2, "throttled frame was lost from replay cache");
        require(view.selectFrame(1) && view.displayedFrame() == 1, "cannot select computed scan angle");
        view.finish();
        vtkNew<vtkXMLImageDataReader> replay;
        replay->SetFileName((output / "ramp/texture.vti").string().c_str());
        replay->Update();
        auto* selected = static_cast<unsigned char*>(replay->GetOutput()->GetScalarPointer());
        require(selected[0] == 163 && selected[3] == 91, "scan selection did not change texture");
        vtkNew<vtkXMLPolyDataReader> selected_detector;
        selected_detector->SetFileName((output / "ramp/detector.vtp").string().c_str());
        selected_detector->Update();
        double corner[3];
        selected_detector->GetOutput()->GetPoint(0, corner);
        near(corner[0], 3, 1e-8, "scan selection did not change detector pose");
        selected_detector->SetFileName((output / "ramp/beam.vtp").string().c_str());
        selected_detector->Update();
        selected_detector->GetOutput()->GetPoint(0, corner);
        near(corner[0], 10, 1e-8, "scan selection did not change beam origin");
        std::ifstream selected_meta(output / "ramp/scene.txt");
        std::string field;
        int selected_view;
        selected_meta >> field >> selected_view;
        require(field == "view" && selected_view == 1, "selected frame ID changed");
        selected_meta >> field;
        double sx, sy, sz;
        selected_meta >> sx >> sy >> sz;
        near(sx, 10, 1e-8, "selected source position changed");
        replay->SetFileName((output / "ramp/energy.vti").string().c_str());
        replay->Update();
        near(*static_cast<float*>(replay->GetOutput()->GetScalarPointer()), 55, 1e-6, "energy panel selected wrong frame");
        replay->SetFileName((output / "ramp/attenuation.vti").string().c_str());
        replay->Update();
        near(*static_cast<float*>(replay->GetOutput()->GetScalarPointer()), .55, 1e-6, "attenuation panel selected wrong frame");
        view.setEditingEnabled(false);
        require(!view.selectFrame(0), "angle browsing not blocked during calculation");
        view.requestRecompute();
        require(!view.takeRecomputeRequest(), "disabled apply button accepted input");
        view.setEditingEnabled(true);
        require(view.setParameter(0, 3), "parameter widget did not accept value");
        require(view.displayedFrame() == 1 && view.parameterValues()[0] == 3, "draft parameter changed computed frame");
        require(!view.setParameter(0, 100), "parameter widget accepted out of bounds value");
        view.requestRecompute();
        require(view.takeRecomputeRequest() && !view.takeRecomputeRequest(), "apply button callback is not one-shot");
        require(view.selectFrame(0) && view.displayedFrame() == 0, "cannot revisit first computed angle");
        require(!view.selectFrame(99) && view.displayedFrame() == 0, "uncomputed frame selection accepted");
        view.finish();
        vtkNew<vtkXMLImageDataReader> ramp;
        ramp->SetFileName((output / "ramp/texture.vti").string().c_str());
        ramp->Update();
        auto* rgb = static_cast<unsigned char*>(ramp->GetOutput()->GetScalarPointer());
        require(rgb[0] == 91 && rgb[3] == 163, "display averaging/orientation incorrect");
        // Reuse the same UI across two real calculations. Cache must be invalidated.
        yk::visualization::LiveView reuse({ 32, 120, true, false, (output / "recompute").string() });
        reuse.setParameters({ { "Source X (mm)", 0, -100, 100 } });
        c.geometry.kind = GeometryKind::FlatCbct;
        c.visualization.enabled = true;
        SpectralProjectionPipeline repeated(primary_detail::makeCudaProjectionSourceRegistry().create("projection.source.multi_spec_fp"));
        repeated.setVisualization(&reuse);
        std::string recompute_error;
        reuse.setEditingEnabled(false);
        require(repeated.run(c, labels, model, output / "recompute-before.raw", recompute_error), recompute_error.c_str());
        require(reuse.cachedFrameCount() == 8 && !reuse.selectFrame(0), "scan was not locked while calculating");
        reuse.setEditingEnabled(true);
        require(reuse.selectFrame(0), "scan was not unlocked after calculating");
        reuse.finish();
        vtkNew<vtkXMLImageDataReader> before;
        before->SetFileName((output / "recompute/energy.vti").string().c_str());
        before->Update();
        const float before_energy = *static_cast<float*>(before->GetOutput()->GetScalarPointer());
        const auto before_output = bytes(output / "recompute-before.raw");
        c.geometry.parameters.source_offset_x_mm += 5;
        reuse.setEditingEnabled(false);
        require(repeated.run(c, labels, model, output / "recompute-after.raw", recompute_error), recompute_error.c_str());
        require(reuse.cachedFrameCount() == 8, "old run frames retained in cache");
        reuse.setEditingEnabled(true);
        require(reuse.selectFrame(0), "new scan selection failed");
        reuse.finish();
        before->Modified();
        before->Update();
        const float after_energy = *static_cast<float*>(before->GetOutput()->GetScalarPointer());
        require(std::abs(before_energy - after_energy) > 1e-4, "energy image was stale after changing source offset");
        require(before_output != bytes(output / "recompute-after.raw"), "source offset did not reach real forward projector");
        c.geometry.parameters.detector_u = 64;
        c.geometry.parameters.detector_v = 48;
        c.geometry.parameters.pixel_u_mm = 1.25;
        c.geometry.parameters.pixel_v_mm = .8;
        reuse.setEditingEnabled(false);
        require(repeated.run(c, labels, model, output / "resized-detector.raw", recompute_error), recompute_error.c_str());
        reuse.setEditingEnabled(true);
        require(reuse.selectFrame(0), "resized detector frame selection failed");
        reuse.finish();
        before->SetFileName((output / "recompute/energy.vti").string().c_str());
        before->Update();
        before->Modified();before->Update();
    const auto* resized_dims = before->GetOutput()->GetDimensions();
        require(resized_dims[0] == 32 && resized_dims[1] == 24, "detector resize did not change projection dimensions");
        require(bytes(output / "resized-detector.raw").size() == 64 * 48 * 8 * sizeof(float), "detector resize did not reach output writer");
        selected_detector->SetFileName((output / "recompute/detector.vtp").string().c_str());
        selected_detector->Update();
        double edge0[3], edge1[3], span = 0;
        selected_detector->GetOutput()->GetPoint(0, edge0);
        selected_detector->GetOutput()->GetPoint(1, edge1);
        for (int i = 0; i < 3; ++i)
            span += (edge1[i] - edge0[i]) * (edge1[i] - edge0[i]);
        near(std::sqrt(span), 80, 1e-4, "pixel spacing did not change physical detector size");
        reuse.invalidateResults();
        require(reuse.cachedFrameCount() == 0 && reuse.displayedFrame() == -1, "cache invalidation failed");
        std::cout << "PASS: generated CUDA projections, six trajectories, full-resolution labels, slider cache/frame selection, poses/offsets, dual projection panels, locked input and recompute invalidation, texture orientation and output invariance\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
