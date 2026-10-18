#pragma once
#include "client/visualization/LiveView.hpp"
#include "projection/spectral/SpectralProjectionPipeline.hpp"
#include <sstream>
#include <stdexcept>
namespace yk::spectral {
// Client-owned bridge. An external view is borrowed; otherwise this adapter owns it.
class LiveProjectionObserver final : public IProjectionObserver {
    yk::visualization::LiveView* display_;
    std::unique_ptr<yk::visualization::LiveView> owned_;
    const SimulationConfig* config_ = nullptr;

public:
    explicit LiveProjectionObserver(yk::visualization::LiveView* view = nullptr)
        : display_(view)
    {
    }
    void begin(const SimulationConfig& c, const std::vector<std::uint8_t>& labels) override
    {
        config_ = &c;
        if (!display_) {
            const auto& v = c.visualization;
            owned_ = std::make_unique<yk::visualization::LiveView>(yk::visualization::LiveViewOptions {
                v.texture_max_size, v.max_fps, v.offscreen, v.hold_open, v.snapshot_directory.string() });
            display_ = owned_.get();
        }
        auto* display = display_;
        const auto& g = c.geometry.parameters;
        yk::visualization::LabelVolume volume;
        volume.dimensions = { g.volume_x, g.volume_y, g.volume_z };
        volume.spacing = { g.voxel_x_mm, g.voxel_y_mm, g.voxel_z_mm };
        volume.labels = labels;
        constexpr double colors[][3] = { { .75, .85, 1 }, { 1, .65, .25 }, { .85, .35, .4 }, { .4, .85, .6 }, { .8, .6, .95 }, { .85, .85, .35 } };
        for (size_t i = 0; i < c.projection.materials.size(); ++i) {
            const auto& m = c.projection.materials[i];
            const auto& color = colors[i % 6];
            volume.materials.push_back({ m.label, m.name, { color[0], color[1], color[2], m.density_g_cm3 < .01 ? 0. : .15 } });
        }
        display->setVolume(volume);
    }
    void prepared(const std::vector<ProjectionViewGeometry>& frames) override { display_->setTrajectory(frames); }
    void frame(const ProjectionFrame& frame, const ProjectionViewGeometry& geometry,
        const std::vector<float>& logarithm) override
    {
        if (!display_->isOpen())
            return;
        if (frame.domain != ProjectionSignalDomain::EnergyKeV)
            throw std::runtime_error("live display currently requires energy-integrating frames");
        const auto& c = *config_;
        const auto& g = c.geometry.parameters;
        auto* display = display_;
        const std::vector<float>* values = &frame.signal_energy;
        if (c.visualization.projection == "primary")
            values = &frame.expected_primary_energy;
        if (c.visualization.projection == "log")
            values = &logarithm;
        std::ostringstream description;
        description << "Live " << c.visualization.projection << " projection | detector offsets U/N/V "
                    << g.offset_u_mm << " / " << g.offset_n_mm << " / " << g.offset_v_mm << " mm\nsource offsets "
                    << g.source_offset_x_mm << " / " << g.source_offset_y_mm << " / " << g.source_offset_z_mm
                    << " mm | detector tilt U/V/N " << g.tilt_u_rad << " / " << g.tilt_v_rad << " / " << g.tilt_n_rad << " rad";
        description << "\nobject offset " << g.phantom_offset_x_mm << " / " << g.phantom_offset_y_mm << " / " << g.phantom_offset_z_mm
                    << " mm | rotation X/Y/Z " << g.phantom_rotation_x_rad << " / " << g.phantom_rotation_y_rad << " / " << g.phantom_rotation_z_rad << " rad";
        description << "\nmaterials:";
        for (const auto& m : c.projection.materials)
            description << " " << int(m.label) << "=" << m.name;
        display->updateProjections(geometry, frame.width, frame.height, *values, frame.signal_energy, logarithm, description.str(), frame.view + 1 == g.views);
    }
    void progress(int completed, int total) override
    {
        if (completed == 1 || completed == total || completed % 10 == 0)
            display_->setStatus("Computing " + std::to_string(completed) + " / " + std::to_string(total) + " views. Editing and browsing unlock after completion.");
    }
    void reset() noexcept override
    {
        config_ = nullptr;
        if (owned_) {
            owned_.reset();
            display_ = nullptr;
        }
    }
    void finish() override
    {
        if (owned_)
            owned_->finish();
        config_ = nullptr;
    }
};
}
