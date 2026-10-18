#pragma once
#include "input/config/SimConfig.hpp"
#include "processing/support/SimLogger.hpp"
#include "recon/backend/LibraryReconstructor.hpp"
#include <chrono>
#include <future>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace yk::spectral {
// Owns computation snapshots and the reconstruction worker. Rendering remains on
// the UI thread. A projection snapshot is published only after a successful run.
class SimulationJobs {
public:
    struct Progress {
        std::string phase = "Preparing";
        int completed = 0, total = 0;
    };
    using Result = std::pair<bool, std::string>;
    using Reconstructor = std::function<bool(const SimulationConfig&, std::string&, TaskProgressCallback)>;

private:
    std::optional<SimulationConfig> projection_, reconstruction_;
    mutable std::mutex mutex_;
    Progress progress_;
    std::vector<Progress> history_;
    std::future<Result> worker_;
    Reconstructor reconstruct_;

public:
    explicit SimulationJobs(Reconstructor reconstruct = [](const SimulationConfig& c, std::string& error, TaskProgressCallback progress) {
        return reconstructWithLibrary(c, c.reconstruction.input_projection_file, c.reconstruction.output_volume_file,
            c.reconstruction.slice_prefix, error, std::move(progress));
    })
        : reconstruct_(std::move(reconstruct))
    {
    }
    ~SimulationJobs()
    {
        if (worker_.valid())
            worker_.wait();
    }
    SimulationJobs(const SimulationJobs&) = delete;
    SimulationJobs& operator=(const SimulationJobs&) = delete;
    const std::optional<SimulationConfig>& projected() const { return projection_; }
    const std::optional<SimulationConfig>& reconstructed() const { return reconstruction_; }
    void recordProjection(const SimulationConfig& c)
    {
        projection_ = c;
        reconstruction_.reset();
    }
    void recordReconstruction(const SimulationConfig& c) { reconstruction_ = c; }
    void invalidateResults()
    {
        projection_.reset();
        reconstruction_.reset();
    }
    void invalidateForScanChange(SimulationConfig& draft)
    {
        if (projection_ && draft.reconstruction.input_projection_file.lexically_normal() == projection_->projection.output_file.lexically_normal())
            draft.reconstruction.input_projection_file.clear();
        invalidateResults();
    }
    static SimulationConfig reconstructionConfig(const SimulationConfig& projected, const SimulationConfig& draft, const char* suffix = "")
    {
        auto c = projected;
        c.reconstruction = draft.reconstruction;
        if (*suffix) {
            c.reconstruction.geometry_override.enabled = false;
            c.reconstruction.input_projection_file = projected.projection.output_file;
            c.reconstruction.output_volume_file = projected.projection.output_file.string() + suffix + ".raw";
            c.reconstruction.slice_prefix = projected.projection.output_file.string() + suffix;
        }
        const auto& grid = draft.geometry.parameters;
        auto& g = c.geometry.parameters;
        g.reconstruction_volume_x = grid.reconstruction_volume_x;
        g.reconstruction_volume_y = grid.reconstruction_volume_y;
        g.reconstruction_volume_z = grid.reconstruction_volume_z;
        g.reconstruction_voxel_x_mm = grid.reconstruction_voxel_x_mm;
        g.reconstruction_voxel_y_mm = grid.reconstruction_voxel_y_mm;
        g.reconstruction_voxel_z_mm = grid.reconstruction_voxel_z_mm;
        g.reconstruction_offset_x_mm = grid.reconstruction_offset_x_mm;
        g.reconstruction_offset_y_mm = grid.reconstruction_offset_y_mm;
        g.reconstruction_offset_z_mm = grid.reconstruction_offset_z_mm;
        c.workflow.mode = WorkflowMode::Reconstruct;
        c.visualization.enabled = false;
        c.visualization.interactive = false;
        normalizeConfig(c);
        validateConfig(c);
        validateRuntimeInputs(c);
        if (c.reconstruction.output_volume_file.empty())
            throw std::runtime_error("Set a reconstruction output volume path");
        return c;
    }
    void startReconstruction(SimulationConfig config)
    {
        if (worker_.valid())
            throw std::logic_error("Collect the current reconstruction before starting another");
        for (const auto& p : { config.reconstruction.output_volume_file, config.reconstruction.slice_prefix })
            if (!p.parent_path().empty())
                std::filesystem::create_directories(p.parent_path());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            progress_ = {};
            history_.clear();
        }
        worker_ = std::async(std::launch::async, [this, config = std::move(config)] {
            std::string diagnostic, logged_phase;
            try {
                const auto progress = [&](TaskProgress value) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        progress_ = { value.phase, value.completed, value.total };
                        history_.push_back(progress_);
                    }
                    if (value.phase != logged_phase || logged_phase == "Iterative updates") {
                        std::ostringstream line;
                        line << "[reconstruction] " << value.phase;
                        if (value.total > 0)
                            line << " " << value.completed << "/" << value.total;
                        SimLogger::instance().info(line.str());
                        logged_phase = value.phase;
                    }
                };
                const bool ok = reconstruct_(config, diagnostic, progress);
                return Result { ok, std::move(diagnostic) };
            } catch (const std::exception& e) {
                return Result { false, e.what() };
            }
        });
    }
    bool ready() const { return worker_.valid() && worker_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready; }
    Progress progress() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return progress_;
    }
    std::vector<Progress> history() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return history_;
    }
    Result finishReconstruction()
    {
        if (!worker_.valid())
            throw std::logic_error("No reconstruction is running");
        return worker_.get();
    }
};
}
