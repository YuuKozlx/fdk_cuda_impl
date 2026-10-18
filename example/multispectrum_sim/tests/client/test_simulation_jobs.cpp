#include "client/SimulationJobs.hpp"
#include <limits>

namespace yk::spectral {
namespace {
    void check(bool ok, const char* message)
    {
        if (!ok)
            throw std::runtime_error(message);
    }
    template <class F>
    void rejects(F action)
    {
        bool rejected = false;
        try {
            action();
        } catch (const std::exception&) {
            rejected = true;
        }
        check(rejected, "invalid native configuration or job state accepted");
    }
}
void testSimulationJobs()
{
    std::promise<void> release;
    auto gate = release.get_future().share();
    int runs = 0;
    SimulationJobs jobs([&](const SimulationConfig&, std::string&, TaskProgressCallback report) {
        gate.wait();
        std::string phase = "owned phase";
        report({ phase.c_str(), 2, 3 });
        phase.assign("overwritten");
        if (++runs == 1)
            throw std::runtime_error("worker failure");
        return true;
    });
    SimulationConfig c;
    jobs.startReconstruction(c);
    // Release even if the assertion fails: destruction joins the worker.
    bool overlapRejected = false;
    try {
        jobs.startReconstruction(c);
    } catch (const std::logic_error&) {
        overlapRejected = true;
    }
    release.set_value();
    check(overlapRejected, "overlapping reconstruction accepted");
    auto failed = jobs.finishReconstruction();
    check(!failed.first && failed.second == "worker failure", "worker exception lost");
    check(jobs.progress().phase == "owned phase" && jobs.history().size() == 1, "progress does not own phase text");
    jobs.startReconstruction(c);
    check(jobs.finishReconstruction().first && jobs.history().size() == 1, "retry retained old progress");
    rejects([&] { jobs.finishReconstruction(); });
    c.projection.output_file = "generated.raw";
    c.reconstruction.input_projection_file = "external.raw";
    jobs.recordProjection(c);
    jobs.recordReconstruction(c);
    jobs.invalidateForScanChange(c);
    check(!jobs.projected() && !jobs.reconstructed() && c.reconstruction.input_projection_file == "external.raw", "external input invalidated");
    jobs.recordProjection(c);
    c.reconstruction.input_projection_file = c.projection.output_file;
    jobs.invalidateForScanChange(c);
    check(c.reconstruction.input_projection_file.empty(), "obsolete generated input retained");
}
void testNativeConfig(const SimulationConfig& original, const std::filesystem::path& dir)
{
    auto c = original;
    normalizeConfig(c);
    validateConfig(c);
    c.source_config_file = dir / "does-not-exist.toml";
    c.visualization.enabled = false;
    c.visualization.interactive = false;
    auto& post = c.projection.detector_postprocess;
    post.optical_crosstalk.enabled = true;
    post.optical_crosstalk.kernel_u = { .2, .6, .2 };
    post.electronic_crosstalk.kernel_v = { .1, .8, .1 };
    for (auto kind : { GeometryKind::CylCbct, GeometryKind::CylHelical }) {
        c.geometry.kind = kind;
        c.workflow.mode = WorkflowMode::Project;
        saveConfig(c, dir / "native.toml");
        auto loaded = loadConfig(dir / "native.toml", false);
        check(loaded.geometry.kind == kind && loaded.projection.detector_postprocess.optical_crosstalk.kernel_u == post.optical_crosstalk.kernel_u && loaded.projection.detector_postprocess.electronic_crosstalk.kernel_v == post.electronic_crosstalk.kernel_v, "native serializer lost geometry or detector kernels");
    }
    c.projection.output_file = "relative-projection.raw";
    saveConfig(c, dir / "relative.toml");
    check(loadConfig(dir / "relative.toml", false).projection.output_file == std::filesystem::absolute(c.projection.output_file), "save changed relative path meaning");
    auto bad = c;
    bad.projection.detector_postprocess.optical_crosstalk.kernel_u = { .5, .5 };
    rejects([&] { validateConfig(bad); });
    bad = c;
    bad.projection.detector_response.enabled = true;
    bad.projection.detector_response.protective_layer.enabled = true;
    bad.projection.detector_response.protective_layer.preset = "unknown";
    rejects([&] { validateConfig(bad); });
    bad = c;
    bad.projection.detector_postprocess.afterglow.enabled = true;
    bad.projection.detector_postprocess.afterglow.p = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { validateConfig(bad); });
    bad = c;
    bad.projection.detector_postprocess.das.enabled = true;
    bad.projection.detector_postprocess.das.adc_lsb_electrons = -1;
    rejects([&] { validateConfig(bad); });
}
}
