#pragma once
#include "cbct_calibration/pipeline.hpp"
#include "input/config/SimConfig.hpp"
#include "processing/stage/Stage.hpp"
#include <optional>
namespace yk::spectral::calibration {
enum class Domain { Measurements,
    Geometry,
    CorrectedGeometry };
enum class InputKind { ProjectionRaw,
    IndexedPoints };
struct Data {
    Domain domain = Domain::Measurements;
    pipeline::Location location = pipeline::Location::Host;
    std::vector<cbct::calibration::Point2> points;
    std::vector<cbct::calibration::Point3> landmarks;
    std::vector<unsigned char> usable;
    std::optional<cbct::calibration::CalibrationResult> result;
    std::optional<GeometrySpec> corrected;
};
struct Context {
    InputKind input = InputKind::ProjectionRaw;
    cbct::calibration::PipelineConfig config;
    bool full_circle = true;
    TaskProgressCallback progress;
    GeometrySpec nominal;
    double maximum_rmse_px = 1.0;
};
GeometrySpec nominalGeometry(GeometrySpec geometry);
// Persist only accepted correction geometry, separately from the diagnostic fit report.
void saveGeometryCorrection(const std::filesystem::path&, const GeometrySpec&, double rmse_px);
// Preserve the caller's reconstruction grid; reject mismatched acquisition sampling.
GeometrySpec loadGeometryCorrection(const std::filesystem::path&, const GeometrySpec& acquisition, double maximum_rmse_px = 1);
using IGeometryCalibrationStage = pipeline::IStage<Data, Context, Domain>;
class GeometryCorrectionStage final : public IGeometryCalibrationStage {
    pipeline::PluginInfo info_;

public:
    GeometryCorrectionStage();
    const pipeline::PluginInfo& info() const noexcept override { return info_; }
    pipeline::Contract<Domain> contract() const override { return { Domain::Geometry, Domain::CorrectedGeometry }; }
    void process(Data&, const Context&) override;
};
class GeometryCalibrationStage final : public IGeometryCalibrationStage {
    pipeline::PluginInfo info_;

public:
    GeometryCalibrationStage();
    const pipeline::PluginInfo& info() const noexcept override { return info_; }
    pipeline::Contract<Domain> contract() const override { return { Domain::Measurements, Domain::Geometry }; }
    void validateContext(const Context&) const override;
    void process(Data&, const Context&) override;
};
}
