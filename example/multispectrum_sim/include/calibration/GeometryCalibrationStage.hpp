#pragma once
#include "cbct_calibration/pipeline.hpp"
#include "config/SimConfig.hpp"
#include "pipeline/Stage.hpp"
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
