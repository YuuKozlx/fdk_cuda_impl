#pragma once
#include "input/calibration/WaterBhcCalibration.hpp"
#include "input/calibration/WaterBhcTable.hpp"
#include "processing/stage/Stage.hpp"
#include <optional>
#include <stdexcept>
namespace yk::spectral::calibration {
enum class PrepareDomain { Inputs, Ready };
struct PrepareData { SimulationConfig config; };
struct PrepareContext {
    bool water_bhc = true;
    bool geometry = false;
    std::optional<GeometrySpec> fitted_geometry;
    TaskProgressCallback progress;
};
// Whole-task preparation; correction stages receive ready resources, never a fit recipe.
class PrepareCalibrateStage final : public pipeline::IStage<PrepareData,PrepareContext,PrepareDomain> {
    pipeline::PluginInfo info_{"calibration.prepare", "PrepareCalibrate", "Prepare in-memory calibration resources before projection corrections", "1", pipeline::PluginKind::CalibrationStage};
public:
    const pipeline::PluginInfo& info() const noexcept override { return info_; }
    pipeline::Contract<PrepareDomain> contract() const override { return {PrepareDomain::Inputs,PrepareDomain::Ready}; }
    void process(PrepareData& data,const PrepareContext& context) override {
        auto prepared=data.config;
        auto& bh=prepared.reconstruction.preprocessing.beam_hardening;
        if(context.water_bhc && prepared.reconstruction.preprocessing.enabled && bh.enabled) {
            if(context.progress)context.progress({"PrepareCalibrate: water BHC",0,1});
            if(!bh.calibration_file.empty()) bh=resolveWaterBhc(bh);
            else if(bh.fit_from_water_plate) bh=fitWaterPlate(prepared,bh,context.progress);
            if(context.progress)context.progress({"PrepareCalibrate: water BHC",1,1});
        }
        if(context.geometry) {
            if(!context.fitted_geometry)throw std::runtime_error("PrepareCalibrate: run geometry calibration or select a calibration file first");
            const auto& src=context.fitted_geometry->parameters;
            const auto& dst=prepared.geometry.parameters;
            if(prepared.geometry.kind!=GeometryKind::FlatCbct || src.views!=dst.views || src.detector_u!=dst.detector_u || src.detector_v!=dst.detector_v || src.pixel_u_mm!=dst.pixel_u_mm || src.pixel_v_mm!=dst.pixel_v_mm)
                throw std::runtime_error("Prepared geometry does not match current acquisition");
            auto geometry=*context.fitted_geometry;
            auto& g=geometry.parameters;
            g.reconstruction_volume_x=dst.reconstruction_volume_x;g.reconstruction_volume_y=dst.reconstruction_volume_y;g.reconstruction_volume_z=dst.reconstruction_volume_z;
            g.reconstruction_voxel_x_mm=dst.reconstruction_voxel_x_mm;g.reconstruction_voxel_y_mm=dst.reconstruction_voxel_y_mm;g.reconstruction_voxel_z_mm=dst.reconstruction_voxel_z_mm;
            g.reconstruction_offset_x_mm=dst.reconstruction_offset_x_mm;g.reconstruction_offset_y_mm=dst.reconstruction_offset_y_mm;g.reconstruction_offset_z_mm=dst.reconstruction_offset_z_mm;
            prepared.geometry=geometry;
            prepared.reconstruction.geometry_override.enabled=false;
        }
        data.config=std::move(prepared);
    }
};
}
