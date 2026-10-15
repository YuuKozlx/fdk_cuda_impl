#pragma once
#include "recon/preprocess/correction/BoneThicknessProvider.hpp"
#include "recon/preprocess/calibration/TwoMaterialCalibration.hpp"
namespace yk::spectral::preprocess {
inline const pipeline::PluginInfo& twoMaterialBhcInfo()
{
    static const pipeline::PluginInfo info = [] {pipeline::PluginInfo p;
        p.id="preprocess.two_material_bhc";p.name="Two-material beam hardening correction";
        p.description="Bone-thickness indexed monotone linearization of scatter-corrected log projections before water BHC.";
        p.version="1.0";p.kind=pipeline::PluginKind::PreprocessStage;p.category="beam_hardening";
        p.tags={"cuda","bone","calibration","2dcalbh"};p.implementation_status=pipeline::ImplementationStatus::Experimental;
        p.documentation="include/recon/preprocess/correction/README.md";return p; }();
    return info;
}
std::unique_ptr<IStage> makeTwoMaterialBhcStage(TwoMaterialCalibration,
    std::shared_ptr<IBoneThicknessProvider>);
}
