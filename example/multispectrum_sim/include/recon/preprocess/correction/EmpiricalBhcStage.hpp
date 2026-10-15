#pragma once
#include "recon/preprocess/correction/BoneThicknessProvider.hpp"
namespace yk::spectral::preprocess {
// Per-scan coefficients: p' = p + square*(L/scale)^2 + cross*p*(L/scale).
// L is deterministic segmented-bone path length in mm, NOT a spectral projection.
struct EmpiricalBhcParameters {
    double square = 0, cross = 0, path_scale_mm = 100;
    void validate() const;
};
inline const pipeline::PluginInfo& empiricalBhcInfo()
{
    static const pipeline::PluginInfo info = [] {
        pipeline::PluginInfo p;
        p.id = "preprocess.empirical_bhc";
        p.name = "Empirical bone beam hardening correction";
        p.description = "Apply per-scan fitted q^2 and p*q correction to baseline log projections; q is scaled bone path length.";
        p.version = "1.0";
        p.kind = pipeline::PluginKind::PreprocessStage;
        p.category = "beam_hardening";
        p.tags = { "cuda", "bone", "ebhc" };
        p.implementation_status = pipeline::ImplementationStatus::Experimental;
        p.documentation = "include/recon/preprocess/correction/EBHC.md";
        return p;
    }();
    return info;
}
// Append after the same base corrections used during fitting. Do not apply twice.
std::unique_ptr<IStage> makeEmpiricalBhcStage(EmpiricalBhcParameters,
    std::shared_ptr<IBoneThicknessProvider>);
}
