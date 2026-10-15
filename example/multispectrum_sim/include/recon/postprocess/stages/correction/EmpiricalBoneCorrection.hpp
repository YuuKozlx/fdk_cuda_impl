#pragma once
#include "recon/postprocess/PostprocessPipeline.hpp"
namespace yk::spectral::postprocess {
struct EbhcCoefficients {
    double linear = 0, cross = 0, square = 0;
};
struct EbhcTvResult {
    EbhcCoefficients coefficients;
    double initial_tv = 0, final_tv = 0;
    int evaluations = 0;
    bool converged = false;
    std::vector<std::array<double, 4>> trace;
};
// Order: f01=soft-segmented attenuation image; f11=R(p0*p2); f02=R(p2*p2).
using EbhcBasis = std::array<std::shared_ptr<const std::vector<float>>, 3>;
// Full-volume isotropic TV, backward adjacent differences, no ROI/Huber/downsampling.
// Nelder-Mead is an explicit implementation choice for the paper's simplex solver.
EbhcTvResult fitEbhcFullVolumeTv(const std::vector<float>& baseline, const EbhcBasis&, const Grid&,
    std::function<void(int, double)> progress = {});
inline const pipeline::PluginInfo& ebhcVolumeInfo()
{
    static const pipeline::PluginInfo p = [] { pipeline::PluginInfo v;
        v.id="postprocess.ebhc_three_term";v.name="Three-term empirical bone correction";
        v.description="Add c01*f01+c11*f11+c02*f02 using per-scan full-volume TV coefficients.";
        v.version="1.0";v.kind=pipeline::PluginKind::PostprocessStage;v.category="beam_hardening";
        v.tags={"cuda","ebhc","three_term"};v.implementation_status=pipeline::ImplementationStatus::Experimental;
        v.documentation="include/recon/preprocess/correction/EBHC.md";return v; }();
    return p;
}
// Fixed-grid, per-scan basis. CUDA default/Context stream; owns pooled basis storage.
std::unique_ptr<IStage> makeEbhcVolumeStage(EbhcCoefficients, EbhcBasis, Grid);
}
