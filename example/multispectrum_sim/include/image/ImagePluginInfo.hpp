#pragma once
#include "pipeline/PluginInfo.hpp"
namespace yk::spectral::image {
inline const std::vector<pipeline::PluginInfo>& pluginDescriptions()
{
    static const auto result = [] {
        using namespace pipeline;
        std::vector<PluginInfo> all;
        for (const bool volume : { false, true })
            for (const bool device : { false, true })
                for (const std::string op : { "resample", "mean", "gaussian", "bilateral", "guided" }) {
                    PluginInfo p;
                    p.id = std::string(volume ? "postprocess" : "preprocess") + ".image." + (device ? "cuda." : "cpu.") + op;
                    p.name = std::string(device ? "CUDA " : "CPU ") + op;
                    p.description = "Scalar float image processing; projection batches are independent; volume neighborhoods support 2D or 3D.";
                    p.version = "1.0";
                    p.kind = volume ? PluginKind::PostprocessStage : PluginKind::PreprocessStage;
                    p.category = "image_processing";
                    p.tags = { op, device ? "cuda" : "cpu", volume ? "volume" : "projection" };
                    p.documentation = "example/multispectrum_sim/include/image/README.md";
                    if (op == "resample")
                        p.parameters = { { "output_dimensions", "Output dimensions", "Spatial dimensions; batch count is unchanged.", ParameterType::RealArray, "voxel" }, { "interpolation", "Interpolation", "Nearest or linear.", ParameterType::Choice, "" }, { "antialias", "Anti-alias", "Area averaging on shrinking axes with linear interpolation.", ParameterType::Boolean, "" } };
                    else {
                        p.parameters = { { "radius", "Radius", "Finite neighborhood radius on each axis.", ParameterType::RealArray, "voxel" }, { "neighborhood", "Neighborhood", "Independent 2D slices or spatial 3D.", ParameterType::Choice, "" } };
                        if (op == "gaussian" || op == "bilateral")
                            p.parameters.push_back({ "sigma_mm", "Spatial sigma", "Gaussian standard deviation in physical space.", ParameterType::RealArray, "mm" });
                        if (op == "bilateral")
                            p.parameters.push_back({ "range_sigma", "Range sigma", "Intensity standard deviation.", ParameterType::Real, "input unit" });
                        if (op == "guided")
                            p.parameters.push_back({ "epsilon", "Epsilon", "Self-guided regularization; squared input units.", ParameterType::Real, "input unit squared" });
                    }
                    all.push_back(std::move(p));
                }
        return all;
    }();
    return result;
}
} // namespace yk::spectral::image
