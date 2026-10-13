#pragma once
#include "ImageProcessing.hpp"
#include "recon/postprocess/PostprocessPipeline.hpp"
#include "recon/preprocess/PreprocessPipeline.hpp"
namespace yk::spectral::image {
// Correction frames keep their detector geometry. Resampling uses image::Pipeline
// and its updated Grid, before a reconstruction geometry snapshot is constructed.
std::unique_ptr<preprocess::IStage> makeProjectionStage(Options,
    preprocess::Domain = preprocess::Domain::LogAttenuation, Location = Location::Device);
std::unique_ptr<postprocess::IStage> makeVolumeStage(Options,
    postprocess::Domain = postprocess::Domain::AttenuationMmInv, Location = Location::Device);
// Owned buffers update with the volume dimensions; statistics count PCIe crossings.
class VolumeTransfer final : public postprocess::IDataTransfer {
public:
    size_t uploads = 0, downloads = 0;
    void ensure(postprocess::Volume&, Location, const postprocess::Context&) override;
};
} // namespace
