#include "ReconstructionPreprocessor.hpp"
#include "recon/preprocess/PreprocessPipeline.hpp"
namespace yk::spectral {
bool prepareReconstructionProjection(const SimulationConfig& c,const std::filesystem::path& input,
    const std::filesystem::path& output,std::string& diagnostic,TaskProgressCallback progress){
    preprocess::Pipeline pipeline;
    return pipeline.run(c,input,output,diagnostic,std::move(progress));
}
}
