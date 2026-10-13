#include "ReconstructionDisplay.hpp"
#include "image/GridMetadata.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>
namespace yk::spectral {
yk::visualization::FloatVolume loadReconstructionDisplay(const std::filesystem::path& path,const GeometryConfig& g){
    yk::visualization::FloatVolume volume;
    volume.dimensions={g.reconstruction_volume_x,g.reconstruction_volume_y,g.reconstruction_volume_z};
    volume.spacing={g.reconstruction_voxel_x_mm,g.reconstruction_voxel_y_mm,g.reconstruction_voxel_z_mm};
    image::Grid output_grid;
    if(image::readGridMetadata(path,output_grid)){volume.dimensions=output_grid.dimensions;volume.spacing=output_grid.spacing_mm;}
    std::size_t count=1;
    for(int dim:volume.dimensions){if(dim<=0 || count>std::numeric_limits<std::size_t>::max()/std::size_t(dim))throw std::runtime_error("invalid reconstruction dimensions");count*=dim;}
    if(count>std::numeric_limits<std::size_t>::max()/sizeof(float) || std::filesystem::file_size(path)!=count*sizeof(float))throw std::runtime_error("reconstruction file size does not match output grid");
    volume.values.resize(count);std::ifstream file(path,std::ios::binary);
    if(!file || !file.read(reinterpret_cast<char*>(volume.values.data()),static_cast<std::streamsize>(count*sizeof(float))))throw std::runtime_error("cannot read reconstruction volume: "+path.string());
    yk::visualization::validateFloatVolume(volume);return volume;
}
}
