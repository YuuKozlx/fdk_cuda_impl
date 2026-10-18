#include "input/io/MatLabelVolume.hpp"
#include "input/examples/DataExamples.hpp"
#include <iostream>
#include <string>
#include <stdexcept>
int main(int argc,char** argv){try{
    if(argc>=4&&std::string(argv[1])=="--example"){
        const auto output=std::filesystem::absolute(std::filesystem::u8path(argv[3]));
        if(std::filesystem::exists(output))throw std::runtime_error("Output already exists; choose a new file");
        yk::spectral::examples::ExampleRequest request{output.parent_path()};
        int centerSlices=0;
        for(int i=4;i<argc;){
            const std::string option=argv[i++];
            if(option=="--center-slices"&&i<argc){
                if(centerSlices)throw std::runtime_error("Duplicate --center-slices");
                centerSlices=std::stoi(argv[i++]);
                if(centerSlices<1)throw std::runtime_error("--center-slices must be positive");
            }else if(option=="--set"&&i+1<argc){
                const std::string key=argv[i++];const std::string value=argv[i++];
                size_t consumed=0;const double number=std::stod(value,&consumed);
                if(consumed!=value.size()||!request.parameters.emplace(key,number).second)throw std::runtime_error("Invalid or duplicate example parameter");
            }else throw std::runtime_error("Expected --center-slices N or --set parameter value");
        }
        auto stage=yk::spectral::examples::clientExamples().create(argv[2]);
        if(!request.parameters.empty()&&stage->parameters().empty())throw std::runtime_error("This data-backed example has no phantom parameters");
        auto sample=stage->generate(request);
        const auto& g=sample.config.geometry.parameters;
        std::array<int,3> dims{g.volume_x,g.volume_y,g.volume_z};
        if(centerSlices){
            const int count=centerSlices;
            if(count<1||count>dims[2]||(dims[2]-count)%2)throw std::runtime_error("Centered crop must fit and preserve voxel centers (same Z parity)");
            const size_t plane=size_t(dims[0])*dims[1],first=size_t((dims[2]-count)/2)*plane;
            std::vector<std::uint8_t> crop(sample.labels.begin()+first,sample.labels.begin()+first+count*plane);
            sample.labels=std::move(crop);dims[2]=count;
        }
        if(!output.parent_path().empty())std::filesystem::create_directories(output.parent_path());
        yk::spectral::writeMatLabels(output,dims,sample.labels);
        if(yk::spectral::readMatLabels(output,dims)!=sample.labels)throw std::runtime_error("Example MAT verification failed");
        std::cout<<"Example MAT saved and verified: "<<output.string()<<" grid="<<dims[0]<<'x'<<dims[1]<<'x'<<dims[2]<<'\n';
        return 0;
    }
    if(argc!=6){std::cerr<<"Usage: multispectrum_pack_mat <labels.raw> <output.mat> <X> <Y> <Z>\n       multispectrum_pack_mat --example <id> <output.mat> [--center-slices N] [--set parameter value ...]\n";return 2;}
    const std::array<int,3> dims={std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5])};
    const auto labels=yk::spectral::readClientLabels(argv[1],dims);
    yk::spectral::writeMatLabels(argv[2],dims,labels);
    if(yk::spectral::readMatLabels(argv[2],dims)!=labels)throw std::runtime_error("MAT lossless verification failed");
    std::cout<<"MAT saved and verified: "<<argv[2]<<" voxels="<<labels.size()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
