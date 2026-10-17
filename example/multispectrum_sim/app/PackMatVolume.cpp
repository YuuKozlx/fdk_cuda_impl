#include "io/MatLabelVolume.hpp"
#include <iostream>
#include <string>
#include <stdexcept>
int main(int argc,char** argv){try{
    if(argc!=6){std::cerr<<"Usage: multispectrum_pack_mat <labels.raw> <output.mat> <X> <Y> <Z>\n";return 2;}
    const std::array<int,3> dims={std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5])};
    const auto labels=yk::spectral::readClientLabels(argv[1],dims);
    yk::spectral::writeMatLabels(argv[2],dims,labels);
    if(yk::spectral::readMatLabels(argv[2],dims)!=labels)throw std::runtime_error("MAT lossless verification failed");
    std::cout<<"MAT saved and verified: "<<argv[2]<<" voxels="<<labels.size()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
