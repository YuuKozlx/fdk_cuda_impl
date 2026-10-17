#include "io/MatLabelVolume.hpp"
#include "examples/DataExamples.hpp"
#include <matio.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace yk::spectral;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F> void rejects(F action,const char* message){
    try{action();}catch(const std::exception&){return;}throw std::runtime_error(message);
}
void fixture(const std::filesystem::path& path,const std::vector<double>& values,bool compressed=true,bool duplicate=false,bool complex=false){
    auto* file=Mat_CreateVer(path.string().c_str(),nullptr,MAT_FT_MAT5);
    require(file!=nullptr,"Cannot create fixture");
    std::size_t dims[]={2,3,4};
    std::vector<double> imaginary(values.size(),0);
    mat_complex_split_t split{const_cast<double*>(values.data()),imaginary.data()};
    auto* var=Mat_VarCreate("volume",MAT_C_DOUBLE,MAT_T_DOUBLE,3,dims,
        complex?static_cast<void*>(&split):static_cast<void*>(const_cast<double*>(values.data())),complex?MAT_F_COMPLEX:0);
    require(var!=nullptr,"Cannot create variable");
    require(Mat_VarWrite(file,var,compressed?MAT_COMPRESSION_ZLIB:MAT_COMPRESSION_NONE)==0,"Cannot write fixture");
    if(duplicate){var->name[0]='x';require(Mat_VarWrite(file,var,MAT_COMPRESSION_NONE)==0,"Cannot write second array");}
    Mat_VarFree(var);require(Mat_Close(file)==0,"Cannot close fixture");
}
void compareRaw(const std::vector<std::uint8_t>& labels,const std::filesystem::path& path){
    require(std::filesystem::file_size(path)==labels.size(),"Reference size differs");
    std::ifstream in(path,std::ios::binary);std::vector<char> chunk(1024*1024);
    for(std::size_t offset=0;offset<labels.size();){
        const auto n=std::min(chunk.size(),labels.size()-offset);
        require(bool(in.read(chunk.data(),n)),"Cannot read reference");
        for(std::size_t i=0;i<n;++i)require(labels[offset+i]==static_cast<unsigned char>(chunk[i]),"MAT/RAW voxel or axis mismatch");
        offset+=n;
    }
}
}
int main(int argc,char** argv){try{
    if(argc==3 && std::string(argv[1])=="--real-data"){
        const auto root=std::filesystem::path(argv[2]);
        const auto registry=examples::clientExamples();
        require(registry.list().size()==19,"Example count differs");
        for(const auto& info:registry.list()){
            if(info.category!="ICRP145" && info.category!="BrainWeb")continue;
            const auto stage=registry.create(info.id);
            const auto result=stage->generate({root/"results"});
            const auto& g=result.config.geometry.parameters;
            require(result.labels.size()==std::size_t(g.volume_x)*g.volume_y*g.volume_z,"Grid size differs");
            const auto reference=info.category=="ICRP145"?root/"icrp145/prepared/inputs"/(info.id+".raw"):
                root/"brainweb/brainweb_normal_1mm_181x217x181_uint8.raw";
            compareRaw(result.labels,reference);
            require(result.config.projection.materials.size()==(info.category=="ICRP145"?53:10),"Material count differs");
            std::cout<<info.id<<": exact voxel match, "<<result.labels.size()<<" voxels\n";
        }
        return 0;
    }
    require(argc==2,"Expected a new fixture directory");const auto dir=std::filesystem::path(argv[1]);
    std::filesystem::create_directories(dir);
    const auto round=dir/"roundtrip.mat";
    // Tests own only the known file names in this dedicated test directory.
    std::filesystem::remove(round);
    const std::array<int,3> dims{2,3,4};std::vector<std::uint8_t> labels(24);
    for(int z=0;z<4;++z)for(int y=0;y<3;++y)for(int x=0;x<2;++x)labels[x+2*(y+3*z)]=std::uint8_t(x+10*y+50*z);
    writeMatLabels(round,dims,labels);require(readClientLabels(round,dims)==labels,"Compressed uint8 round trip differs");
    rejects([&]{writeMatLabels(round,dims,labels);},"Existing MAT overwritten");
    rejects([&]{readMatLabels(round,{3,2,4});},"Axis mismatch accepted");
    rejects([&]{readMatLabels(round,{0,3,4});},"Zero dimension accepted");
    std::vector<double> values(24);for(int i=0;i<24;++i)values[i]=i%2?10000:0;
    const OrganLabelMap map{{0,0},{10000,7}};
    for(bool compressed:{false,true}){
        const auto p=dir/(compressed?"double-compressed.mat":"double.mat");fixture(p,values,compressed);
        const auto converted=readMatLabels(p,dims,map);
        for(int i=0;i<24;++i)require(converted[i]==(i%2?7:0),"Organ conversion differs");
        rejects([&]{readMatLabels(p,dims);},"Unmapped organ IDs accepted");
        rejects([&]{readMatLabels(p,dims,{{0,0}});},"Missing organ ID accepted");
    }
    const auto invalid=dir/"invalid.mat";
    for(double bad:{-1.,.5,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),4294967296.}){
        values[3]=bad;fixture(invalid,values);rejects([&]{readMatLabels(invalid,dims,map);},"Invalid label accepted");
    }
    values[3]=0;
    fixture(invalid,values,true,true);rejects([&]{readMatLabels(invalid,dims,map);},"Ambiguous arrays accepted");
    fixture(invalid,values,true,false,true);rejects([&]{readMatLabels(invalid,dims,map);},"Complex labels accepted");
    {std::ofstream out(invalid);out<<"invalid MAT";}
    rejects([&]{readMatLabels(invalid,dims);},"Corrupt MAT accepted");
    {std::ofstream out(invalid);out<<"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 24\n";}
    rejects([&]{readClientLabels(invalid,dims);},"LFS pointer accepted");
    const auto raw=dir/"labels.raw";
    {std::ofstream out(raw,std::ios::binary);out.write(reinterpret_cast<const char*>(labels.data()),labels.size());}
    require(readClientLabels(raw,dims)==labels,"RAW compatibility differs");
    rejects([&]{readClientLabels(raw,{2,3,5});},"Wrong RAW size accepted");
    require(examples::clientExamples().list().size()==19,"Example registry differs");
    std::cout<<"MAT label import/export and validation passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
