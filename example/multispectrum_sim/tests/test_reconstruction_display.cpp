#include "ReconstructionDisplay.hpp"
#include "image/GridMetadata.hpp"
#include <fstream>
#include <iostream>
#include <limits>
using namespace yk::spectral;
void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
int main(){try{
    GeometryConfig grid;grid.reconstruction_volume_x=3;grid.reconstruction_volume_y=4;grid.reconstruction_volume_z=5;
    grid.reconstruction_voxel_x_mm=.5;grid.reconstruction_voxel_y_mm=1;grid.reconstruction_voxel_z_mm=2;
    const auto file=std::filesystem::current_path()/"reconstruction-display-test.raw";
    std::vector<float> values;
    for(int z=0;z<5;++z)for(int y=0;y<4;++y)for(int x=0;x<3;++x)values.push_back(float(100*z+10*y+x));
    auto write=[&]{std::ofstream f(file,std::ios::binary);f.write(reinterpret_cast<char*>(values.data()),values.size()*sizeof(float));};write();
    const auto volume=loadReconstructionDisplay(file,grid);
    auto xy=yk::visualization::extractFloatSlice(volume,2,2);
    require(xy.width==3&&xy.height==4&&xy.values[8]==222&&xy.spacing[0]==.5&&xy.spacing[1]==1,"axial orientation/spacing mismatch");
    auto xz=yk::visualization::extractFloatSlice(volume,1,1);
    require(xz.width==3&&xz.height==5&&xz.values[11]==312&&xz.spacing[1]==2,"coronal orientation/spacing mismatch");
    auto yz=yk::visualization::extractFloatSlice(volume,0,2);
    require(yz.width==4&&yz.height==5&&yz.values[11]==232&&yz.spacing[0]==1&&yz.spacing[1]==2,"sagittal orientation/spacing mismatch");
    require(yk::visualization::automaticFloatWindow(volume)[1]>yk::visualization::automaticFloatWindow(volume)[0],"invalid automatic window");
    bool rejected=false;try{yk::visualization::extractFloatSlice(volume,2,5);}catch(const std::exception&){rejected=true;}require(rejected,"invalid slice accepted");
    values.pop_back();write();rejected=false;try{loadReconstructionDisplay(file,grid);}catch(const std::exception&){rejected=true;}require(rejected,"short volume accepted");
    values.push_back(std::numeric_limits<float>::quiet_NaN());write();rejected=false;try{loadReconstructionDisplay(file,grid);}catch(const std::exception&){rejected=true;}require(rejected,"nonfinite volume accepted");
    values.assign(12,.02f);write();
    image::Grid outputGrid;outputGrid.dimensions={2,3,2};outputGrid.spacing_mm={.75,1.5,3};
    image::writeGridMetadata(file,outputGrid);
    const auto resampled=loadReconstructionDisplay(file,grid);
    require(resampled.dimensions==outputGrid.dimensions&&resampled.spacing==outputGrid.spacing_mm,"display ignored resampled output grid");
    std::filesystem::remove(image::gridMetadataPath(file));
    std::filesystem::remove(file);std::cout<<"Reconstruction load, three orientations and spacing passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
