#pragma once
#include <array>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <algorithm>
namespace yk::spectral {
struct FlatFrame { std::array<double,3> source,center,u,v; };
inline double frameNorm(const std::array<double,3>& a){return std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);}
inline std::vector<FlatFrame> readFlatFrames(const std::filesystem::path& path){
    std::ifstream input(path);if(!input)throw std::runtime_error("Cannot open flat geometry: "+path.string());
    std::vector<FlatFrame> frames;std::string line;int row=0;
    while(std::getline(input,line)){
        ++row;if(line.find_first_not_of(" \t\r")==std::string::npos || line[line.find_first_not_of(" \t")]=='#')continue;
        if(line.rfind("source_x,",0)==0 && frames.empty())continue;
        std::replace(line.begin(),line.end(),',',' ');std::istringstream values(line);FlatFrame f;
        for(auto* a:{&f.source,&f.center,&f.u,&f.v})for(auto& x:*a)if(!(values>>x)||!std::isfinite(x)||std::abs(x)>1e8)throw std::runtime_error("Invalid flat geometry row "+std::to_string(row));
        std::string extra;if(values>>extra)throw std::runtime_error("Flat geometry requires exactly 12 columns");
        const double du=frameNorm(f.u),dv=frameNorm(f.v);
        if(du<1e-8||dv<1e-8)throw std::runtime_error("Zero detector pixel vector");
        double dot=0;for(int k=0;k<3;++k)dot+=f.u[k]*f.v[k];
        if(std::abs(dot)>1e-5*du*dv)throw std::runtime_error("Detector U/V must be perpendicular");
        std::array<double,3> normal{f.u[1]*f.v[2]-f.u[2]*f.v[1],f.u[2]*f.v[0]-f.u[0]*f.v[2],f.u[0]*f.v[1]-f.u[1]*f.v[0]};
        double distance=0;for(int k=0;k<3;++k)distance+=(f.source[k]-f.center[k])*normal[k];
        if(std::abs(distance)/(du*dv)<1e-5)throw std::runtime_error("Source lies in detector plane");
        if(!frames.empty() && (std::abs(du-frameNorm(frames[0].u))>1e-5*du || std::abs(dv-frameNorm(frames[0].v))>1e-5*dv))throw std::runtime_error("Changing pixel spacing across frames is unsupported");
        frames.push_back(f);
    }
    if(frames.empty())throw std::runtime_error("Empty flat geometry table");
    return frames;
}
inline void writeFlatFrames(const std::filesystem::path& path,const std::vector<FlatFrame>& frames){
    std::ofstream out(path);if(!out)throw std::runtime_error("Cannot write flat geometry");
    out<<"source_x,source_y,source_z,center_x,center_y,center_z,u_x,u_y,u_z,v_x,v_y,v_z\n"<<std::setprecision(17);
    for(const auto& f:frames){bool first=true;for(const auto* a:{&f.source,&f.center,&f.u,&f.v})for(double x:*a){if(!first)out<<',';out<<x;first=false;}out<<'\n';}
    if(!out)throw std::runtime_error("Failed writing flat geometry");
}
}
