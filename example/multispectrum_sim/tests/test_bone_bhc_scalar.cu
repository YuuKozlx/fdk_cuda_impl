#include "recon/bhc/BoneBhcStage.hpp"
#include "WaterReference.hpp"
#include "primary/ProjectionGeometry.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace yk::spectral;
namespace pd=primary_detail;
namespace fs=std::filesystem;
static void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
static double boxLength(pd::ScannerRay ray,const double* lo,const double* hi){
    double start=0,end=1,length2=0;
    const double s[]={ray.source.x,ray.source.y,ray.source.z};
    const double d[]={double(ray.detector.x)-s[0],double(ray.detector.y)-s[1],double(ray.detector.z)-s[2]};
    for(int k=0;k<3;++k){length2+=d[k]*d[k];if(std::abs(d[k])<1e-12){if(s[k]<lo[k]||s[k]>hi[k])return 0;continue;}
        double a=(lo[k]-s[k])/d[k],b=(hi[k]-s[k])/d[k];if(a>b)std::swap(a,b);start=std::max(start,a);end=std::min(end,b);}
    return std::max(0.,end-start)*std::sqrt(length2);
}
template<class T>static void write(const fs::path& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));}
int main(int argc,char** argv){try{
    require(argc==2,"fixture directory required");
    const auto root=fs::path(argv[1])/std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());fs::create_directories(root);
    int index=0;
    for(auto kind:{GeometryKind::FlatCbct,GeometryKind::FlatHelical,GeometryKind::CylHelical})for(double scale:{1.,2.}){
        SimulationConfig c;c.geometry.kind=kind;auto& g=c.geometry.parameters;
        g.detector_u=12;g.detector_v=10;g.views=g.views_per_turn=7;g.sid_mm=60;g.sdd_mm=100;
        g.pixel_u_mm=.6;g.pixel_v_mm=.7;g.offset_u_mm=.2;g.offset_v_mm=-.1;
        g.start_z_mm=-.5;g.pitch_mm_per_turn=kind==GeometryKind::FlatCbct?0:1;
        g.reconstruction_volume_x=9;g.reconstruction_volume_y=11;g.reconstruction_volume_z=13;
        g.reconstruction_voxel_x_mm=.7;g.reconstruction_voxel_y_mm=.8;g.reconstruction_voxel_z_mm=1.1;
        g.reconstruction_offset_x_mm=1.2;g.reconstruction_offset_y_mm=-.8;g.reconstruction_offset_z_mm=.3;
        auto& bh=c.reconstruction.preprocessing.beam_hardening;bh.enabled=true;bh.reference_energy_keV=70;
        const auto water=resolveWaterReference(c).mu_water_mm_inv;
        std::vector<float> volume(9*11*13,float(water));
        for(int z=0;z<13;++z)for(int y=0;y<11;++y)for(int x=0;x<9;++x){
            if(x==0||y==0||z==0||x==8||y==10||z==12)volume[(z*11+y)*9+x]=0;
            if(x>=3&&x<=5&&y>=4&&y<=6&&z>=4&&z<=8)volume[(z*11+y)*9+x]=float(2*water);
        }
        preprocess::TwoMaterialCalibration table;table.bone_rows=table.projection_columns=2;table.bone_step_mm=1000;
        table.reference_energy_keV=70;table.target_water_mu_mm=water;table.target_bone_mu_mm=2*water;
        table.minimum_log={0,0};table.log_step={20,20};table.target_log={0,20,1,21};table.provenance="test only: p+0.001*bone_mm";
        const auto folder=root/std::to_string(index++);fs::create_directory(folder);
        const size_t n=size_t(g.views)*g.detector_u*g.detector_v;
        std::vector<float> measured(n);for(size_t i=0;i<n;++i)measured[i]=.2f+float(i%13)*.01f;
        write(folder/"volume.raw",volume);write(folder/"input.raw",measured);
        bone_bhc::TwoMaterialLutBhc algorithm(c,table,scale);algorithm.correct(folder/"volume.raw",folder/"input.raw",folder/"output.raw");
        std::vector<float> actual(n);std::ifstream f(folder/"output.raw",std::ios::binary);
        require(bool(f.read(reinterpret_cast<char*>(actual.data()),n*sizeof(float))),"missing LUT output");
        const double lo[]={g.reconstruction_offset_x_mm-1.5*.7,g.reconstruction_offset_y_mm-1.5*.8,g.reconstruction_offset_z_mm-2.5*1.1};
        const double hi[]={g.reconstruction_offset_x_mm+1.5*.7,g.reconstruction_offset_y_mm+1.5*.8,g.reconstruction_offset_z_mm+2.5*1.1};
        const auto dg=pd::makeDeviceGeometry(c,2,1);double peak=0;int hit=0,miss=0;
        for(int v=0;v<g.views;++v)for(int y=0;y<g.detector_v;++y)for(int x=0;x<g.detector_u;++x){
            const auto i=(size_t(v)*g.detector_v+y)*g.detector_u+x;
            const double bone=boxLength(pd::scannerRay(dg,v,float(x),float(y)),lo,hi);
            bone>0?++hit:++miss;require(std::isfinite(actual[i]),"nonfinite LUT correction");
            peak=std::max(peak,std::abs(actual[i]-(measured[i]+.001*bone*scale)));
        }
        require(hit>0&&miss>0,"degenerate bone path fixture");require(peak<2e-6,"bone scalar FP/scale/pose does not match analytic box length");
        std::cout<<"kind="<<int(kind)<<" scale="<<scale<<" max_abs="<<peak<<" hit="<<hit<<" miss="<<miss<<'\n';
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
