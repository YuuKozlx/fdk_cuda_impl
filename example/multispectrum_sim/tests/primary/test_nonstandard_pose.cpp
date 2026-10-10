#include "TestChecks.hpp"
#include "LibraryGeometry.hpp"
#include "NonStandardFpBp/SourceOrbitTomography/YkSourceOrbitTomography.hpp"
#include "NonStandardFpBp/NikonTiltedCt/YkNikonTiltedCtGeometry.hpp"
#include <iostream>
#include <limits>
using namespace primary_test;
namespace {
constexpr double pi=3.14159265358979323846;
void vectorNear(float3 v,float3 e,const char* name) {
    near(v.x,e.x,3e-5,std::string(name)+" X");
    near(v.y,e.y,3e-5,std::string(name)+" Y");
    near(v.z,e.z,3e-5,std::string(name)+" Z");
}
float3 xyz(float4 v){return make_float3(v.x,v.y,v.z);}
float3 center(const YK::SConeProjGeomVec& g,int u,int v){
    return make_float3(g.detS.x+.5f*(u-1)*g.detU.x+.5f*(v-1)*g.detV.x,
        g.detS.y+.5f*(u-1)*g.detU.y+.5f*(v-1)*g.detV.y,
        g.detS.z+.5f*(u-1)*g.detU.z+.5f*(v-1)*g.detV.z);
}
float3 rx(float3 v,double a){return make_float3(v.x,v.y*std::cos(a)-v.z*std::sin(a),v.y*std::sin(a)+v.z*std::cos(a));}
float3 rz(float3 v,double a){return make_float3(v.x*std::cos(a)-v.y*std::sin(a),v.x*std::sin(a)+v.y*std::cos(a),v.z);}
void compare(const std::vector<YK::SConeProjGeomVec>& a,const std::vector<YK::SConeProjGeomVec>& b){
    require(a.size()==b.size(),"geometry view count");
    for(size_t i=0;i<a.size();++i){
        vectorNear(xyz(a[i].src),xyz(b[i].src),"source");
        vectorNear(xyz(a[i].detS),xyz(b[i].detS),"detector start");
        vectorNear(xyz(a[i].detU),xyz(b[i].detU),"detector U");
        vectorNear(xyz(a[i].detV),xyz(b[i].detV),"detector V");
    }
}
void planar(){
    YK::NonStandardFpBp::SourceOrbitGeometryConfig c{};
    c.detector_u=5;c.detector_v=3;c.detector_pixel_u_mm=.5f;c.detector_pixel_v_mm=.8f;
    c.sid_mm=200;c.detector_y_mm=100;c.source_orbit_radius_mm=15;
    const std::vector<float> angles={0,float(pi/2),float(pi)};
    std::vector<YK::SConeProjGeomVec> baseline,g;
    require(YK::NonStandardFpBp::buildSourceOrbitGeometry(c,angles,baseline),"nominal planar build");
    for(const auto& view:baseline)vectorNear(center(view,5,3),make_float3(0,100,0),"nominal center");
    c.detector_pose.offset_unv_mm=make_float3(5,2,10);
    require(YK::NonStandardFpBp::buildSourceOrbitGeometry(c,angles,g),"offset planar build");
    for(size_t i=0;i<g.size();++i){
        vectorNear(center(g[i],5,3),make_float3(5,98,10),"offset center");
        vectorNear(xyz(g[i].src),xyz(baseline[i].src),"offset must not move source");
    }
    c.detector_pose.tilt_u_rad=float(pi/6);
    require(YK::NonStandardFpBp::buildSourceOrbitGeometry(c,angles,g),"U tilt planar build");
    const float s=.5f,cs=std::sqrt(.75f);
    for(const auto& view:g){
        vectorNear(xyz(view.detU),make_float3(.5f,0,0),"U axis preserved");
        vectorNear(xyz(view.detV),make_float3(0,-.8f*s,.8f*cs),"U tilt V");
        vectorNear(center(view,5,3),make_float3(5,100-2*cs-10*s,-2*s+10*cs),"tilted local offsets");
    }
    c.detector_pose={};c.detector_pose.tilt_v_rad=float(pi/6);
    require(YK::NonStandardFpBp::buildSourceOrbitGeometry(c,angles,g),"V tilt planar build");
    vectorNear(xyz(g[0].detU),make_float3(.5f*cs,.5f*s,0),"V tilt U");
    vectorNear(xyz(g[0].detV),make_float3(0,0,.8f),"V axis preserved");
    c.detector_pose={};c.detector_pose.tilt_n_rad=float(pi/2);
    require(YK::NonStandardFpBp::buildSourceOrbitGeometry(c,angles,g),"N tilt planar build");
    vectorNear(xyz(g[0].detU),make_float3(0,0,.5f),"N tilt U");
    vectorNear(xyz(g[0].detV),make_float3(-.8f,0,0),"N tilt V");
    c.detector_pose.tilt_n_rad=std::numeric_limits<float>::quiet_NaN();
    require(!YK::NonStandardFpBp::buildSourceOrbitGeometry(c,angles,g),"NaN pose accepted");
}
void tilted(){
    YK::NonStandardFpBp::NikonTiltedCt::GeometryConfig c{};
    c.detector_u=5;c.detector_v=3;c.detector_pixel_u_mm=.5f;c.detector_pixel_v_mm=.8f;
    c.sid_mm=200;c.sdd_mm=300;c.source=make_float3(0,-200,0);c.tilt_rad=float(pi/6);
    c.detector_pose.offset_unv_mm=make_float3(5,2,10);c.detector_pose.tilt_u_rad=float(pi/12);
    std::vector<YK::NonStandardFpBp::NonStandardViewPose> poses;
    std::vector<YK::SConeProjGeomVec> g;
    const std::vector<float> angles={0,float(pi/2),float(pi)};
    require(YK::NonStandardFpBp::NikonTiltedCt::buildTiltedAxisGeometry(c,angles,poses,g),"tilted pose build");
    const float3 panel=make_float3(5,100-2*std::cos(pi/12)-10*std::sin(pi/12),
        -2*std::sin(pi/12)+10*std::cos(pi/12));
    for(size_t i=0;i<g.size();++i){
        vectorNear(center(g[i],5,3),rz(rx(panel,pi/6),-angles[i]),"panel tilt then scan");
        vectorNear(xyz(g[i].detU),rz(make_float3(.5f,0,0),-angles[i]),"tilted U");
        vectorNear(xyz(g[i].detV),rz(rx(make_float3(0,0,.8f),pi/4),-angles[i]),"tilted V");
        vectorNear(xyz(g[i].src),rz(rx(make_float3(0,-200,0),pi/6),-angles[i]),"source unaffected by detector pose");
    }
}
void adapter(){
    using namespace yk::spectral;
    SimulationConfig config;
    auto& g=config.geometry.parameters;
    g.detector_u=5;g.detector_v=3;g.views=3;g.views_per_turn=3;g.sid_mm=200;g.sdd_mm=300;
    g.pixel_u_mm=.5;g.pixel_v_mm=.8;g.volume_x=g.volume_y=g.volume_z=16;
    g.reconstruction_volume_x=g.reconstruction_volume_y=g.reconstruction_volume_z=16;
    g.offset_u_mm=5;g.offset_n_mm=2;g.offset_v_mm=10;g.tilt_u_rad=pi/12;
    std::vector<float> angles;
    require(YK::NonStandardFpBp::buildUniformTrajectoryParameters({3,0,float(2*pi),1},angles),"adapter angles");
    config.geometry.kind=GeometryKind::PlanarSourceOrbit;g.planar_detector_y_mm=100;g.source_orbit_radius_mm=15;
    YK::NonStandardFpBp::SourceOrbitGeometryConfig p{};
    p.detector_u=5;p.detector_v=3;p.detector_pixel_u_mm=.5f;p.detector_pixel_v_mm=.8f;
    p.sid_mm=200;p.detector_y_mm=100;p.source_orbit_radius_mm=15;
    p.detector_pose.offset_unv_mm=make_float3(5,2,10);p.detector_pose.tilt_u_rad=float(pi/12);
    std::vector<YK::SConeProjGeomVec> expected;
    require(YK::NonStandardFpBp::buildSourceOrbitGeometry(p,angles,expected),"adapter planar baseline");
    compare(buildNonStandardProjectionViews(config),expected);
    config.geometry.kind=GeometryKind::NikonTilted;g.nikon_tilt_rad=pi/6;
    YK::NonStandardFpBp::NikonTiltedCt::GeometryConfig n{};
    n.detector_u=5;n.detector_v=3;n.detector_pixel_u_mm=.5f;n.detector_pixel_v_mm=.8f;
    n.sid_mm=200;n.sdd_mm=300;n.source=make_float3(0,-200,0);n.tilt_rad=float(pi/6);n.detector_pose=p.detector_pose;
    std::vector<YK::NonStandardFpBp::NonStandardViewPose> poses;
    require(YK::NonStandardFpBp::NikonTiltedCt::buildTiltedAxisGeometry(n,angles,poses,expected),"adapter tilted baseline");
    compare(buildNonStandardProjectionViews(config),expected);
}
}
int main(){
    try{planar();tilted();adapter();std::cout<<"non-standard flat detector pose: PASS\n";return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
