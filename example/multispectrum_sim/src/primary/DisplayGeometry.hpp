#pragma once
#include "SpectralProjectionPipeline.hpp"
#include "ProjectionGeometry.cuh"
namespace yk::spectral::primary_detail {
// Same resolved host geometry for TOML scene preview and computed frames.
inline ProjectionViewGeometry resolveDisplayGeometry(DeviceGeometry g,int view) {
    auto point=[](float3 p)->yk::visualization::Point{return {p.x,p.y,p.z};};
    auto unit=[](float3 p){return scale3(p,1.f/sqrtf(p.x*p.x+p.y*p.y+p.z*p.z));};
    ProjectionViewGeometry out;out.view=view;
    const float cu=.5f*(g.nu-1),cv=.5f*(g.nv-1);
    auto center=scannerRay(g,view,cu,cv);out.source=point(center.source);out.detector_center=point(center.detector);
    float3 u,v,n;
    if(g.use_cone_views){u=unit(g.cone_views[view].detector_u);v=unit(g.cone_views[view].detector_v);}
    else {
        const float pi=3.14159265358979323846f;
        const float angle=g.start+g.rotation_direction*2.f*pi*view/g.views_per_turn-.5f*pi;
        detectorFrame(g,angle,u,v,n);
        if(g.cylindrical){const float gamma=g.offset_u/g.sdd;u=add3(scale3(u,cosf(gamma)),scale3(n,sinf(gamma)));}
        u=unit(u);v=unit(v);
    }
    out.detector_axes={point(u),point(v),point(unit(cross3(u,v)))};
    const int segments=g.cylindrical?128:1;
    for(int row=0;row<2;++row)for(int col=0;col<=segments;++col)
        out.detector.push_back(point(scannerRay(g,view,-.5f+g.nu*float(col)/segments,row?g.nv-.5f:-.5f).detector));
    const auto origin=phantomForward(g,make_float3(0,0,0),false);
    const float3 basis[]={make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
    const auto t=point(origin);for(int r=0;r<3;++r)out.object_pose[r*4+3]=t[r];
    for(int c=0;c<3;++c){const auto b=point(phantomForward(g,basis[c],true));for(int r=0;r<3;++r)out.object_pose[r*4+c]=b[r];}
    return out;
}
}
namespace yk::spectral {
std::vector<ProjectionViewGeometry> buildDisplayGeometry(const SimulationConfig&);
}
