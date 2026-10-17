#include "primary/DisplayGeometry.hpp"
#include "LibraryGeometry.hpp"
#include <stdexcept>
namespace yk::spectral {
std::vector<ProjectionViewGeometry> buildDisplayGeometry(const SimulationConfig& c) {
    using namespace primary_detail;
    if(c.geometry.parameters.views<=0)throw std::runtime_error("geometry needs positive views");
    if(c.geometry.parameters.views_per_turn<=0)throw std::runtime_error("geometry needs positive views per turn");
    auto g=makeDeviceGeometry(c,static_cast<int>(c.projection.materials.size()),1);
    const auto cone_geometry=buildNonStandardProjectionViews(c);
    std::vector<DeviceConeView> cone_views;
    for(const auto& v:cone_geometry)cone_views.push_back({make_float3(v.src.x,v.src.y,v.src.z),make_float3(v.detS.x,v.detS.y,v.detS.z),
        make_float3(v.detU.x,v.detU.y,v.detU.z),make_float3(v.detV.x,v.detV.y,v.detV.z)});
    if(!cone_views.empty()&&static_cast<int>(cone_views.size())!=g.views)throw std::runtime_error("geometry view count mismatch");
    g.use_cone_views=cone_views.empty()?0:1;g.cone_views=cone_views.data();
    std::vector<ProjectionViewGeometry> frames;frames.reserve(g.views);
    for(int view=0;view<g.views;++view)frames.push_back(resolveDisplayGeometry(g,view));
    return frames;
}
}
