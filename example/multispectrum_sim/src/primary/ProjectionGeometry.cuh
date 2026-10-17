#pragma once

// Internal device geometry and world/phantom coordinate transforms.

#include <cuda_runtime.h>
#include <cmath>
#include "config/SimConfig.hpp"

namespace yk::spectral::primary_detail {

struct DeviceConeView { float3 source, detector_start, detector_u, detector_v; };
struct DeviceGeometry {
    int nu{};
    int nv{};
    int views{};
    int views_per_turn{};
    int rotation_direction{};
    int samples{};
    int nx{};
    int ny{};
    int nz{};
    int nm{};
    int ne{};
    float pu{};
    float pv{};
    float sid{};
    float sdd{};
    float vx{};
    float vy{};
    float vz{};
    float normal_fu{};
    float normal_fv{};
    float defocus_fu{};
    float defocus_fv{};
    float focal_epsilon{};
    float offset_u{};
    float offset_n{};
    float offset_v{};
    float source_x{};
    float source_y{};
    float source_z{};
    float tilt_u{};
    float tilt_v{};
    float tilt_n{};
    float start{};
    float pitch{};
    float start_z{};
    float phantom_x{};
    float phantom_y{};
    float phantom_z{};
    float phantom_rx{};
    float phantom_ry{};
    float phantom_rz{};
    int cylindrical{};
    int geometry_flux{};
    int detector_response{};
    int use_constant_tables{};
    int use_label_texture{};
    const float* detector_efficiency{};
    cudaTextureObject_t label_texture{};
    const DeviceConeView* cone_views{};
    int use_cone_views{};
};

inline DeviceGeometry makeDeviceGeometry(const SimulationConfig& c,
    int nm, int ne) {
    const auto& g = c.geometry.parameters;
    const int cylindrical = c.geometry.kind == GeometryKind::CylCbct ||
        c.geometry.kind == GeometryKind::CylHelical;
    DeviceGeometry result{};
    result.nu = g.detector_u;
    result.nv = g.detector_v;
    result.views = g.views;
    result.views_per_turn = g.views_per_turn;
    result.rotation_direction = g.rotation_direction;
    result.samples = c.projection.sampling.samples_per_pixel;
    result.nx = g.volume_x;
    result.ny = g.volume_y;
    result.nz = g.volume_z;
    result.nm = nm;
    result.ne = ne;
    result.pu = (float)g.pixel_u_mm;
    result.pv = (float)g.pixel_v_mm;
    result.sid = (float)g.sid_mm;
    result.sdd = (float)g.sdd_mm;
    result.vx = (float)g.voxel_x_mm;
    result.vy = (float)g.voxel_y_mm;
    result.vz = (float)g.voxel_z_mm;
    result.normal_fu = (float)(c.projection.focal_spot.enabled?c.projection.focal_spot.normal_size_u_mm:0);
    result.normal_fv = (float)(c.projection.focal_spot.enabled?c.projection.focal_spot.normal_size_v_mm:0);
    result.defocus_fu = (float)(c.projection.focal_spot.defocus_enabled?c.projection.focal_spot.defocus_size_u_mm:0);
    result.defocus_fv = (float)(c.projection.focal_spot.defocus_enabled?c.projection.focal_spot.defocus_size_v_mm:0);
    result.focal_epsilon = (float)(c.projection.focal_spot.defocus_enabled?c.projection.focal_spot.epsilon:0);
    result.offset_u = (float)g.offset_u_mm;
    result.offset_n = (float)g.offset_n_mm;
    result.offset_v = (float)g.offset_v_mm;
    result.source_x = (float)g.source_offset_x_mm;
    result.source_y = (float)g.source_offset_y_mm;
    result.source_z = (float)g.source_offset_z_mm;
    result.tilt_u = (float)g.tilt_u_rad;
    result.tilt_v = (float)g.tilt_v_rad;
    result.tilt_n = (float)g.tilt_n_rad;
    result.start = (float)g.start_angle_rad;
    const bool helical = c.geometry.kind == GeometryKind::FlatHelical || c.geometry.kind == GeometryKind::CylHelical;
    result.pitch = helical ? (float)g.pitch_mm_per_turn : 0.f;
    result.start_z = (float)g.start_z_mm;
    result.phantom_x = (float)g.phantom_offset_x_mm;
    result.phantom_y = (float)g.phantom_offset_y_mm;
    result.phantom_z = (float)g.phantom_offset_z_mm;
    result.phantom_rx = (float)g.phantom_rotation_x_rad;
    result.phantom_ry = (float)g.phantom_rotation_y_rad;
    result.phantom_rz = (float)g.phantom_rotation_z_rad;
    result.cylindrical = cylindrical;
    result.geometry_flux = c.projection.geometry_flux.enabled ? 1 : 0;
    result.detector_response = 0;
    result.use_constant_tables = 0;
    result.use_label_texture = 0;
    result.detector_efficiency = nullptr;
    result.label_texture = 0;
    return result;
}

__host__ __device__ inline float3 add3(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 scale3(float3 a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline float3 cross3(float3 a, float3 b) { return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__host__ __device__ inline float3 rotateAxis(float3 value, float3 axis, float angle) {
    const float c=cosf(angle), s=sinf(angle);
    return add3(add3(scale3(value,c),scale3(cross3(axis,value),s)),scale3(axis,(axis.x*value.x+axis.y*value.y+axis.z*value.z)*(1.f-c)));
}
__host__ __device__ inline void detectorFrame(DeviceGeometry g, float angle, float3& u, float3& v, float3& n) {
    const float ca=cosf(angle), sa=sinf(angle);
    u=make_float3(-sa,ca,0.f); v=make_float3(0.f,0.f,1.f); n=make_float3(ca,sa,0.f);
    v=rotateAxis(v,u,g.tilt_u); n=rotateAxis(n,u,g.tilt_u);
    u=rotateAxis(u,v,g.tilt_v); n=rotateAxis(n,v,g.tilt_v);
    u=rotateAxis(u,n,g.tilt_n); v=rotateAxis(v,n,g.tilt_n);
}
// Flat focal spot sizes use detector-local U/V for both scanner and ConeVec rays.
// Normalize pixel vectors so focal dimensions remain physical millimeters.
__host__ __device__ inline float3 sampleFlatFocalSource(float3 nominal, float3 u, float3 v,
    float offset_u_mm, float offset_v_mm) {
    const float ul = sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
    const float vl = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return add3(nominal, add3(scale3(u, offset_u_mm / ul),
        scale3(v, offset_v_mm / vl)));
}

__host__ __device__ inline float3 phantomInverse(DeviceGeometry g, float3 value, bool direction) {
    const float3 x = make_float3(1.f, 0.f, 0.f);
    const float3 y = make_float3(0.f, 1.f, 0.f);
    const float3 z = make_float3(0.f, 0.f, 1.f);
    value = rotateAxis(value, z, -g.phantom_rz);
    value = rotateAxis(value, y, -g.phantom_ry);
    value = rotateAxis(value, x, -g.phantom_rx);
    // Forward pose order is offset first, then XYZ rotation. The inverse ray
    // transform therefore applies inverse rotation before removing the offset.
    if (!direction) value = make_float3(value.x-g.phantom_x, value.y-g.phantom_y, value.z-g.phantom_z);
    return value;
}

// Shared by the CUDA projector and its display snapshot. u/v are sample indices;
// integer values are pixel centers, -.5 and size-.5 are physical panel edges.
struct ScannerRay { float3 source, detector; };
__host__ __device__ inline ScannerRay scannerRay(DeviceGeometry g, int view,
    float u, float v, float focal_u=0, float focal_v=0, float jitter_u=0, float jitter_v=0) {
    if(g.use_cone_views) {
        const auto c=g.cone_views[view];
        return {sampleFlatFocalSource(c.source,c.detector_u,c.detector_v,focal_u,focal_v),
            add3(c.detector_start,add3(scale3(c.detector_u,u+jitter_u),scale3(c.detector_v,v+jitter_v)))};
    }
    const float pi=3.14159265358979323846f;
    const float angle=g.start+static_cast<float>(g.rotation_direction)*2.f*pi*view/g.views_per_turn;
    const float phase=angle-.5f*pi, ca=cosf(phase),sa=sinf(phase);
    const float z=g.start_z+g.pitch*(angle-g.start)/(2.f*pi);
    float3 ua,va,na;detectorFrame(g,phase,ua,va,na);
    float3 source=make_float3(g.sid*ca+g.source_x*ca-g.source_y*sa,
        g.sid*sa+g.source_x*sa+g.source_y*ca,z+g.source_z);
    // Preserve the projector's pixel-center subtraction / jitter order.
    const float du=(u-.5f*(g.nu-1)+jitter_u)*g.pu+g.offset_u;
    const float dv=(v-.5f*(g.nv-1)+jitter_v)*g.pv+g.offset_v;
    const float3 principal=add3(make_float3(0,0,z),scale3(make_float3(ca,sa,0),-(g.sdd-g.sid)));
    if(g.cylindrical){
        source=add3(source,make_float3(focal_u*ca,focal_u*sa,focal_v));
        const float gamma=du/g.sdd;
        const auto center=add3(principal,scale3(na,g.offset_n+g.sdd));
        return {source,add3(add3(center,scale3(add3(scale3(na,-cosf(gamma)),scale3(ua,sinf(gamma))),g.sdd)),scale3(va,dv))};
    }
    source=sampleFlatFocalSource(source,ua,va,focal_u,focal_v);
    return {source,add3(add3(add3(principal,scale3(na,g.offset_n)),scale3(ua,du)),scale3(va,dv))};
}
__host__ __device__ inline float3 phantomForward(DeviceGeometry g,float3 value,bool direction) {
    if(!direction)value=add3(value,make_float3(g.phantom_x,g.phantom_y,g.phantom_z));
    value=rotateAxis(value,make_float3(1,0,0),g.phantom_rx);
    value=rotateAxis(value,make_float3(0,1,0),g.phantom_ry);
    return rotateAxis(value,make_float3(0,0,1),g.phantom_rz);
}


} // namespace yk::spectral::primary_detail
