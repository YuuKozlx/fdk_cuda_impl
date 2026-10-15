#include "primary/ScalarProjector.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace yk::spectral::scalar_fp {
namespace {
    bool finite(float3 p) { return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z); }
    float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    float3 cross(float3 a, float3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
    float3 sub(float3 a, float3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
}
void validateGrid(const Grid& g, int nu, int nv, int views, Method method)
{
    if ((method != Method::Siddon && method != Method::Joseph) || g.nx <= 0 || g.ny <= 0 || g.nz <= 0 || nu <= 0 || nv <= 0 || views <= 0 || !finite(g.spacing) || !finite(g.center) || g.spacing.x <= 0 || g.spacing.y <= 0 || g.spacing.z <= 0)
        throw std::invalid_argument("invalid scalar FP grid/shape/method");
    size_t n = 1;
    for (int d : { g.nx, g.ny, g.nz }) {
        if (n > size_t(-1) / sizeof(float) / size_t(d))
            throw std::invalid_argument("FP volume overflow");
        n *= d;
    }
    n = 1;
    for (int d : { nu, nv, views }) {
        if (n > size_t(-1) / sizeof(float) / size_t(d))
            throw std::invalid_argument("FP projection overflow");
        n *= d;
    }
}
std::vector<DeviceView> prepareViews(const std::vector<View>& views, const Grid& g, int nu, int nv, Method method, DetectorAnchor anchor)
{
    if (views.size() > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("too many FP views");
    validateGrid(g, nu, nv, int(views.size()), method);
    if (anchor != DetectorAnchor::Center && anchor != DetectorAnchor::Pixel00)
        throw std::invalid_argument("invalid detector anchor");
    std::vector<DeviceView> out;
    out.reserve(views.size());
    for (const auto& v : views) {
        auto normal = cross(v.u, v.v);
        auto center = v.detector;
        if (anchor == DetectorAnchor::Pixel00) {
            center.x += .5f * (nu - 1) * v.u.x + .5f * (nv - 1) * v.v.x;
            center.y += .5f * (nu - 1) * v.u.y + .5f * (nv - 1) * v.v.y;
            center.z += .5f * (nu - 1) * v.u.z + .5f * (nv - 1) * v.v.z;
        }
        auto ray = sub(center, v.source);
        const double normal2 = dot(normal, normal), ray2 = dot(ray, ray);
        if (!finite(v.source) || !finite(v.detector) || !finite(v.u) || !finite(v.v) || !std::isfinite(normal2) || !std::isfinite(ray2) || normal2 <= 0 || ray2 <= 0 || std::abs(dot(normal, ray)) <= 1e-7 * std::sqrt(normal2 * ray2))
            throw std::invalid_argument("invalid scalar FP view");
        DeviceView d;
        d.src = v.source;
        d.detU = v.u;
        d.detV = v.v;
        d.detS = { v.detector.x - .5f * (nu - 1) * v.u.x - .5f * (nv - 1) * v.v.x, v.detector.y - .5f * (nu - 1) * v.u.y - .5f * (nv - 1) * v.v.y, v.detector.z - .5f * (nu - 1) * v.u.z - .5f * (nv - 1) * v.v.z };
        if (anchor == DetectorAnchor::Pixel00)
            d.detS = v.detector;
        const float x = std::abs(ray.x), y = std::abs(ray.y), z = std::abs(ray.z);
        d.axis = x >= y && x >= z ? 0 : y >= z ? 1
                                               : 2;
        if (method == Method::Joseph) {
            const float ix = 1.f / g.spacing.x, iy = 1.f / g.spacing.y, iz = 1.f / g.spacing.z;
            auto point = [&](float3 p) { return make_float3((p.x - g.center.x) * ix, (p.y - g.center.y) * iy, (p.z - g.center.z) * iz); };
            auto vector = [&](float3 p) { return make_float3(p.x * ix, p.y * iy, p.z * iz); };
            d.src = point(d.src);
            d.detS = point(d.detS);
            d.detU = vector(d.detU);
            d.detV = vector(d.detV);
        }
        if (!finite(d.src) || !finite(d.detS) || !finite(d.detU) || !finite(d.detV))
            throw std::invalid_argument("scalar FP coordinate overflow");
        out.push_back(d);
    }
    return out;
}
}
