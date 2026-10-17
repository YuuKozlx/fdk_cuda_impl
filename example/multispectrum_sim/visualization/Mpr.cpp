#include "Mpr.hpp"
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkImageReslice.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkPointData.h>

namespace yk::visualization {
namespace {
    using Vector = std::array<double, 3>;
    double dot(const Vector& a, const Vector& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    Vector cross(const Vector& a, const Vector& b) { return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] }; }
    Vector rotated(Vector p, const Vector& degrees)
    {
        for (int axis = 0; axis < 3; ++axis) {
            const double a = degrees[axis] * 3.14159265358979323846 / 180.;
            const int j = (axis + 1) % 3, k = (axis + 2) % 3;
            const double y = p[j], z = p[k];
            p[j] = std::cos(a) * y - std::sin(a) * z;
            p[k] = std::sin(a) * y + std::cos(a) * z;
        }
        return p;
    }
    Vector axisRotated(const Vector& p, const Vector& axis, double radians)
    {
        Vector out {};
        const auto perpendicular = cross(axis, p);
        const double c = std::cos(radians), s = std::sin(radians), projection = dot(axis, p);
        for (int a = 0; a < 3; ++a)
            out[a] = p[a] * c + perpendicular[a] * s + axis[a] * projection * (1 - c);
        return out;
    }
    MprPlane referenceFrame(const MprSettings& state, int panel)
    {
        if (panel < 0 || panel > 2)
            throw std::invalid_argument("Invalid reference panel");
        for (double v : state.reference_axes)
            if (!std::isfinite(v))
                throw std::invalid_argument("Nonfinite reference frame");
        for (double v : state.reference_roll_deg)
            if (!std::isfinite(v))
                throw std::invalid_argument("Nonfinite reference roll");
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double d = 0;
                for (int r = 0; r < 3; ++r)
                    d += state.reference_axes[r * 3 + a] * state.reference_axes[r * 3 + b];
                if (std::abs(d - (a == b ? 1. : 0.)) > 1e-5)
                    throw std::invalid_argument("Reference frame must be orthonormal");
            }
        const int u = panel == 2 ? 1 : 0, v = panel == 0 ? 1 : 2;
        MprPlane p;
        for (int a = 0; a < 3; ++a) {
            p.u[a] = state.reference_axes[a * 3 + u];
            p.v[a] = state.reference_axes[a * 3 + v];
        }
        p.normal = cross(p.u, p.v);
        const double angle = state.reference_roll_deg[panel] * 3.14159265358979323846 / 180.;
        p.u = axisRotated(p.u, p.normal, angle);
        p.v = axisRotated(p.v, p.normal, angle);
        return p;
    }
}
bool mprReferenceIsOrthogonal(const MprSettings& state)
{
    return state.reference_axes == std::array<double, 9> { 1, 0, 0, 0, 1, 0, 0, 0, 1 } && state.reference_roll_deg == std::array<double, 3> {};
}
void setMprRotation(MprSettings& state, const std::array<double, 3>& degrees)
{
    for (double v : degrees)
        if (!std::isfinite(v))
            throw std::invalid_argument("Nonfinite MPR angle");
    auto candidate = state;
    candidate.rotation_deg = degrees;
    candidate.reference_roll_deg = {};
    for (int col = 0; col < 3; ++col) {
        Vector axis {};
        axis[col] = 1;
        axis = rotated(axis, degrees);
        for (int r = 0; r < 3; ++r)
            candidate.reference_axes[r * 3 + col] = axis[r];
    }
    referenceFrame(candidate, 0);
    state = candidate;
}
void rotateMprReference(MprSettings& state, int panel, double radians)
{
    if (!std::isfinite(radians))
        throw std::invalid_argument("Nonfinite reference rotation");
    const auto plane = referenceFrame(state, panel);
    auto candidate = state;
    for (int col = 0; col < 3; ++col) {
        Vector axis {};
        for (int r = 0; r < 3; ++r)
            axis[r] = state.reference_axes[r * 3 + col];
        axis = axisRotated(axis, plane.normal, radians);
        for (int r = 0; r < 3; ++r)
            candidate.reference_axes[r * 3 + col] = axis[r];
    }
    candidate.reference_roll_deg[panel] = std::remainder(candidate.reference_roll_deg[panel] - radians * 180. / 3.14159265358979323846, 360.);
    referenceFrame(candidate, panel);
    const auto& r = candidate.reference_axes;
    const double cosine_y = std::hypot(r[0], r[3]);
    const double y = std::atan2(-r[6], cosine_y);
    const double x = cosine_y > 1e-10 ? std::atan2(r[7], r[8]) : 0;
    const double z = cosine_y > 1e-10 ? std::atan2(r[3], r[0]) : std::atan2(-r[1], r[4]);
    constexpr double degrees = 180. / 3.14159265358979323846;
    candidate.rotation_deg = { x * degrees, y * degrees, z * degrees };
    state = candidate;
}
MprPlane makeMprPlane(const FloatVolume& volume, const MprSettings& state, int panel)
{
    if (panel < 0 || panel > 3)
        throw std::invalid_argument("Invalid MPR panel");
    for (double a : state.rotation_deg)
        if (!std::isfinite(a))
            throw std::invalid_argument("Nonfinite MPR angle");
    if (!std::isfinite(state.offset_mm))
        throw std::invalid_argument("Nonfinite MPR offset");
    MprPlane p;
    for (int a = 0; a < 3; ++a) {
        if (volume.dimensions[a] <= 0 || !std::isfinite(volume.spacing[a]) || volume.spacing[a] <= 0)
            throw std::invalid_argument("Invalid MPR grid");
        p.center[a] = std::clamp(state.indices[a], 0, volume.dimensions[a] - 1) * volume.spacing[a];
    }
    if (panel < 3 && mprReferenceIsOrthogonal(state)) {
        const int fixed = panel == 0 ? 2 : panel == 1 ? 1
                                                      : 0;
        const int x = fixed == 0 ? 1 : 0, y = fixed == 2 ? 1 : 2;
        p.u = {};
        p.v = {};
        p.u[x] = 1;
        p.v[y] = 1;
        p.normal = cross(p.u, p.v);
        p.width = volume.dimensions[x];
        p.height = volume.dimensions[y];
        p.spacing = { volume.spacing[x], volume.spacing[y] };
        p.start = { -p.center[x], -p.center[y] };
        return p;
    }
    if (panel < 3) {
        const auto frame = referenceFrame(state, panel);
        p.u = frame.u;
        p.v = frame.v;
        p.normal = frame.normal;
    } else {
        p.u = rotated({ 1, 0, 0 }, state.rotation_deg);
        p.v = rotated({ 0, 1, 0 }, state.rotation_deg);
        p.normal = cross(p.u, p.v);
        for (int a = 0; a < 3; ++a)
            p.center[a] += p.normal[a] * state.offset_mm;
    }
    if (panel < 3) {
        const int u = int(std::max_element(p.u.begin(), p.u.end(), [](double a, double b) { return std::abs(a) < std::abs(b); }) - p.u.begin());
        const int v = int(std::max_element(p.v.begin(), p.v.end(), [](double a, double b) { return std::abs(a) < std::abs(b); }) - p.v.begin());
        if (u != v && std::abs(std::abs(p.u[u]) - 1) < 1e-10 && std::abs(std::abs(p.v[v]) - 1) < 1e-10) {
            p.width = volume.dimensions[u];
            p.height = volume.dimensions[v];
            p.spacing = { volume.spacing[u], volume.spacing[v] };
            p.start = { p.u[u] > 0 ? -p.center[u] : p.center[u] - (volume.dimensions[u] - 1) * volume.spacing[u],
                p.v[v] > 0 ? -p.center[v] : p.center[v] - (volume.dimensions[v] - 1) * volume.spacing[v] };
            return p;
        }
    }
    double minU = std::numeric_limits<double>::infinity(), minV = minU, maxU = -minU, maxV = -minU;
    for (int corner = 0; corner < 8; ++corner) {
        Vector r {};
        for (int a = 0; a < 3; ++a)
            r[a] = ((corner & (1 << a)) ? (volume.dimensions[a] - 1) * volume.spacing[a] : 0) - p.center[a];
        const double u = dot(r, p.u), v = dot(r, p.v);
        minU = std::min(minU, u);
        maxU = std::max(maxU, u);
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
    }
    // Keep native finest voxel spacing. A configurable lower display resolution can be added independently.
    double step = *std::min_element(volume.spacing.begin(), volume.spacing.end());
    const double w = std::ceil((maxU - minU) / step) + 1, h = std::ceil((maxV - minV) / step) + 1;
    if (w > 8192 || h > 8192 || w * h > 16777216)
        throw std::runtime_error("MPR plane exceeds display allocation limit");
    p.width = std::max(1, int(w));
    p.height = std::max(1, int(h));
    p.start = { minU, minV };
    p.spacing = { step, step };
    return p;
}
std::array<double, 3> mprWorldPosition(const FloatVolume& volume, const std::array<int, 3>& index)
{
    auto position = volume.origin;
    for (int r = 0; r < 3; ++r)
        for (int a = 0; a < 3; ++a)
            position[r] += volume.direction[r * 3 + a] * index[a] * volume.spacing[a];
    return position;
}
std::array<int, 3> mprPick(const FloatVolume& volume, const MprPlane& p, double x, double y)
{
    std::array<int, 3> index {};
    for (int a = 0; a < 3; ++a) {
        double mm = p.center[a] + p.u[a] * (p.start[0] + x * p.spacing[0]) + p.v[a] * (p.start[1] + y * p.spacing[1]);
        index[a] = std::clamp(int(std::lround(mm / volume.spacing[a])), 0, volume.dimensions[a] - 1);
    }
    return index;
}
struct MprReslicer::Impl {
    vtkNew<vtkImageData> image;
    vtkNew<vtkImageReslice> reslice;
    bool ready = false;
};
MprReslicer::MprReslicer()
    : impl_(std::make_unique<Impl>())
{
}
MprReslicer::~MprReslicer() = default;
void MprReslicer::clear()
{
    impl_->reslice->SetInputData(nullptr);
    impl_->image->Initialize();
    impl_->ready = false;
}
void MprReslicer::setInput(const FloatVolume& volume)
{
    clear();
    auto& p = *impl_;
    p.image->SetDimensions(volume.dimensions.data());
    p.image->SetSpacing(volume.spacing.data());
    vtkNew<vtkFloatArray> scalars;
    scalars->SetNumberOfComponents(1);
    scalars->SetArray(const_cast<float*>(volume.values.data()), static_cast<vtkIdType>(volume.values.size()), 1);
    p.image->GetPointData()->SetScalars(scalars);
    p.reslice->SetInputData(p.image);
    p.reslice->SetOutputDimensionality(2);
    p.reslice->SetBackgroundLevel(std::numeric_limits<float>::quiet_NaN());
    p.reslice->BorderOff();
    p.ready = true;
}
FloatSlice MprReslicer::slice(const MprPlane& plane, MprInterpolation interpolation)
{
    if (!impl_->ready)
        throw std::runtime_error("No MPR input volume");
    if (interpolation < MprInterpolation::Nearest || interpolation > MprInterpolation::Cubic)
        throw std::invalid_argument("Invalid MPR interpolation");
    auto& filter = impl_->reslice;
    vtkNew<vtkMatrix4x4> axes;
    axes->Identity();
    for (int r = 0; r < 3; ++r) {
        axes->SetElement(r, 0, plane.u[r]);
        axes->SetElement(r, 1, plane.v[r]);
        axes->SetElement(r, 2, plane.normal[r]);
        axes->SetElement(r, 3, plane.center[r]);
    }
    filter->SetResliceAxes(axes);
    filter->SetOutputExtent(0, plane.width - 1, 0, plane.height - 1, 0, 0);
    filter->SetOutputOrigin(plane.start[0], plane.start[1], 0);
    filter->SetOutputSpacing(plane.spacing[0], plane.spacing[1], 1);
    filter->SetInterpolationMode(static_cast<int>(interpolation));
    filter->Update();
    FloatSlice result;
    result.width = plane.width;
    result.height = plane.height;
    result.spacing = plane.spacing;
    const auto* data = static_cast<const float*>(filter->GetOutput()->GetScalarPointer());
    result.values.assign(data, data + size_t(plane.width) * plane.height);
    return result;
}
}
