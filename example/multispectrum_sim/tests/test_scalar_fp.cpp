#include "primary/ScalarProjector.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#ifdef SCALAR_FP_TEST_DLL
#include "YKCBCT/interface/YkReconstructionApi.hpp"
#endif
namespace fp = yk::spectral::scalar_fp;
void check(bool b, const char* m)
{
    if (!b)
        throw std::runtime_error(m);
}
void cudaCheck(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
template <class F>
void reject(F f)
{
    bool failed = false;
    try {
        f();
    } catch (const std::exception&) {
        failed = true;
    }
    check(failed, "invalid FP accepted");
}
template <class T>
struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { cudaCheck(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void upload(const std::vector<T>& v) { cudaCheck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
};
struct Texture {
    cudaArray_t array = nullptr;
    cudaTextureObject_t id = 0;
    Texture(const std::vector<float>& values, const fp::Grid& g, bool linear = true)
    {
        auto channel = cudaCreateChannelDesc<float>();
        cudaCheck(cudaMalloc3DArray(&array, &channel, make_cudaExtent(g.nx, g.ny, g.nz)));
        cudaMemcpy3DParms c {};
        c.srcPtr = make_cudaPitchedPtr(const_cast<float*>(values.data()), g.nx * sizeof(float), g.nx, g.ny);
        c.dstArray = array;
        c.extent = make_cudaExtent(g.nx, g.ny, g.nz);
        c.kind = cudaMemcpyHostToDevice;
        cudaCheck(cudaMemcpy3D(&c));
        cudaResourceDesc r {};
        r.resType = cudaResourceTypeArray;
        r.res.array.array = array;
        cudaTextureDesc t {};
        t.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
        t.readMode = cudaReadModeElementType;
        for (auto& a : t.addressMode)
            a = cudaAddressModeBorder;
        cudaCheck(cudaCreateTextureObject(&id, &r, &t, nullptr));
    }
    ~Texture()
    {
        cudaDestroyTextureObject(id);
        cudaFreeArray(array);
    }
};
std::vector<float> run(const std::vector<float>& volume, const fp::Grid& g, const std::vector<fp::View>& views, int nu, int nv, fp::Method method, int chunk = 0, int steps = 1, fp::DetectorAnchor anchor = fp::DetectorAnchor::Center, fp::SiddonMemory memory = fp::SiddonMemory::Linear)
{
    auto geometry = fp::prepareViews(views, g, nu, nv, method, anchor);
    Device<float> dv(volume.size()), out(size_t(nu) * nv * views.size());
    Device<fp::DeviceView> dg(geometry.size());
    dv.upload(volume);
    dg.upload(geometry);
    Texture texture(volume, g, memory != fp::SiddonMemory::Texture);
    cudaStream_t stream {};
    cudaCheck(cudaStreamCreate(&stream));
    try {
        int batch = chunk ? chunk : int(views.size());
        for (int start = 0; start < int(views.size()); start += batch)
            fp::launch(method, dv.p, texture.id, dg.p + start, out.p + size_t(start) * nu * nv, g, nu, nv, std::min(batch, int(views.size()) - start), false, stream, steps, memory);
        cudaCheck(cudaStreamSynchronize(stream));
        std::vector<float> result(size_t(nu) * nv * views.size());
        cudaCheck(cudaMemcpy(result.data(), out.p, result.size() * 4, cudaMemcpyDeviceToHost));
        fp::launch(method, dv.p, texture.id, dg.p, out.p, g, nu, nv, int(views.size()), true, stream, steps, memory);
        cudaCheck(cudaStreamSynchronize(stream));
        std::vector<float> doubled(result.size());
        cudaCheck(cudaMemcpy(doubled.data(), out.p, result.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < result.size(); ++i)
            check(std::isfinite(result[i]) && std::abs(doubled[i] - 2 * result[i]) < 2e-5, "FP accumulation mismatch");
        cudaStreamDestroy(stream);
        return result;
    } catch (...) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        throw;
    }
}
// Independent exact piecewise-constant integral: intersect ALL grid planes, sort
// segment parameters, then integrate midpoint cells. No traversal recurrence.
double oracle(const std::vector<float>& volume, const fp::Grid& g, const fp::View& view, int nu, int nv, int u, int v)
{
    double s[] = { view.source.x, view.source.y, view.source.z };
    double d[] = { view.detector.x + (u - .5 * (nu - 1)) * view.u.x + (v - .5 * (nv - 1)) * view.v.x - s[0], view.detector.y + (u - .5 * (nu - 1)) * view.u.y + (v - .5 * (nv - 1)) * view.v.y - s[1], view.detector.z + (u - .5 * (nu - 1)) * view.u.z + (v - .5 * (nv - 1)) * view.v.z - s[2] };
    int dims[] = { g.nx, g.ny, g.nz };
    double spacing[] = { g.spacing.x, g.spacing.y, g.spacing.z }, center[] = { g.center.x, g.center.y, g.center.z };
    std::vector<double> cuts { 0, 1 };
    for (int axis = 0; axis < 3; ++axis)
        if (std::abs(d[axis]) > 1e-14)
            for (int i = 0; i <= dims[axis]; ++i) {
                double t = (center[axis] + (i - .5 * dims[axis]) * spacing[axis] - s[axis]) / d[axis];
                if (t > 0 && t < 1)
                    cuts.push_back(t);
            }
    std::sort(cuts.begin(), cuts.end());
    double sum = 0;
    for (size_t i = 1; i < cuts.size(); ++i) {
        double t = .5 * (cuts[i] + cuts[i - 1]);
        int cell[3];
        bool inside = true;
        for (int a = 0; a < 3; ++a) {
            cell[a] = int(std::floor((s[a] + t * d[a] - center[a]) / spacing[a] + .5 * dims[a]));
            inside = inside && cell[a] >= 0 && cell[a] < dims[a];
        }
        if (inside)
            sum += volume[(cell[2] * g.ny + cell[1]) * g.nx + cell[0]] * (cuts[i] - cuts[i - 1]);
    }
    return sum * std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}
int main(int argc, char** argv)
{
    try {
        check(argc == 2, "test case required");
        std::string name = argv[1];
        if (name == "invalid") {
            fp::Grid g { 5, 6, 7, { .5f, .8f, 1.2f }, { 1, 2, 3 } };
            fp::View v { { -50, 0, 0 }, { 50, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            reject([&] { fp::prepareViews({}, g, 5, 5, fp::Method::Siddon); });
            auto bad = v;
            bad.v = bad.u;
            reject([&] { fp::prepareViews({ bad }, g, 5, 5, fp::Method::Joseph); });
            bad = v;
            bad.source = bad.detector;
            reject([&] { fp::prepareViews({ bad }, g, 5, 5, fp::Method::Siddon); });
            bad = v;
            bad.source.x = NAN;
            reject([&] { fp::prepareViews({ bad }, g, 5, 5, fp::Method::Joseph); });
            auto bg = g;
            bg.spacing.z = 0;
            reject([&] { fp::prepareViews({ v }, bg, 5, 5, fp::Method::Joseph); });
            reject([&] { fp::launch(fp::Method::Siddon, nullptr, 0, nullptr, nullptr, g, 1, 1, 1); });
        } else if (name == "analytic") {
            for (int axis = 0; axis < 3; ++axis)
                for (int size : { 1, 8, 259 })
                    for (int steps : { 1, 2, 4 }) {
                        fp::Grid g { 3, 3, 3, { .5f, .8f, 1.1f }, { 1.3f, -.7f, 2.1f } };
                        if (axis == 0)
                            g.nx = size;
                        if (axis == 1)
                            g.ny = size;
                        if (axis == 2)
                            g.nz = size;
                        fp::View v;
                        v.source = g.center;
                        v.detector = g.center;
                        float length;
                        if (axis == 0) {
                            v.source.x -= 500;
                            v.detector.x += 500;
                            v.u = { 0, 1, 0 };
                            v.v = { 0, 0, 1 };
                            length = g.nx * g.spacing.x;
                        } else if (axis == 1) {
                            v.source.y -= 500;
                            v.detector.y += 500;
                            v.u = { 1, 0, 0 };
                            v.v = { 0, 0, 1 };
                            length = g.ny * g.spacing.y;
                        } else {
                            v.source.z -= 500;
                            v.detector.z += 500;
                            v.u = { 1, 0, 0 };
                            v.v = { 0, 1, 0 };
                            length = g.nz * g.spacing.z;
                        }
                        std::vector<float> volume(size_t(g.nx) * g.ny * g.nz, .02f);
                        // Border texture quadrature loses the subvoxel edge triangles at ss>1.
                        const float edge = steps == 1 ? 0 : steps == 2 ? .25f
                                                                       : .25f;
                        float spacing = axis == 0 ? g.spacing.x : axis == 1 ? g.spacing.y
                                                                            : g.spacing.z;
                        auto result = run(volume, g, { v }, 1, 1, fp::Method::Joseph, 0, steps);
                        check(std::abs(result[0] - .02f * (length - edge * spacing)) < 2e-5, "Joseph constant-volume analytic path");
                        if (steps == 1) {
                            result = run(volume, g, { v }, 1, 1, fp::Method::Siddon);
                            check(std::abs(result[0] - .02f * length) < 2e-5, "Siddon constant-volume path");
                        }
                    }
        } else if (name == "siddon_oracle" || name == "siddon_texture" || name == "batch") {
            std::mt19937 rng(1729);
            std::uniform_real_distribution<float> random(-1, 1);
            for (int trial = 0; trial < 12; ++trial) {
                fp::Grid g { 5 + trial % 4, 4 + trial % 3, 3 + trial % 5, { .4f + trial * .03f, .7f, 1.1f }, { .7f, -1.2f, .3f } };
                std::vector<float> volume(size_t(g.nx) * g.ny * g.nz);
                for (float& x : volume)
                    x = .02f + .01f * random(rng);
                std::vector<fp::View> views;
                for (int k = 0; k < 7; ++k) {
                    float a = k * .91f + .03f * trial, c = std::cos(a), s = std::sin(a), z = (k - 3) * .7f;
                    views.push_back({ { -40 * c, -40 * s, z }, { 30 * c, 30 * s, z + .4f }, { -s * .6f, c * .6f, .02f }, { .03f, -.02f, .6f } });
                }
                const int nu = 9, nv = 7;
                if (name == "siddon_oracle" || name == "siddon_texture") {
                    const auto memory = name == "siddon_texture" ? fp::SiddonMemory::Texture : fp::SiddonMemory::Linear;
                    auto got = run(volume, g, views, nu, nv, fp::Method::Siddon, 0, 1, fp::DetectorAnchor::Center, memory);
                    for (int k = 0; k < 7; ++k)
                        for (int y = 0; y < nv; ++y)
                            for (int x = 0; x < nu; ++x)
                                check(std::abs(got[(k * nv + y) * nu + x] - oracle(volume, g, views[k], nu, nv, x, y)) < 2e-5, "Siddon independent plane-intersection oracle");
                    if (name == "siddon_texture") {
                        const auto linear = run(volume, g, views, nu, nv, fp::Method::Siddon);
                        const auto batched = run(volume, g, views, nu, nv, fp::Method::Siddon, 2, 1, fp::DetectorAnchor::Center, memory);
                        check(got == linear && got == batched, "Siddon texture/linear/batch mismatch");
                    }
                } else
                    for (auto method : { fp::Method::Siddon, fp::Method::Joseph }) {
                        auto full = run(volume, g, views, nu, nv, method), batch = run(volume, g, views, nu, nv, method, 2);
                        for (size_t i = 0; i < full.size(); ++i)
                            check(std::abs(full[i] - batch[i]) < 1e-7, "FP batch differs from full views");
                    }
            }
#ifdef SCALAR_FP_TEST_DLL
        } else if (name == "dll") {
            for (bool helical : { false, true })
                for (auto method : { fp::Method::Siddon, fp::Method::Joseph })
                    for (int trial = 0; trial < 3; ++trial) {
                        YK::SSystemSpec sys;
                        auto& s = sys.geometry;
                        s.volume = {};
                        s.volume.nx = 11;
                        s.volume.ny = 9;
                        s.volume.nz = 7;
                        s.volume.voxel_x_mm = .6f;
                        s.volume.voxel_y_mm = .8f;
                        s.volume.voxel_z_mm = 1.1f;
                        s.volume.center_mm = make_float3(.7f, -1.1f, .3f);
                        s.flat_detector.channels = 13;
                        s.flat_detector.rows = 9;
                        s.flat_detector.channel_size_mm = .7f;
                        s.flat_detector.row_size_mm = .9f;
                        s.flat_detector.pose.tilt_u_rad = .03f * trial;
                        s.flat_detector.pose.tilt_v_rad = -.04f * trial;
                        s.flat_detector.pose.offset_unv_mm = make_float3(.2f * trial, -.1f * trial, 0);
                        s.circular.total_views = 11;
                        s.circular.views_per_turn = 11;
                        s.circular.sid_mm = 60;
                        s.circular.sdd_mm = 100;
                        s.circular.start_angle_rad = .13f;
                        s.circular.source_offset_mm = make_float3(.1f * trial, 0, .2f);
                        if (helical) {
                            s.trajectory = YK::ETrajectoryKind::Helical;
                            s.helical.total_views = 11;
                            s.helical.views_per_turn = 11;
                            s.helical.sid_mm = 60;
                            s.helical.sdd_mm = 100;
                            s.helical.start_angle_rad = .13f;
                            s.helical.pitch_mm_per_turn = 4;
                            s.helical.start_z_mm = 2;
                            s.helical.source_offset_mm = s.circular.source_offset_mm;
                        }
                        sys.reconstruction.pipeline = YK::EPipeline::ForwardProjection;
                        sys.reconstruction.forward_projector = method == fp::Method::Siddon ? YK::EForwardProjector::Siddon : YK::EForwardProjector::Joseph;
                        std::vector<YK::SConeProjGeomVec> flat;
                        std::vector<YK::SCylConeProjGeomVec> cyl;
                        YK::SVolGeom vg;
                        check(YK::buildSystemGeometry(s, flat, cyl, vg), "DLL geometry failed");
                        fp::Grid grid { vg.Nx, vg.Ny, vg.Nz, { vg.vox_x, vg.vox_y, vg.vox_z }, vg.center };
                        std::vector<fp::View> views;
                        for (const auto& v : flat) {
                            fp::View a;
                            a.source = { v.src.x, v.src.y, v.src.z };
                            a.u = { v.detU.x, v.detU.y, v.detU.z };
                            a.v = { v.detV.x, v.detV.y, v.detV.z };
                            a.detector = { v.detS.x, v.detS.y, v.detS.z };
                            views.push_back(a);
                        }
                        std::vector<float> volume(11 * 9 * 7);
                        for (size_t i = 0; i < volume.size(); ++i)
                            volume[i] = .02f + .01f * std::sin(float(i) * .3f);
                        std::vector<float> expected(11 * 13 * 9);
                        std::unique_ptr<YK::IReconstructionSession, void (*)(YK::IReconstructionSession*)> session(YK::ReconstructionSessionFactory::create(), YK::ReconstructionSessionFactory::destroy);
                        check(bool(session), "DLL session missing");
                        if (!session->initialize(sys))
                            throw std::runtime_error(session->lastErrorMessage());
                        YK::SExecutionRequest request;
                        request.volume = { volume.data(), YK::EMemoryLocation::Host, volume.size() };
                        request.projection = { expected.data(), YK::EMemoryLocation::Host, expected.size() };
                        request.view_count = 11;
                        if (!session->execute(request))
                            throw std::runtime_error(session->lastErrorMessage());
                        auto actual = run(volume, grid, views, 13, 9, method, 0, 1, fp::DetectorAnchor::Pixel00);
                        double peak = 0;
                        for (size_t i = 0; i < actual.size(); ++i)
                            peak = std::max(peak, double(std::abs(actual[i] - expected[i])));
                        std::cout << "DLL " << helical << " " << int(method) << " " << trial << " max=" << peak << '\n';
                        check(peak < 2e-5, "copied FP disagrees with DLL");
                    }
#endif
        } else
            throw std::runtime_error("unknown test case");
        std::cout << name << " passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
