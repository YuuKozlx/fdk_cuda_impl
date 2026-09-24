#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "YKCBCT/interface/YkReconstructionApi.hpp"
#include "../Yktest/YkTestImage.hpp" // Host-only BMP writer, no reconstruction code.

namespace {
using Session = std::unique_ptr<YK::IReconstructionSession,
    decltype(&YK::ReconstructionSessionFactory::destroy)>;

Session makeSession()
{
    Session session(YK::ReconstructionSessionFactory::create(),
        &YK::ReconstructionSessionFactory::destroy);
    if (!session) throw std::runtime_error("Cannot create DLL session");
    return session;
}

void require(bool success, YK::IReconstructionSession& session, const char* operation)
{
    if (!success) throw std::runtime_error(std::string(operation) + ": " +
        session.lastErrorMessage());
}

YK::SSystemSpec makeSystem()
{
    YK::SSystemSpec system{};
    system.device = 0;
    auto& g = system.geometry;
    g.detector = YK::EDetectorKind::Flat;
    g.trajectory = YK::ETrajectoryKind::Circular;
    g.circular.total_views = 360;
    g.circular.views_per_turn = 360;
    g.circular.sid_mm = 200.f;
    g.circular.sdd_mm = 400.f;
    g.flat_detector.channels = 128;
    g.flat_detector.rows = 128;
    g.flat_detector.channel_size_mm = 0.8f;
    g.flat_detector.row_size_mm = 0.8f;
    g.volume = {64, 64, 64, 0.75f, 0.75f, 0.75f, {0.f, 0.f, 0.f}};
    system.reconstruction.pipeline = YK::EPipeline::FDK;
    system.reconstruction.fdk.filter = YK::EFdkFilter::RamLak;
    return system;
}

// Modified 3-D Shepp-Logan ellipsoids, matching the ASTRA test phantom.
// ASTRA src/SheppLogan.cpp: imec / University of Antwerp / CWI,
// SPDX-License-Identifier: GPL-3.0-or-later
std::vector<float> makePhantom(const YK::SVolumeGridSpec& grid)
{
    struct Ellipsoid { double x, y, z, a, b, c, degrees, value; };
    const Ellipsoid ellipsoids[] = {
        {0,0,0,.69,.92,.81,0,1}, {0,-.0184,0,.6624,.874,.78,0,-.8},
        {.22,0,0,.11,.31,.22,-18,-.2}, {-.22,0,0,.16,.41,.28,18,-.2},
        {0,.35,0,.21,.25,.41,0,.1}, {0,.1,0,.046,.046,.05,0,.1},
        {0,-.1,0,.046,.046,.05,0,.1}, {-.08,-.605,0,.046,.023,.05,0,.1},
        {0,-.605,0,.023,.023,.02,0,.1}, {.06,-.605,0,.023,.046,.02,0,.1}
    };
    std::vector<float> result(size_t(grid.nx) * grid.ny * grid.nz);
    for (int z = 0; z < grid.nz; ++z)
        for (int y = 0; y < grid.ny; ++y)
            for (int x = 0; x < grid.nx; ++x) {
                const double px = (x + .5 - grid.nx * .5) * grid.voxel_x_mm / 20.;
                const double py = -(y + .5 - grid.ny * .5) * grid.voxel_y_mm / 20.;
                const double pz = -(z + .5 - grid.nz * .5) * grid.voxel_z_mm / 20.;
                double value = 0.;
                for (const auto& e : ellipsoids) {
                    const double angle = e.degrees * 3.141592653589793 / 180.;
                    const double dx = px - e.x, dy = py - e.y, dz = pz - e.z;
                    const double rx = dx * std::cos(angle) - dy * std::sin(angle);
                    const double ry = dx * std::sin(angle) + dy * std::cos(angle);
                    if (rx*rx/(e.a*e.a) + ry*ry/(e.b*e.b) + dz*dz/(e.c*e.c) <= 1.)
                        value += e.value;
                }
                result[(size_t(z) * grid.ny + y) * grid.nx + x] =
                    float(std::max(value, 0.) * .02);
            }
    return result;
}

YK::Buffer host(std::vector<float>& data)
{
    return {data.data(), YK::EMemoryLocation::Host, data.size()};
}
}

int main(int argc, char** argv)
{
    try {
        if (argc > 2) throw std::runtime_error("Usage: dlltest_shepp_logan [output-directory]");
        const auto output = std::filesystem::absolute(argc == 2 ? argv[1] :
            "out/test-artifacts/dll-shepp-logan");
        auto system = makeSystem();
        const auto& grid = system.geometry.volume;
        auto truth = makePhantom(grid);
        const int views = system.geometry.circular.total_views;
        const size_t view_elements = size_t(system.geometry.flat_detector.channels) *
            system.geometry.flat_detector.rows;
        std::vector<float> projection(view_elements * views);
        std::vector<float> reconstruction(truth.size()), streamed(truth.size());

        // Both operators cross the DLL boundary. The caller owns only host arrays.
        auto forward = makeSession();
        auto forward_system = system;
        forward_system.reconstruction.pipeline = YK::EPipeline::ForwardProjection;
        require(forward->initialize(forward_system), *forward, "FP initialize");
        YK::SExecutionRequest request{};
        request.volume = host(truth);
        request.projection = host(projection);
        request.view_count = views;
        const auto started = std::chrono::steady_clock::now();
        require(forward->execute(request), *forward, "FP execute");
        forward.reset();
        const auto projected = std::chrono::steady_clock::now();

        auto fdk = makeSession();
        require(fdk->initialize(system), *fdk, "FDK initialize");
        request.volume = host(reconstruction);
        require(fdk->execute(request), *fdk, "FDK execute");
        const auto reconstructed = std::chrono::steady_clock::now();

        fdk->reset();
        constexpr int batch_views = 47; // Non-divisor also exercises the final short batch.
        for (int offset = 0; offset < views; offset += batch_views) {
            const int count = std::min(batch_views, views - offset);
            request.projection = {projection.data() + size_t(offset) * view_elements,
                YK::EMemoryLocation::Host, size_t(count) * view_elements};
            request.volume = host(streamed);
            request.view_offset = offset;
            request.view_count = count;
            request.clear_output = offset == 0;
            require(fdk->execute(request), *fdk, "FDK streaming execute");
        }

        double energy = 0., output_energy = 0., product = 0., error2 = 0., mae = 0.;
        double max_stream_difference = 0.;
        std::vector<float> error(truth.size());
        for (size_t i = 0; i < truth.size(); ++i) {
            if (!std::isfinite(reconstruction[i]) || !std::isfinite(streamed[i]))
                throw std::runtime_error("Non-finite reconstruction");
            const double t = truth[i], r = reconstruction[i], delta = r - t;
            energy += t*t; output_energy += r*r; product += t*r;
            error2 += delta*delta; mae += std::abs(delta);
            max_stream_difference = std::max(max_stream_difference,
                std::abs(r - streamed[i]));
            error[i] = float(delta);
        }
        const double similarity = product / std::sqrt(std::max(energy * output_energy, 1e-30));
        const double nrmse = std::sqrt(error2 / std::max(energy, 1e-30));
        mae /= truth.size();
        const bool passed = similarity > .85 && nrmse < .65 && max_stream_difference < 1e-5;
        std::filesystem::create_directories(output);
        if (!YK::TestImage::writeGrayMontageBmp(output / "comparison.bmp",
            {{&truth, grid.nx, grid.ny, grid.nz, grid.nz/2, 1.f, 0.f, .02f, false},
             {&reconstruction, grid.nx, grid.ny, grid.nz, grid.nz/2, 1.f, 0.f, .02f, false},
             {&error, grid.nx, grid.ny, grid.nz, grid.nz/2, 1.f, 0.f, .02f, true}},
            3, 4, 4)) throw std::runtime_error("Cannot write comparison image");
        std::ofstream metrics(output / "metrics.txt");
        metrics << "Flat circular DLL FP -> DLL FDK; modified 3-D Shepp-Logan\n"
            << "volume=64x64x64, voxel=0.75mm, detector=128x128, pixel=0.8mm, views=360\n"
            << "SID=200mm, SDD=400mm, filter=RamLak\n"
            << "cosine_similarity=" << similarity << "\nNRMSE=" << nrmse
            << "\nMAE=" << mae << "\nstreaming_max_abs=" << max_stream_difference
            << "\nFP_ms=" << std::chrono::duration<double,std::milli>(projected-started).count()
            << "\nFDK_ms=" << std::chrono::duration<double,std::milli>(reconstructed-projected).count()
            << "\nImage left to right: truth, FDK, absolute error; common window [0,0.02]\n"
            << (passed ? "PASS" : "FAIL") << '\n';
        metrics.close();
        if (!metrics) throw std::runtime_error("Cannot write metrics");
        std::cout << "DLL Shepp-Logan: cosine=" << similarity << " NRMSE=" << nrmse
            << " MAE=" << mae << " streaming_max_abs=" << max_stream_difference
            << ' ' << (passed ? "PASS" : "FAIL") << "\nArtifacts: " << output << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "DLL Shepp-Logan test failed: " << e.what() << '\n';
        return 1;
    }
}
