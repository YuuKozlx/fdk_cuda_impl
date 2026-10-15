#include "primary/JosephScalar.cuh"
#include "recon/postprocess/stages/correction/EmpiricalBoneCorrection.hpp"
#include <cmath>
#include <iostream>
using namespace yk::spectral;
namespace pp = yk::spectral::postprocess;
void require(bool b, const char* m)
{
    if (!b)
        throw std::runtime_error(m);
}
void check(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
__global__ void rayTest(const float* v, float* out) { out[0] = primary_detail::josephScalar(v, 8, 8, 8, make_float3(1, 1, 1), make_float3(-10, 0, 0), make_float3(1, 0, 0)); }
int main()
{
    try {
        std::vector<float> cube(512, .02f);
        auto ray = [&](float3 s, float3 d) { return primary_detail::josephScalar(cube.data(), 8, 8, 8, make_float3(1, 1, 1), s, d); };
        require(std::abs(ray(make_float3(-10, 0, 0), make_float3(1, 0, 0)) - .16) < 1e-6, "Joseph axis length");
        require(std::abs(ray(make_float3(10, 0, 0), make_float3(-1, 0, 0)) - .16) < 1e-6, "Joseph reversed ray");
        const float dx = 1 / std::sqrt(1.01f);
        require(std::abs(ray(make_float3(-10, -1, 0), make_float3(dx, .1f * dx, 0)) - .16 / dx) < 1e-6, "Joseph oblique path");
        require(ray(make_float3(-10, 10, 0), make_float3(1, 0, 0)) == 0, "Joseph slab miss");
        float *dc = nullptr, *dr = nullptr;
        check(cudaMalloc(&dc, 512 * 4));
        check(cudaMalloc(&dr, 4));
        check(cudaMemcpy(dc, cube.data(), 512 * 4, cudaMemcpyHostToDevice));
        rayTest<<<1, 1>>>(dc, dr);
        float out;
        check(cudaMemcpy(&out, dr, 4, cudaMemcpyDeviceToHost));
        require(std::abs(out - .16) < 1e-6, "Joseph CPU GPU parity");
        check(cudaFree(dc));
        check(cudaFree(dr));
        pp::Grid g;
        g.dimensions = { 16, 16, 16 };
        g.spacing_mm = { .5, .7, 1.2 };
        const size_t n = 4096;
        std::array<std::shared_ptr<std::vector<float>>, 3> data;
        pp::EbhcBasis b;
        for (int k = 0; k < 3; ++k) {
            data[k] = std::make_shared<std::vector<float>>(n);
            b[k] = data[k];
        }
        std::vector<float> f(n);
        for (int z = 0; z < 16; ++z)
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x) {
                    size_t i = (z * 16 + y) * 16 + x;
                    (*data[0])[i] = .02f * std::sin(x * .2f);
                    (*data[1])[i] = .015f * std::cos(y * .3f);
                    (*data[2])[i] = .01f * std::sin(z * .4f);
                    f[i] = .02f - .3f * (*data[0])[i] + .04f * (*data[1])[i] - .02f * (*data[2])[i];
                }
        double expected = 0;
        for (size_t i = 0; i < n; ++i) {
            double x = i % 16 ? (f[i] - f[i - 1]) / .5 : 0, y = (i / 16) % 16 ? (f[i] - f[i - 16]) / .7 : 0, z = i / 256 ? (f[i] - f[i - 256]) / 1.2 : 0;
            expected += std::sqrt(x * x + y * y + z * z);
        }
        expected /= n;
        auto fit = pp::fitEbhcFullVolumeTv(f, b, g);
        require(fit.converged, "TV optimizer did not converge");
        require(std::abs(fit.initial_tv - expected) < 1e-9, "CPU CUDA full-volume TV mismatch");
        require(std::abs(fit.coefficients.linear - .3) < 1e-4 && std::abs(fit.coefficients.cross + .04) < 1e-4 && std::abs(fit.coefficients.square - .02) < 1e-4, "Three coefficient recovery");
        float* d = nullptr;
        check(cudaMalloc(&d, n * 4));
        pp::Context context;
        context.grid = g;
        auto stage = pp::makeEbhcVolumeStage(fit.coefficients, b, g);
        pipeline::validatePluginInfo(stage->info());
        for (int repeat = 0; repeat < 2; ++repeat) {
            check(cudaMemcpy(d, f.data(), n * 4, cudaMemcpyHostToDevice));
            stage->prepare(context);
            pp::Volume v;
            v.grid = g;
            v.elements = n;
            v.device_data = d;
            v.location = pp::Location::Device;
            stage->process(v, context);
            std::vector<float> got(n);
            check(cudaMemcpy(got.data(), d, n * 4, cudaMemcpyDeviceToHost));
            for (float x : got)
                require(std::abs(x - .02) < 1e-6, "Stage combination mismatch");
            stage->reset();
        }
        check(cudaFree(d));
        std::cout << "Joseph geometry, CUDA TV, three-term fit, Stage lifecycle passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
