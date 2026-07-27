#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "Iter/YkAlgebraicReconstructor.hpp"
#include "Iter/YkCGLS.hpp"
#include "Iter/YkCglsReconstructor.hpp"
#include "Iter/YkOSSART.hpp"
#include "Iter/YkSART.hpp"
#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"
#include "util/YkCudaTimer.hpp"

namespace {

using namespace YK;

SCBCTParams makeIterativeParams()
{
    SCBCTParams p{};
    p.iPU = 28; p.iPV = 20;
    p.iPAng = 8; p.iPAngTotal = 8;
    p.iVX = 16; p.iVY = 16; p.iVZ = 12;
    p.du_mm = 1.f; p.dv_mm = 1.f;
    p.vox_x_mm = 1.f; p.vox_y_mm = 1.f; p.vox_z_mm = 1.f;
    p.SID = 80.f; p.SDD = 160.f;
    p.scan_range_rad = 2.f * CUDA_PI;
    p.angle_list.resize(p.iPAng);
    for (int i = 0; i < p.iPAng; ++i)
        p.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / p.iPAng;
    return p;
}

bool cudaOk(cudaError_t status, const char* action)
{
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(status));
    return false;
}

bool validVolume(const std::vector<float>& volume)
{
    bool nonzero = false;
    for (float value : volume) {
        if (!std::isfinite(value)) return false;
        nonzero = nonzero || std::fabs(value) > 1e-8f;
    }
    return nonzero;
}

// Common test fixture.  Reconstruction classes only receive caller-owned
// device pointers; measurement is generated through the same public operator
// contract that production callers use, never through a retired runner.
class IterativeFixture {
public:
    bool prepare()
    {
        p = makeIterativeParams();
        volume_n = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
        sino_n = static_cast<size_t>(p.iPAng) * p.iPU * p.iPV;
        const auto h_truth = TestPhantom::makeCatphanLike(p);
        if (!cudaOk(cudaStreamCreate(&stream), "create stream")) return false;
        d_truth = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
        d_sino = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
        d_recon = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
        if (!cudaOk(cudaMemcpyAsync(d_truth.data(), h_truth.data(), volume_n * sizeof(float),
            cudaMemcpyHostToDevice, stream), "upload phantom"))
            return false;

        if (!geometry.initialize(p)) return false;
        resources.attach(stream, 0);
        auto fp = makeForwardOperator(ETask::FP_Joseph);
        const bool ok = fp->prepare(geometry, resources) &&
            fp->apply(d_truth.data(), p, d_sino.data(), resources) &&
            cudaOk(cudaMemsetAsync(d_recon.data(), 0, volume_n * sizeof(float), stream), "clear recon") &&
            cudaOk(cudaStreamSynchronize(stream), "generate measurement");
        fp->release();
        detail::buildCircularViews(p, external_geometry);
        return ok;
    }

    bool finish(const char* name, bool run_ok)
    {
        std::vector<float> output(volume_n);
        run_ok = run_ok && cudaOk(cudaStreamSynchronize(stream), "reconstructor synchronize") &&
            cudaOk(cudaMemcpy(output.data(), d_recon.data(), volume_n * sizeof(float),
                cudaMemcpyDeviceToHost), "download reconstruction") && validVolume(output);
        std::printf("%s: %s\n", name, run_ok ? "PASS" : "FAIL");
        return run_ok;
    }

    ~IterativeFixture()
    {
        if (stream) cudaStreamDestroy(stream);
    }

    SCBCTParams p{};
    std::vector<SConeProjGeomVec> external_geometry{};
    cudaStream_t stream = nullptr;
    Mem::DeviceLinearBuffer3D<float> d_truth{};
    Mem::DeviceLinearBuffer3D<float> d_sino{};
    Mem::DeviceLinearBuffer3D<float> d_recon{};

private:
    GeometryContext geometry{};
    ResourceContext resources{};
    Mem::MemoryController memory{};
    size_t volume_n = 0;
    size_t sino_n = 0;
};

template <typename Run>
int executeReconstructorSmoke(const char* name, Run&& run)
{
    IterativeFixture fixture;
    if (!fixture.prepare()) {
        std::printf("%s: fixture setup FAIL\n", name);
        return 1;
    }
    Util::CudaEventTimer timer;
    timer.start(fixture.stream);
    const bool run_ok = run(fixture);
    timer.stop(fixture.stream);
    const float elapsed_ms = run_ok ? timer.elapsed_ms() : NAN;
    std::printf("%s GPU time: %.3f ms\n", name, elapsed_ms);
    return fixture.finish(name, run_ok) ? 0 : 1;
}

} // namespace

int main_sart_smoke()
{
    return executeReconstructorSmoke("SART", [](IterativeFixture& f) {
        SART recon;
        SART::Config cfg{};
        cfg.n_iter = 1; cfg.lambda = 0.2f; cfg.use_min = true;
        cfg.fp_task = ETask::FP_Joseph; cfg.bp_task = ETask::BP_Joseph_v2;
        const bool ok = recon.init(f.p, cfg, f.stream) && recon.run(f.d_sino.data(), f.d_recon.data(), f.p, f.stream);
        recon.release();
        return ok;
    });
}

int main_sirt_smoke()
{
    return executeReconstructorSmoke("SIRT", [](IterativeFixture& f) {
        SIRT recon;
        SIRT::Config cfg{};
        cfg.n_iter = 1; cfg.n_batch = 2; cfg.lambda = 0.2f; cfg.use_min = true;
        cfg.fp_task = ETask::FP_Joseph; cfg.bp_task = ETask::BP_Joseph_v3;
        const bool ok = recon.init(f.p, cfg, f.stream) && recon.run(f.d_sino.data(), f.d_recon.data(), f.p, f.stream);
        recon.release();
        return ok;
    });
}

int main_ossart_tigre_smoke()
{
    return executeReconstructorSmoke("OSSART_TIGRE", [](IterativeFixture& f) {
        OSSART_TIGRE recon;
        OSSART_TIGRE::Config cfg{};
        cfg.n_iter = 1; cfg.n_subset = 2; cfg.lambda = 0.2f; cfg.use_min = true;
        cfg.fp_task = ETask::FP_Joseph; cfg.bp_task = ETask::BP_Joseph_v2;
        const bool ok = recon.init(f.p, cfg, f.stream) && recon.run(f.d_sino.data(), f.d_recon.data(), f.p, f.stream);
        recon.release();
        return ok;
    });
}

int main_ossart_smoke()
{
    return executeReconstructorSmoke("OSSART", [](IterativeFixture& f) {
        OSSART recon;
        OSSART::Config cfg{};
        cfg.n_iter = 1; cfg.n_subset = 2; cfg.lambda = 0.2f; cfg.use_min = true;
        cfg.fp_task = ETask::FP_Joseph; cfg.bp_task = ETask::BP_Joseph_v2;
        const bool ok = recon.init(f.p, cfg, f.stream) && recon.run(f.d_sino.data(), f.d_recon.data(), f.p, f.stream);
        recon.release();
        return ok;
    });
}

int main_ossart_ex_smoke()
{
    return executeReconstructorSmoke("OSSARTEx", [](IterativeFixture& f) {
        OSSARTEx recon;
        OSSARTEx::Config cfg{};
        cfg.n_iter = 1; cfg.n_subset = 2; cfg.lambda = 0.2f; cfg.use_min = true;
        cfg.fp_task = ETask::FP_Joseph; cfg.bp_task = ETask::BP_Joseph_v2;
        const bool ok = recon.init(f.p, cfg, f.external_geometry, f.stream) &&
            recon.run(f.d_sino.data(), f.d_recon.data(), f.stream);
        recon.release();
        return ok;
    });
}

int main_algebraic_ex_smoke()
{
    const Iter::EAlgebraicMethod methods[] = {
        Iter::EAlgebraicMethod::Sirt,
        Iter::EAlgebraicMethod::Sart,
        Iter::EAlgebraicMethod::Ossart,
        Iter::EAlgebraicMethod::Ossart
    };
    const Iter::EAlgebraicWeightModel weights[] = {
        Iter::EAlgebraicWeightModel::DetailedSubset,
        Iter::EAlgebraicWeightModel::DetailedSubset,
        Iter::EAlgebraicWeightModel::DetailedSubset,
        Iter::EAlgebraicWeightModel::TigreApprox
    };
    const char* names[] = {
        "SIRT-Ex-Detailed", "SART-Ex-Detailed",
        "OSSART-Ex-Detailed", "OSSART-Ex-TIGRE"
    };
    for (int i = 0; i < 4; ++i) {
        const int result = executeReconstructorSmoke(names[i], [&, i](IterativeFixture& f) {
            Iter::AlgebraicReconstructionConfig cfg{};
            cfg.method = methods[i];
            cfg.weight_model = weights[i];
            cfg.iterations = 1;
            cfg.subset_count = 2;
            cfg.relaxation = 0.2f;
            cfg.use_min = true;
            cfg.fp_task = ETask::FP_Joseph;
            cfg.bp_task = ETask::BP_Joseph_v3;
            Iter::AlgebraicReconstructorEx recon;
            return recon.prepare(f.p, f.external_geometry, cfg, f.stream) &&
                recon.reconstruct(f.d_sino.data(), f.d_recon.data());
        });
        if (result != 0) return result;
    }
    return 0;
}

int main_algebraic_smoke()
{
    return executeReconstructorSmoke("AlgebraicReconstructor", [](IterativeFixture& f) {
        Iter::AlgebraicReconstructor::Config cfg{};
        cfg.method = Iter::EAlgebraicMethod::Ossart;
        cfg.iterations = 1;
        cfg.subset_count = 2;
        cfg.relaxation = 0.2f;
        cfg.use_min = true;
        Iter::AlgebraicReconstructor recon;
        return recon.prepare(f.p, cfg, f.stream) &&
            recon.reconstruct(f.d_sino.data(), f.d_recon.data());
    });
}

int main_ossart_tv_smoke()
{
    return executeReconstructorSmoke("OS-SART-Smoothed-TV", [](IterativeFixture& f) {
        Iter::AlgebraicReconstructorEx::Config cfg{};
        cfg.method = Iter::EAlgebraicMethod::Ossart;
        cfg.weight_model = Iter::EAlgebraicWeightModel::DetailedSubset;
        cfg.iterations = 2;
        cfg.subset_count = 2;
        cfg.relaxation = 0.2f;
        cfg.use_min = true;
        cfg.min_constraint = 0.f;
        cfg.regularization.type = Iter::EAlgebraicRegularizer::SmoothedTv;
        cfg.regularization.tv_dimensionality = Iter::ETvDimensionality::Volume3D;
        cfg.regularization.strength = 1e-3f;
        cfg.regularization.inner_iterations = 2;
        cfg.regularization.epsilon = 1e-4f;
        cfg.regularization.strength_reduction = 0.95f;

        Iter::AlgebraicReconstructorEx recon;
        return recon.prepare(f.p, f.external_geometry, cfg, f.stream) &&
            recon.actualSubsetCount() == 2 &&
            recon.reconstruct(f.d_sino.data(), f.d_recon.data()) &&
            recon.totalSubsetUpdates() == 4;
    });
}

int main_cgls_smoke()
{
    return executeReconstructorSmoke("CGLS", [](IterativeFixture& f) {
        CGLS recon;
        CGLS::Config cfg{};
        cfg.n_iter = 1; cfg.use_min = true;
        cfg.fp_task = ETask::FP_Joseph; cfg.bp_task = ETask::BP_Joseph_v2;
        const bool ok = recon.init(f.p, cfg, f.stream) && recon.run(f.d_sino.data(), f.d_recon.data(), f.p, f.stream);
        recon.release();
        return ok;
    });
}

int main_cgls_astra_smoke()
{
    return executeReconstructorSmoke("CGLSAstra", [](IterativeFixture& f) {
        CGLSAstra recon;
        CGLSAstra::Config cfg{};
        cfg.n_iter = 1; cfg.use_min = true;
        cfg.fp_task = ETask::FP_Joseph; cfg.bp_task = ETask::BP_Joseph_v2;
        const bool ok = recon.init(f.p, cfg, f.stream) && recon.run(f.d_sino.data(), f.d_recon.data(), f.p, f.stream);
        recon.release();
        return ok;
    });
}

int main_cgls_ex_smoke()
{
    return executeReconstructorSmoke("CGLSEx", [](IterativeFixture& f) {
        CGLSEx recon;
        CGLSEx::Config cfg{};
        cfg.n_iter = 1; cfg.use_min = true;
        cfg.fp_task = ETask::FP_Joseph; cfg.bp_task = ETask::BP_Joseph_v2;
        const bool ok = recon.init(f.p, cfg, f.external_geometry, f.stream) &&
            recon.run(f.d_sino.data(), f.d_recon.data(), f.p, f.stream);
        recon.release();
        return ok;
    });
}

int main_cgls_unified_smoke()
{
    const Iter::ECglsStrategy strategies[] = {
        Iter::ECglsStrategy::RobustRestart,
        Iter::ECglsStrategy::AstraClassic
    };
    const char* names[] = { "CGLS-Robust", "CGLS-Astra" };
    for (int i = 0; i < 2; ++i) {
        int result = executeReconstructorSmoke(names[i], [&, i](IterativeFixture& f) {
            Iter::CglsReconstructor::Config cfg{};
            cfg.strategy = strategies[i];
            cfg.iterations = 1;
            cfg.use_min = true;
            Iter::CglsReconstructor recon;
            return recon.prepare(f.p, cfg, f.stream) &&
                recon.reconstruct(f.d_sino.data(), f.d_recon.data());
        });
        if (result) return result;
        result = executeReconstructorSmoke(
            i == 0 ? "CGLS-Ex-Robust" : "CGLS-Ex-Astra",
            [&, i](IterativeFixture& f) {
                Iter::CglsReconstructorEx::Config cfg{};
                cfg.strategy = strategies[i];
                cfg.iterations = 1;
                cfg.use_min = true;
                Iter::CglsReconstructorEx recon;
                return recon.prepare(f.p, f.external_geometry, cfg, f.stream) &&
                    recon.reconstruct(f.d_sino.data(), f.d_recon.data());
            });
        if (result) return result;
    }
    return 0;
}
