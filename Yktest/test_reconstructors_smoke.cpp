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
#include "Iter/YkTigreGradientReconstructor.hpp"
#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"
#include "util/YkCudaTimer.hpp"

namespace {

using namespace YK;

SCBCTParams makeIterativeParams()
{
    SCBCTParams p{};
    // 32x24 同时覆盖迭代算子和 FDK 初始化路径；FDK 的纹理/FFT 工作区
    // 使用这个已验证的最小探测器尺寸。
    p.iPU = 32; p.iPV = 24;
    p.iPAng = 8; p.iPAngTotal = 8;
    p.iVX = 16; p.iVY = 16; p.iVZ = 12;
    p.du_mm = 1.f; p.dv_mm = 1.f;
    p.vox_x_mm = 1.f; p.vox_y_mm = 1.f; p.vox_z_mm = 1.f;
    p.SID = 80.f; p.SDD = 160.f;
    p.scan_range_rad = 2.f * CUDA_PI;
    p.scan_start_angle_rad = 0.f;
    p.nDirSign = 1;
    p.bShortScan = false;
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

int main_iterative_convergence_smoke()
{
    int result = executeReconstructorSmoke("OSSART-convergence", [](IterativeFixture& f) {
        Iter::AlgebraicReconstructionConfig cfg{};
        cfg.method = Iter::EAlgebraicMethod::Ossart;
        cfg.weight_model = Iter::EAlgebraicWeightModel::DetailedSubset;
        cfg.iterations = 4;
        cfg.subset_count = 2;
        cfg.relaxation = 0.2f;
        cfg.use_min = true;
        // 阈值大于 1，保证首轮检查即满足；验证提前停止的数据流与统计。
        cfg.convergence.relative_residual_tolerance = 2.f;
        cfg.convergence.minimum_iterations = 1;
        cfg.convergence.check_interval = 1;
        cfg.convergence.patience = 1;
        cfg.fp_task = ETask::FP_Joseph;
        cfg.bp_task = ETask::BP_Joseph_v3;
        Iter::AlgebraicReconstructorEx recon;
        const bool ok = recon.prepare(f.p, f.external_geometry, cfg, f.stream) &&
            recon.reconstruct(f.d_sino.data(), f.d_recon.data());
        const auto stats = recon.convergenceStatistics();
        return ok && stats.completed_iterations == 1 &&
            recon.totalSubsetUpdates() == 2 && stats.convergence_checks == 1 &&
            stats.stopped_by_relative_residual &&
            std::isfinite(stats.relative_projection_residual);
    });
    if (result) return result;

    return executeReconstructorSmoke("CGLS-convergence", [](IterativeFixture& f) {
        Iter::CglsReconstructionConfig cfg{};
        cfg.strategy = Iter::ECglsStrategy::RobustRestart;
        cfg.iterations = 4;
        cfg.use_min = true;
        cfg.convergence.relative_residual_tolerance = 2.f;
        cfg.convergence.minimum_iterations = 1;
        cfg.convergence.check_interval = 1;
        cfg.convergence.patience = 1;
        cfg.fp_task = ETask::FP_Joseph;
        cfg.bp_task = ETask::BP_Joseph_v3;
        Iter::CglsReconstructorEx recon;
        const bool ok = recon.prepare(f.p, f.external_geometry, cfg, f.stream) &&
            recon.reconstruct(f.d_sino.data(), f.d_recon.data());
        const auto stats = recon.convergenceStatistics();
        return ok && stats.completed_iterations == 1 &&
            stats.convergence_checks == 1 && stats.stopped_by_relative_residual &&
            std::isfinite(stats.relative_projection_residual);
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

int main_tigre_gradient_family_smoke()
{
    using Algorithm = Iter::ETigreGradientAlgorithm;
    struct Case { Algorithm algorithm; const char* name; int block_size; };
    const Case cases[] = {
        { Algorithm::Sart, "TIGRE-SART", 1 },
        { Algorithm::OsSart, "TIGRE-OS-SART", 4 },
        { Algorithm::Sirt, "TIGRE-SIRT", 8 },
        { Algorithm::AsdPocs, "TIGRE-ASD-POCS", 1 },
        { Algorithm::OsAsdPocs, "TIGRE-OS-ASD-POCS", 4 },
        { Algorithm::BAsdPocsBeta, "TIGRE-B-ASD-POCS-beta", 1 },
        { Algorithm::Pcsd, "TIGRE-PCSD", 1 },
        { Algorithm::OsPcsd, "TIGRE-OS-PCSD", 4 },
        { Algorithm::AwPcsd, "TIGRE-AwPCSD", 1 },
        { Algorithm::OsAwPcsd, "TIGRE-OS-AwPCSD", 4 },
        { Algorithm::AwAsdPocs, "TIGRE-Aw-ASD-POCS", 1 },
        { Algorithm::OsAwAsdPocs, "TIGRE-OS-Aw-ASD-POCS", 4 }
    };
    for (const auto& item : cases) {
        const int result = executeReconstructorSmoke(item.name,
            [&](IterativeFixture& f) {
                Iter::TigreGradientReconstructor::Config config{};
                config.algorithm = item.algorithm;
                config.iterations = 2;
                config.block_size = item.block_size;
                config.lambda = 0.2f;
                config.lambda_reduction = 0.99f;
                config.initialization = Iter::ETigreInitialization::Zero;
                config.tv_iterations = 2;
                config.alpha = 0.002f;
                config.max_l2_error = 0.f;
                config.bregman_interval = 1;
                config.fp_task = ETask::FP_Joseph;
                config.bp_task = ETask::BP_Joseph_v3;
                Iter::TigreGradientReconstructor reconstructor;
                const bool ok = reconstructor.prepare(
                    f.p, config, f.stream) &&
                    reconstructor.reconstruct(
                        f.d_sino.data(), f.d_recon.data());
                const auto statistics = reconstructor.statistics();
                const bool finite_pocs_statistics =
                    item.algorithm == Algorithm::Sart ||
                    item.algorithm == Algorithm::OsSart ||
                    item.algorithm == Algorithm::Sirt ||
                    (std::isfinite(statistics.projection_l2) &&
                        std::isfinite(statistics.data_update_l2) &&
                        std::isfinite(statistics.regularization_update_l2) &&
                        std::isfinite(statistics.direction_cosine) &&
                        std::isfinite(statistics.tv_step));
                return ok && statistics.completed_iterations >= 1 &&
                    statistics.subset_updates >=
                        static_cast<unsigned int>(reconstructor.actualSubsetCount()) &&
                    std::isfinite(statistics.beta) &&
                    finite_pocs_statistics;
            });
        if (result != 0) return result;
    }

    // MATLAB SART/SIRT/OS-SART 的 lambda='nesterov' 分支单独验证。
    int result = executeReconstructorSmoke("TIGRE-OS-SART-Nesterov",
        [](IterativeFixture& f) {
            Iter::TigreGradientReconstructor::Config config{};
            config.algorithm = Algorithm::OsSart;
            config.iterations = 2;
            config.block_size = 4;
            config.relaxation_mode = Iter::ETigreRelaxationMode::Nesterov;
            config.initialization = Iter::ETigreInitialization::Zero;
            config.fp_task = ETask::FP_Joseph;
            config.bp_task = ETask::BP_Joseph_v3;
            Iter::TigreGradientReconstructor reconstructor;
            return reconstructor.prepare(f.p, config, f.stream) &&
                reconstructor.reconstruct(f.d_sino.data(), f.d_recon.data()) &&
                reconstructor.statistics().completed_iterations >= 1;
        });
    if (result != 0) return result;

    // Ex 只改变 geometry 的输入方式，不应绑定或削减算法能力。
    result = executeReconstructorSmoke("TIGRE-Ex-OS-ASD-POCS",
        [](IterativeFixture& f) {
            Iter::TigreGradientReconstructorEx::Config config{};
            config.algorithm = Algorithm::OsAsdPocs;
            config.iterations = 2;
            config.block_size = 4;
            config.lambda = 0.2f;
            config.tv_iterations = 2;
            config.max_l2_error = 0.f;
            config.fp_task = ETask::FP_Joseph;
            config.bp_task = ETask::BP_Joseph_v3;
            Iter::TigreGradientReconstructorEx reconstructor;
            return reconstructor.prepare(f.p, f.external_geometry, config, f.stream) &&
                reconstructor.reconstruct(f.d_sino.data(), f.d_recon.data()) &&
                reconstructor.statistics().subset_updates == 4;
        });
    if (result != 0) return result;

    result = executeReconstructorSmoke("TIGRE-FDK-initialization",
        [](IterativeFixture& f) {
            Iter::TigreGradientReconstructor::Config config{};
            config.algorithm = Algorithm::OsSart;
            config.iterations = 1;
            config.block_size = 4;
            config.lambda = 0.05f;
            config.initialization = Iter::ETigreInitialization::Fdk;
            config.fp_task = ETask::FP_Joseph;
            config.bp_task = ETask::BP_Joseph_v3;
            Iter::TigreGradientReconstructor reconstructor;
            return reconstructor.prepare(f.p, config, f.stream) &&
                reconstructor.reconstruct(f.d_sino.data(), f.d_recon.data());
        });
    if (result != 0) return result;

    result = executeReconstructorSmoke("TIGRE-device-volume-initialization",
        [](IterativeFixture& f) {
            Iter::TigreGradientReconstructor::Config config{};
            config.algorithm = Algorithm::Sirt;
            config.iterations = 1;
            config.lambda = 0.01f;
            config.initialization = Iter::ETigreInitialization::DeviceVolume;
            config.d_initial_volume = f.d_truth.data();
            config.fp_task = ETask::FP_Joseph;
            config.bp_task = ETask::BP_Joseph_v3;
            Iter::TigreGradientReconstructor reconstructor;
            return reconstructor.prepare(f.p, config, f.stream) &&
                reconstructor.reconstruct(f.d_sino.data(), f.d_recon.data());
        });
    if (result != 0) return result;

    // B-ASD-POCS-beta 只更新内部工作投影，调用者传入的测量必须保持只读。
    return executeReconstructorSmoke("TIGRE-Bregman-projection-readonly",
        [](IterativeFixture& f) {
            const size_t count = static_cast<size_t>(f.p.iPAng) * f.p.iPU * f.p.iPV;
            std::vector<float> before(count);
            std::vector<float> after(count);
            if (!cudaOk(cudaMemcpyAsync(before.data(), f.d_sino.data(),
                    count * sizeof(float), cudaMemcpyDeviceToHost, f.stream),
                    "download projection before Bregman"))
                return false;
            Iter::TigreGradientReconstructor::Config config{};
            config.algorithm = Algorithm::BAsdPocsBeta;
            config.iterations = 2;
            config.lambda = 0.2f;
            config.tv_iterations = 2;
            config.max_l2_error = 0.f;
            config.bregman_interval = 1;
            config.fp_task = ETask::FP_Joseph;
            config.bp_task = ETask::BP_Joseph_v3;
            Iter::TigreGradientReconstructor reconstructor;
            if (!reconstructor.prepare(f.p, config, f.stream) ||
                !reconstructor.reconstruct(f.d_sino.data(), f.d_recon.data()))
                return false;
            if (!cudaOk(cudaMemcpyAsync(after.data(), f.d_sino.data(),
                    count * sizeof(float), cudaMemcpyDeviceToHost, f.stream),
                    "download projection after Bregman") ||
                !cudaOk(cudaStreamSynchronize(f.stream), "compare Bregman projection"))
                return false;
            return before == after;
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
