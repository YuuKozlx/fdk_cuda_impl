#include "LibraryGeometry.hpp"
#include "Reconstruction/Iterative/Statistical/StatisticalContracts.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace YK::Iter;
using namespace YK::Iter::Statistical;
void ck(bool b, const char* m)
{
    if (!b)
        throw std::runtime_error(m);
}
struct Diagonal final : IDataFidelity {
    std::vector<float> target, weights;
    std::string message;
    DataFidelityCapabilities capabilities() const override { return { true, true, true, false, AdjointStatus::Verified, true }; }
    size_t workspaceBytes(const SReconstructionParams&) const override { return 0; }
    bool prepare(const DataFidelitySetup&) override { return true; }
    bool bind(const ProjectionDataSet&) override { return true; }
    bool algebraicUpdate(float* x,float) override { return cudaMemcpy(x,target.data(),target.size()*4,cudaMemcpyHostToDevice)==cudaSuccess; }
    const std::string& error() const override { return message; }
    bool evaluate(const float* x, float* gradient, double* value) override
    {
        std::vector<float> h(target.size()), g(h.size());
        ck(cudaMemcpy(h.data(), x, h.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess, "mock download");
        double f = 0;
        for (size_t i = 0; i < h.size(); ++i) {
            double r = h[i] - target[i];
            f += .5 * weights[i] * r * r;
            g[i] = float(weights[i] * r);
        }
        if (gradient)
            ck(cudaMemcpy(gradient, g.data(), g.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess, "mock upload");
        if (value)
            *value = f;
        return true;
    }
    bool estimateLipschitz(float*, float*, double& value) override
    {
        value = *std::max_element(weights.begin(), weights.end());
        return true;
    }
};
void testOptimizer()
{
    Diagonal d;
    for (int i = 0; i < 17; ++i) {
        d.target.push_back(.2f + .01f * i);
        d.weights.push_back(1.f + float(i % 3));
    }
    float* x = nullptr;
    cudaMalloc(&x, d.target.size() * 4);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    YK::SStatisticalOptimizerSpec cfg;
    cfg.kind = YK::EStatisticalOptimizer::GradientDescent;
    cfg.iterations = 80;
    cfg.step_size = 10;
    cfg.relative_tolerance = 1e-6f;
    auto opt = createOptimizer(cfg.kind);
    ck(opt && opt->prepare(d.target.size(), cfg, {}, stream, 0), "optimizer prepare");
    SReconstructionParams p {};
    p.volume.Nx = 17;
    p.volume.Ny = p.volume.Nz = 1;
    p.volume.voxelX_mm = p.volume.voxelY_mm = p.volume.voxelZ_mm = 1;
    AlgebraicRegularizationConfig reg_config;
    reg_config.gradient_only = true;
    auto reg = createRegularizerPlugin("tikhonov");
    ck(reg->prepare(p, reg_config, stream, 0), "tikhonov prepare");
    ReconstructionProblem problem { d, reg.get(), .2f, reg->gradientLipschitz() };
    cudaMemset(x, 0, d.target.size() * 4);
    ck(opt->run(problem, x), "GD known quadratic run");
    std::vector<float> h(d.target.size()), repeat(h.size());
    cudaMemcpy(h.data(), x, h.size() * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < h.size(); ++i)
        ck(std::abs(h[i] - d.weights[i] * d.target[i] / (d.weights[i] + .2f)) < 2e-5, "GD known solution mismatch");
    ck(opt->statistics().final_objective <= opt->statistics().initial_objective && opt->statistics().total_backtracking > 0, "backtracking did not reduce objective");
    opt->reset();
    cudaMemset(x, 0, h.size() * 4);
    ck(opt->run(problem, x), "GD reset");
    cudaMemcpy(repeat.data(), x, h.size() * 4, cudaMemcpyDeviceToHost);
    ck(repeat == h, "GD reset changed state");
    struct StopAfterTwo final : IOptimizerObserver {
        bool continueIteration(const OptimizerStatistics& s) override { return s.completed_iterations < 2; }
    } stop;
    problem.observer = &stop;
    cudaMemset(x, 0, h.size() * 4);
    ck(opt->run(problem, x) && opt->statistics().cancelled && opt->statistics().completed_iterations == 2, "clean optimizer cancellation");
    problem.observer = nullptr;
    ck(!createOptimizer(YK::EStatisticalOptimizer::Fista), "FISTA silently dispatched to GD");
    reg->release();
    opt->release();
    cudaFree(x);
    cudaStreamDestroy(stream);
}
void testCompositionAndProx() {
    using Strategy=ERegularizationStrategy;
    YK::SStatisticalReconstructionSpec defaults;
    ck(defaultComposition(defaults).strategy==Strategy::Proximal,"default Huber must use Prox");
    defaults.regularization.plugin_id="huber_tv";
    ck(defaultComposition(defaults).strategy==Strategy::Proximal,"Huber alias must use Prox");
    defaults.optimizer.kind=YK::EStatisticalOptimizer::GradientDescent;
    ck(defaultComposition(defaults).strategy==Strategy::AlternatingGradient,"raw GD policy changed");
    SReconstructionParams p{};p.volume.Nx=2;p.volume.Ny=p.volume.Nz=1;
    p.volume.voxelX_mm=p.volume.voxelY_mm=p.volume.voxelZ_mm=1;
    float *x=nullptr,*out=nullptr,*g=nullptr;cudaMalloc(&x,8);cudaMalloc(&out,8);cudaMalloc(&g,8);
    Diagonal data;data.target={2,4};data.weights={1,1};
    auto tik=createRegularizerPlugin("tikhonov");AlgebraicRegularizationConfig rc;rc.strength=1;
    ck(tik->prepare(p,rc,nullptr,0),"prepare composition regularizer");
    for(auto strategy:{Strategy::AlternatingGradient,Strategy::SynchronousGradient,Strategy::Proximal}){
        auto opt=createOptimizer(YK::EStatisticalOptimizer::RegularizedAlgebraic);
        SRegularizationCompositionSpec comp;comp.strategy=strategy;
        ck(opt->configureComposition(comp),"composition configuration");
        YK::SStatisticalOptimizerSpec cfg;cfg.iterations=1;cfg.step_size=.25f;cfg.backtracking=false;
        ck(opt->prepare(2,cfg,{},nullptr,0),"composition prepare");
        float old[2]={1,3},actual[2];cudaMemcpy(x,old,8,cudaMemcpyHostToDevice);
        ReconstructionProblem problem{data,tik.get(),.5f,1};
        ck(opt->run(problem,x),"composition update");cudaMemcpy(actual,x,8,cudaMemcpyDeviceToHost);
        for(int i=0;i<2;++i){
            float expected=strategy==Strategy::Proximal?data.target[i]/1.125f:data.target[i]-.125f*(strategy==Strategy::SynchronousGradient?old[i]:data.target[i]);
            ck(std::abs(actual[i]-expected)<1e-6,"composition uses wrong reference volume");
        }
        problem.strength=0;cudaMemcpy(x,old,8,cudaMemcpyHostToDevice);
        ck(opt->run(problem,x),"zero composition");cudaMemcpy(actual,x,8,cudaMemcpyDeviceToHost);
        ck(actual[0]==2&&actual[1]==4,"zero penalty changes algebraic update");
    }
    for(const char* id:{"tv","smoothed_tv","huber_tv","tikhonov"}){
        auto reg=createRegularizerPlugin(id);rc.require_prox=true;rc.inner_iterations=3000;rc.epsilon=rc.huber_delta=.2f;
        rc.non_negative=false;rc.lower_bound=-10;rc.upper_bound=10;
        ck(reg->capabilities().prox&&reg->prepare(p,rc,nullptr,0),"prox capability/prepare");
        float z[2]={0,2},u[2],grad[2];cudaMemcpy(x,z,8,cudaMemcpyHostToDevice);
        ck(reg->prox(x,out,.25f),"fixed reference prox");cudaMemcpy(u,out,8,cudaMemcpyDeviceToHost);
        if(std::string(id)=="tv")ck(std::abs(u[0]-.25f)<2e-5&&std::abs(u[1]-1.75f)<2e-5,"standard TV analytic two-pixel prox");
        else {
            ck(reg->gradient(out,g),"prox optimality gradient");cudaMemcpy(grad,g,8,cudaMemcpyDeviceToHost);
            for(int i=0;i<2;++i)ck(std::abs(u[i]-z[i]+.25f*grad[i])<2e-5,"prox stationarity mismatch");
        }
        ck(reg->prox(x,x,.25f),"aliased prox");float alias[2];cudaMemcpy(alias,x,8,cudaMemcpyDeviceToHost);
        ck(std::abs(alias[0]-u[0])<1e-6&&std::abs(alias[1]-u[1])<1e-6,"prox changed fixed reference when aliased");
        ck(reg->prox(x,out,0),"zero prox");cudaMemcpy(alias,out,8,cudaMemcpyDeviceToHost);ck(alias[0]==u[0]&&alias[1]==u[1],"gamma zero changed volume");
    }
    cudaFree(x);cudaFree(out);cudaFree(g);
}
void testRegularization()
{
    SReconstructionParams p {};
    p.volume.Nx = 4;
    p.volume.Ny = 3;
    p.volume.Nz = 2;
    p.volume.voxelX_mm = .7f;
    p.volume.voxelY_mm = 1.3f;
    p.volume.voxelZ_mm = 2.1f;
    std::vector<float> h(24), g(24);
    for (size_t i = 0; i < h.size(); ++i)
        h[i] = .3f + .05f * float(std::sin(i * .7));
    float *x = nullptr, *out = nullptr;
    cudaMalloc(&x, h.size() * 4);
    cudaMalloc(&out, h.size() * 4);
    for (const char* id : { "smoothed_tv", "huber_tv", "tikhonov" }) {
        auto reg = createRegularizerPlugin(id);
        AlgebraicRegularizationConfig c;
        c.gradient_only = true;
        c.huber_delta = c.epsilon = .2f;
        ck(reg && reg->capabilities().value && reg->capabilities().gradient && reg->prepare(p, c, nullptr, 0), "gradient regularizer capability");
        ck(reg->workspaceBytes(p, c) == 8192, "gradient-only plugin allocated prox volumes");
        cudaMemcpy(x, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
        ck(reg->gradient(x, out), "plugin gradient");
        cudaMemcpy(g.data(), out, h.size() * 4, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < h.size(); ++i) {
            auto z = h;
            float plus = z[i] + 1e-3f, minus = z[i] - 1e-3f;
            z[i] = plus;
            cudaMemcpy(x, z.data(), z.size() * 4, cudaMemcpyHostToDevice);
            double fp = 0, fm = 0;
            ck(reg->value(x, fp), "penalty value plus");
            z[i] = minus;
            cudaMemcpy(x, z.data(), z.size() * 4, cudaMemcpyHostToDevice);
            ck(reg->value(x, fm), "penalty value minus");
            ck(std::abs((fp - fm) / (plus - minus) - g[i]) < 2e-4, "plugin value/gradient finite difference mismatch");
        }
        reg->release();
    }
    auto tv = createRegularizerPlugin("tv");
    ck(tv && tv->capabilities().value && !tv->capabilities().gradient, "standard TV capability declaration");
    cudaFree(x);
    cudaFree(out);
}
using namespace yk::spectral;
struct Reader {
    std::vector<float> data;
    size_t pixels = 32 * 24;
    bool fail = false;
    static bool read(void* user, const int* indices, int count, float* dest, size_t elements)
    {
        auto& r = *static_cast<Reader*>(user);
        if (r.fail)
            return false;
        ck(elements == size_t(count) * r.pixels, "reader shape");
        for (int i = 0; i < count; ++i)
            std::copy_n(r.data.data() + size_t(indices[i]) * r.pixels, r.pixels, dest + size_t(i) * r.pixels);
        return true;
    }
};
SimulationConfig fixture()
{
    SimulationConfig c;
    auto& g = c.geometry.parameters;
    g.views = g.views_per_turn = 13;
    g.detector_u = 32;
    g.detector_v = 24;
    g.volume_x = g.volume_y = g.volume_z = 16;
    g.reconstruction_volume_x = g.reconstruction_volume_y = g.reconstruction_volume_z = 16;
    c.reconstruction.type = "statistical";
    c.reconstruction.statistical.optimizer.iterations = 3;
    c.reconstruction.statistical.regularization.strength = .01;
    return c;
}
std::vector<float> runSdk(const char* penalty, int source, bool weighted, bool zero = false, bool sirt = false, bool unit_weights = false )
{
    auto c = fixture();
    c.reconstruction.statistical.regularization.plugin = penalty;
    if (zero)
        c.reconstruction.statistical.regularization.strength = 0;
    auto s = makeLibrarySystem(c, YK::EPipeline::Statistical, YK::EProjectionModel::Joseph, YK::EFdkFilter::RamLak, GeometryUse::Reconstruction);
    const auto strategy=defaultComposition(s.statistical).strategy;
    s.statistical.data_fidelity.use_projection_weights = weighted;
    YK::SStatisticalWorkspaceEstimate estimate;
    ck(YK::estimateStatisticalWorkspace(s, estimate), "workspace estimate");
    const std::string id=penalty;
    size_t reg_bytes=id=="none"?0:8192;
    if(strategy==ERegularizationStrategy::Proximal && id!="none" && id!="tikhonov" && (!zero||id=="smoothed_tv"))
        reg_bytes+=size_t(16*16*16)*sizeof(float)*(id=="tv"?5:2);
    ck(estimate.optimizer_bytes == 3 * 16 * 16 * 16 * sizeof(float) + 8192 && estimate.regularization_bytes == reg_bytes, "workspace module accounting");
    ck(estimate.caller_volume_bytes == 16 * 16 * 16 * sizeof(float) && ((estimate.weight_staging_bytes > 0) == weighted), "workspace source accounting");
    if (sirt) {
        s.reconstruction.pipeline = YK::EPipeline::SIRT;
        s.reconstruction.iterative.iterations = 3;
        s.reconstruction.iterative.subsets = 1;
    }
    auto* session = YK::ReconstructionSessionFactory::create();
    ck(session->initialize(s), session->lastErrorMessage());
    Reader b, w;
    for (int i = 0; i < 13 * 32 * 24; ++i) {
        b.data.push_back(.1f + .001f * (i % 29));
        w.data.push_back(unit_weights ? 1.f : .5f + .1f * (i % 7));
    }
    float *x = nullptr, *db = nullptr, *dw = nullptr;
    cudaMalloc(&x, 16 * 16 * 16 * 4);
    std::vector<float> output(16 * 16 * 16), repeat(output.size());
    YK::SExecutionRequest r;
    r.volume = { x, YK::EMemoryLocation::Device, output.size() };
    r.view_count = 13;
    r.iteration_count = 3;
    r.projection_batch_views = 5;
    if (source == 0) {
        cudaMalloc(&db, b.data.size() * 4);
        cudaMemcpy(db, b.data.data(), b.data.size() * 4, cudaMemcpyHostToDevice);
        r.projection = { db, YK::EMemoryLocation::Device, b.data.size() };
    } else if (source == 1)
        r.projection = { b.data.data(), YK::EMemoryLocation::Host, b.data.size() };
    else {
        r.projection_read = &Reader::read;
        r.projection_read_user_data = &b;
    }
    if (weighted) {
        if (source == 0) {
            cudaMalloc(&dw, w.data.size() * 4);
            cudaMemcpy(dw, w.data.data(), w.data.size() * 4, cudaMemcpyHostToDevice);
            r.projection_weights = { dw, YK::EMemoryLocation::Device, w.data.size() };
        } else if (source == 1)
            r.projection_weights = { w.data.data(), YK::EMemoryLocation::Host, w.data.size() };
        else {
            r.projection_weights_read = &Reader::read;
            r.projection_weights_read_user_data = &w;
        }
    }
    int calls = 0;
    r.progress_callback = [](void* data, int done, int total) { ck(done == ++*static_cast<int*>(data) && total == 3, "statistical progress"); };
    r.progress_user_data = &calls;
    cudaMemset(x, 0, output.size() * 4);
    ck(session->execute(r), session->lastErrorMessage());
    ck(calls == 3, "statistical progress missing");
    cudaMemcpy(output.data(), x, output.size() * 4, cudaMemcpyDeviceToHost);
    session->reset();
    r.progress_callback = nullptr;
    cudaMemset(x, 0, output.size() * 4);
    ck(session->execute(r), session->lastErrorMessage());
    cudaMemcpy(repeat.data(), x, repeat.size() * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < output.size(); ++i)
        ck(std::isfinite(output[i]) && output[i] >= 0 && std::abs(output[i] - repeat[i]) < 1e-6, "statistical reset/constraint mismatch");
    if (weighted && source == 1) {
        w.data[0] = -1;
        session->reset();
        ck(!session->execute(r), "negative W accepted");
        w.data[0] = std::numeric_limits<float>::quiet_NaN();
        session->reset();
        ck(!session->execute(r), "NaN W accepted");
    }
    if (source == 2) {
        b.fail = true;
        session->reset();
        ck(!session->execute(r), "failed projection reader accepted");
        b.fail = false;
        if (weighted) {
            w.fail = true;
            session->reset();
            ck(!session->execute(r), "failed weight reader accepted");
        }
    }
    cudaFree(x);
    if (db)
        cudaFree(db);
    if (dw)
        cudaFree(dw);
    YK::ReconstructionSessionFactory::destroy(session);
    return output;
}
void testDataFidelity()
{
    const auto tv_resident=runSdk("tv",0,true);
    const auto tv_batched=runSdk("tv",2,true);
    for(size_t i=0;i<tv_resident.size();++i)ck(std::abs(tv_resident[i]-tv_batched[i])<2e-6,"internal TV policy batch mismatch");
    auto c = fixture();
    auto s = makeLibrarySystem(c, YK::EPipeline::Statistical, YK::EProjectionModel::Joseph, YK::EFdkFilter::RamLak, GeometryUse::Reconstruction);
    std::vector<YK::SConeProjGeomVec> geometry;
    std::vector<YK::SCylConeProjGeomVec> cylindrical;
    YK::SVolGeom volume;
    ck(YK::buildSystemGeometry(s.geometry, geometry, cylindrical, volume), "data geometry");
    SReconstructionParams p {};
    p.scan.Nu = 32;
    p.scan.Nv = 24;
    p.scan.NAng = p.scan.totalViews = 13;
    p.scan.du_mm = p.scan.dv_mm = float(c.geometry.parameters.pixel_u_mm);
    p.scan.sid_mm = float(c.geometry.parameters.sid_mm);
    p.scan.sdd_mm = float(c.geometry.parameters.sdd_mm);
    for (const auto& g : geometry)
        p.scan.angles.push_back(g.angle.x);
    p.volume.Nx = p.volume.Ny = p.volume.Nz = 16;
    p.volume.voxelX_mm = p.volume.voxelY_mm = p.volume.voxelZ_mm = float(c.geometry.parameters.voxel_x_mm);
    YK::SDataFidelitySpec config;
    config.scale = 2;
    config.normalize_by_ray_count = true;
    config.use_projection_weights = true;
    auto data = createDataFidelity(config.kind);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ck(data && data->prepare({ p, geometry, config, YK::ETask::FP_Joseph, YK::ETask::BP_Joseph_v3, stream, 0 }), "data fidelity prepare");
    std::vector<float> b(13 * 32 * 24), w(b.size());
    double expected = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = .1f + .001f * float(i % 19);
        w[i] = .5f + .1f * float(i % 7);
        expected += .5 * double(w[i]) * b[i] * b[i];
    }
    expected *= 2.0 / b.size();
    float *db = nullptr, *dw = nullptr, *x = nullptr, *gradient = nullptr;
    cudaMalloc(&db, b.size() * 4);
    cudaMalloc(&dw, w.size() * 4);
    cudaMalloc(&x, 16 * 16 * 16 * 4);
    cudaMalloc(&gradient, 16 * 16 * 16 * 4);
    cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(x, 0, 16 * 16 * 16 * 4);
    ck(data->bind({ db, nullptr, dw, nullptr }), "bind W");
    double objective = 0;
    ck(data->evaluate(x, gradient, &objective) && std::abs(objective - expected) < 1e-8, "weighted normalized objective mismatch");
    cudaMemset(dw, 0, w.size() * 4);
    ck(data->evaluate(x, gradient, &objective) && objective == 0, "zero W objective");
    std::vector<float> g(16 * 16 * 16);
    cudaMemcpy(g.data(), gradient, g.size() * 4, cudaMemcpyDeviceToHost);
    for (float value : g)
        ck(value == 0, "zero W gradient");
    data.reset();
    cudaFree(db);
    cudaFree(dw);
    cudaFree(x);
    cudaFree(gradient);
    cudaStreamDestroy(stream);
}
void testSdk()
{
    for (int source : { 0, 1, 2 }) {
        const auto reference = runSdk("none", source, false, true, true);
        const auto normalized = runSdk("none", source, false, true);
        const auto with_unit_w = runSdk("none", source, true, true, false, true);
        for (size_t i = 0; i < reference.size(); ++i) {
            ck(std::abs(reference[i] - normalized[i]) < 1e-6, "zero penalty differs from internal SIRT");
            ck(std::abs(reference[i] - with_unit_w[i]) < 1e-6, "W=I algebraic update differs from SIRT");
        }
    }
    for (const char* id : { "smoothed_tv", "huber_tv", "tikhonov" })
        for (bool weights : { false, true }) {
            auto reference = runSdk(id, 0, weights);
            for (int source : { 1, 2 }) {
                auto observed = runSdk(id, source, weights);
                for (size_t i = 0; i < reference.size(); ++i)
                    ck(std::abs(reference[i] - observed[i]) < 2e-6, "statistical resident/batched mismatch");
            }
        }
    auto c = fixture();
    auto s = makeLibrarySystem(c, YK::EPipeline::Statistical, YK::EProjectionModel::Joseph, YK::EFdkFilter::RamLak, GeometryUse::Reconstruction);
    auto* session = YK::ReconstructionSessionFactory::create();
    s.statistical.optimizer.kind = YK::EStatisticalOptimizer::Fista;
    ck(!session->initialize(s) && session->lastError() == YK::EApiErrorCode::UnsupportedCombination, "FISTA unsupported check");
    s.statistical.optimizer.kind = YK::EStatisticalOptimizer::GradientDescent;
    s.statistical.regularization.plugin_id = "tv";
    ck(!session->initialize(s) && session->lastError() == YK::EApiErrorCode::UnsupportedCombination, "nondifferentiable TV accepted");
    s.statistical.optimizer.kind=YK::EStatisticalOptimizer::RegularizedAlgebraic;
    ck(session->initialize(s),"standard TV prox rejected");
    s.statistical.regularization.plugin_id = "huber_tv";
    s.statistical.data_fidelity.require_verified_adjoint = true;
    ck(!session->initialize(s), "unverified adjoint accepted as verified");
    s.statistical.data_fidelity.require_verified_adjoint = false;
    s.struct_size = offsetof(YK::SSystemSpec, statistical);
    ck(!session->initialize(s), "missing statistical tail accepted");
    s.reconstruction.pipeline = YK::EPipeline::SIRT;
    s.regularization.plugin_id = nullptr;
    ck(session->initialize(s), "old system ABI rejected");
    YK::ReconstructionSessionFactory::destroy(session);
}
int main()
{
    try {
        testOptimizer();
        testCompositionAndProx();
        testRegularization();
        testDataFidelity();
        testSdk();
        std::cout << "Statistical contracts, known-solution GD, gradient checks, weighted/batched SDK passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
