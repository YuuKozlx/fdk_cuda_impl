#include "Reconstruction/Iterative/Regularization/HuberTvLaunch.cuh"
#include "Reconstruction/Iterative/Common/kernels/YkTvRegularizationLaunch.cuh"
#include "Reconstruction/Iterative/Regularization/RegularizerPlugin.hpp"
#include "YKCBCT/interface/YkReconstructionApi.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace YK::Iter;
void ck(bool b, const char* m)
{
    if (!b)
        throw std::runtime_error(m);
}
struct Grid {
    int x, y, z, dim;
    double hx = .7, hy = 1.3, hz = 2.1, delta = .2;
    size_t n() const { return size_t(x) * y * z; }
    size_t id(int i, int j, int k) const { return (size_t(k) * y + j) * x + i; }
};
// Independent CPU definition: scatter each forward difference's contribution
// to its two endpoints, rather than evaluating GPU's neighbor divergence.
double energy(const std::vector<double>& u, const Grid& p, std::vector<double>* gradient = nullptr)
{
    if (gradient)
        gradient->assign(u.size(), 0.0);
    double value = 0;
    for (int k = 0; k < p.z; ++k)
        for (int j = 0; j < p.y; ++j)
            for (int i = 0; i < p.x; ++i) {
                const size_t a = p.id(i, j, k);
                const size_t b[3] = { i + 1 < p.x ? p.id(i + 1, j, k) : a, j + 1 < p.y ? p.id(i, j + 1, k) : a, k + 1 < p.z && p.dim == 3 ? p.id(i, j, k + 1) : a };
                const double h[3] = { p.hx, p.hy, p.hz };
                double d[3], norm2 = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    d[axis] = (u[b[axis]] - u[a]) / h[axis];
                    norm2 += d[axis] * d[axis];
                }
                const double norm = std::sqrt(norm2);
                value += norm <= p.delta ? norm2 / (2 * p.delta) : norm - p.delta / 2;
                if (gradient)
                    for (int axis = 0; axis < 3; ++axis) {
                        double v = d[axis] / (std::max(p.delta, norm) * h[axis]);
                        (*gradient)[a] -= v;
                        (*gradient)[b[axis]] += v;
                    }
            }
    return value;
}
SReconstructionParams params(const Grid& g)
{
    SReconstructionParams p {};
    p.volume.Nx = g.x;
    p.volume.Ny = g.y;
    p.volume.Nz = g.z;
    p.volume.voxelX_mm = float(g.hx);
    p.volume.voxelY_mm = float(g.hy);
    p.volume.voxelZ_mm = float(g.hz);
    return p;
}
void checkGrid(Grid g)
{
    std::vector<float> input(g.n()), gpu(g.n());
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = float(.3 + .12 * std::sin(i * .7) + .05 * std::cos(i * .13));
    std::vector<double> z(input.begin(), input.end()), gradient;
    energy(z, g, &gradient);
    float *du = nullptr, *dg = nullptr;
    ck(cudaMalloc(&du, g.n() * 4) == cudaSuccess && cudaMalloc(&dg, g.n() * 4) == cudaSuccess, "allocation");
    ck(cudaMemcpy(du, input.data(), g.n() * 4, cudaMemcpyHostToDevice) == cudaSuccess, "upload");
    huber_tv_gradient_launch(du, dg, g.x, g.y, g.z, float(g.hx), float(g.hy), float(g.hz), float(g.delta), g.dim, nullptr);
    ck(cudaMemcpy(gpu.data(), dg, g.n() * 4, cudaMemcpyDeviceToHost) == cudaSuccess, "gradient kernel");
    for (size_t i = 0; i < z.size(); ++i) {
        ck(std::abs(gradient[i] - gpu[i]) < 3e-6, "CPU/GPU gradient mismatch");
        auto v = z;
        const double eps = 1e-6;
        v[i] += eps;
        double plus = energy(v, g);
        v[i] -= 2 * eps;
        double minus = energy(v, g);
        ck(std::abs((plus - minus) / (2 * eps) - gpu[i]) < 3e-6, "finite difference mismatch");
    }
    AlgebraicRegularizationConfig c;
    c.plugin_id = "huber_tv_prox";
    c.strength = .04f;
    c.huber_delta = float(g.delta);
    c.inner_iterations = 100;
    c.tv_dimensionality = static_cast<ETvDimensionality>(g.dim);
    c.strength_reduction = .8f;
    auto plugin = createRegularizerPlugin(c.plugin_id);
    ck(plugin && plugin->prepare(params(g), c, nullptr, 0), "prox prepare");
    ck(plugin->workspaceBytes(params(g), c) == 2 * g.n() * 4 + 8192, "prox workspace");
    ck(plugin->apply(du, {}), "prox apply");
    ck(cudaMemcpy(gpu.data(), du, g.n() * 4, cudaMemcpyDeviceToHost) == cudaSuccess, "prox read");
    auto objective = [&](const std::vector<double>& u) {double f=c.strength*energy(u,g);for(size_t i=0;i<u.size();++i)f+=.5*(u[i]-z[i])*(u[i]-z[i]);return f; };
    std::vector<double> cpu = z;
    const double bound = 1 + c.strength * 4 * (1 / (g.hx * g.hx) + 1 / (g.hy * g.hy) + (g.dim == 3 ? 1 / (g.hz * g.hz) : 0)) / c.huber_delta;
    for (int step = 0; step < c.inner_iterations; ++step) {
        energy(cpu, g, &gradient);
        for (size_t i = 0; i < cpu.size(); ++i)
            cpu[i] = std::max(0.0, cpu[i] - (cpu[i] - z[i] + c.strength * gradient[i]) / bound);
    }
    std::vector<double> observed(gpu.begin(), gpu.end());
    ck(objective(observed) <= objective(z) + 1e-7, "prox increased objective");
    for (size_t i = 0; i < cpu.size(); ++i)
        ck(std::abs(cpu[i] - gpu[i]) < 2e-6, "fixed-z prox mismatch");
    plugin->reset();
    cudaMemcpy(du, input.data(), g.n() * 4, cudaMemcpyHostToDevice);
    ck(plugin->apply(du, {}), "reset apply");
    std::vector<float> repeat(g.n());
    cudaMemcpy(repeat.data(), du, g.n() * 4, cudaMemcpyDeviceToHost);
    ck(repeat == gpu, "reset schedule changed result");
    plugin->release();
    c.strength = 0;
    c.non_negative = true;
    ck(plugin->prepare(params(g), c, nullptr, 0), "zero strength");
    std::fill(input.begin(), input.end(), -.2f);
    cudaMemcpy(du, input.data(), g.n() * 4, cudaMemcpyHostToDevice);
    ck(plugin->apply(du, {}), "zero strength apply");
    cudaMemcpy(gpu.data(), du, g.n() * 4, cudaMemcpyDeviceToHost);
    for (float v : gpu)
        ck(v == 0, "nonnegative projection");
    c.huber_delta = 0;
    ck(!plugin->prepare(params(g), c, nullptr, 0), "invalid delta accepted");
    c.huber_delta = .2f;
    c.strength = .04f;
    c.step_size = 2;
    ck(!plugin->prepare(params(g), c, nullptr, 0), "unsafe step accepted");
    c.step_size = 0;
    c.strength = -1;
    ck(!plugin->prepare(params(g), c, nullptr, 0), "negative strength accepted");
    cudaFree(du);
    cudaFree(dg);
}
struct Probe final : IAlgebraicRegularizer {
    static inline int calls = 0;
    bool prepare(const SReconstructionParams&, const AlgebraicRegularizationConfig&, cudaStream_t, int) override { return true; }
    bool apply(float*, const AlgebraicRegularizationContext& c) override
    {
        ck(c.outer_iteration == calls % 3, "plugin not scheduled per sweep");
        ++calls;
        return true;
    }
    void reset() override { }
    void release() override { }
};
void checkRegistryAndAbi()
{
    ck(registerRegularizerPlugin("test_probe", [] { return std::make_unique<Probe>(); }), "register plugin");
    ck(!registerRegularizerPlugin("test_probe", [] { return std::make_unique<Probe>(); }), "duplicate registration");
    ck(!createRegularizerPlugin("missing"), "unknown plugin");
    YK::SSystemSpec s;
    auto& g = s.geometry;
    g.circular.total_views = g.circular.views_per_turn = 5;
    g.flat_detector.channels = 16;
    g.flat_detector.rows = 16;
    g.volume.nx = g.volume.ny = g.volume.nz = 8;
    g.volume.voxel_x_mm = g.volume.voxel_y_mm = g.volume.voxel_z_mm = 1;
    g.circular.sid_mm = 709;
    g.circular.sdd_mm = 1143;
    g.flat_detector.channel_size_mm = g.flat_detector.row_size_mm = 1;
    s.reconstruction.iterative.iterations = 3;
    float *dp = nullptr, *dv = nullptr;
    cudaMalloc(&dp, 5 * 16 * 16 * 4);
    cudaMalloc(&dv, 8 * 8 * 8 * 4);
    cudaMemset(dp, 0, 5 * 16 * 16 * 4);
    YK::SExecutionRequest request;
    request.view_count = 5;
    request.projection = { dp, YK::EMemoryLocation::Device, 5 * 16 * 16 };
    request.volume = { dv, YK::EMemoryLocation::Device, 8 * 8 * 8 };
    for (auto pipeline : { YK::EPipeline::SART, YK::EPipeline::SIRT, YK::EPipeline::OSSART }) {
        auto* session = YK::ReconstructionSessionFactory::create();
        s.reconstruction.pipeline = pipeline;
        s.reconstruction.iterative.subsets = 2;
        s.regularization.plugin_id = "test_probe";
        Probe::calls = 0;
        ck(session->initialize(s), "registered plugin SDK init");
        ck(session->execute(request) && Probe::calls == 3, "SDK full-sweep insertion failed");
        session->reset();
        ck(session->execute(request) && Probe::calls == 6, "plugin reset failed");
        s.regularization.plugin_id = "missing";
        ck(!session->initialize(s), "unknown SDK plugin accepted");
        s.struct_size = offsetof(YK::SSystemSpec, regularization);
        ck(session->initialize(s), "old ABI inspected regularization tail");
        s.struct_size = offsetof(YK::SSystemSpec, regularization) + 1;
        ck(!session->initialize(s) && session->lastError() == YK::EApiErrorCode::InvalidConfig, "truncated plugin tail accepted");
        s.struct_size = sizeof(s);
        YK::ReconstructionSessionFactory::destroy(session);
    }
    cudaFree(dp);
    cudaFree(dv);
}

// Scatter backward-difference fluxes to endpoints independently of GPU gather.
void checkTigreTvDirections() {
    for(Grid p:{Grid{5,4,3,3},Grid{1,4,3,3},Grid{5,1,3,3},Grid{5,4,1,3},Grid{1,1,1,3}}){
        float *x=nullptr,*g=nullptr;ck(cudaMalloc(&x,p.n()*4)==cudaSuccess&&cudaMalloc(&g,p.n()*4)==cudaSuccess,"TIGRE gradient buffers");
        for(int pattern=0;pattern<4;++pattern)for(bool adaptive:{false,true}){
            std::vector<float> input(p.n()),observed(p.n());
            for(int k=0;k<p.z;++k)for(int j=0;j<p.y;++j)for(int i=0;i<p.x;++i)
                input[p.id(i,j,k)]=pattern==0?float(i==p.x/2):pattern==1?float(j==p.y/2):pattern==2?float(k==p.z/2):float(.3+.2*std::sin(p.id(i,j,k)*.73));
            const double eps=1e-5,delta=.3;
            std::vector<double> expected(p.n(),0);
            for(int k=0;k<p.z;++k)for(int j=0;j<p.y;++j)for(int i=0;i<p.x;++i){
                const auto a=p.id(i,j,k);
                const size_t b[3]={i?p.id(i-1,j,k):a,j?p.id(i,j-1,k):a,k?p.id(i,j,k-1):a};
                double diff[3],w[3],norm2=0;
                for(int axis=0;axis<3;++axis){diff[axis]=double(input[a])-input[b[axis]];w[axis]=adaptive?std::exp(-diff[axis]*diff[axis]/(delta*delta)):1;norm2+=w[axis]*diff[axis]*diff[axis];}
                for(int axis=0;axis<3;++axis){double flux=w[axis]*diff[axis]/(std::sqrt(norm2)+eps);expected[a]+=flux;expected[b[axis]]-=flux;}
            }
            ck(cudaMemcpy(x,input.data(),p.n()*4,cudaMemcpyHostToDevice)==cudaSuccess,"TIGRE upload");
            tigre_tv_gradient_launch(x,g,p.x,p.y,p.z,float(eps),adaptive,float(delta),nullptr);
            ck(cudaMemcpy(observed.data(),g,p.n()*4,cudaMemcpyDeviceToHost)==cudaSuccess,"TIGRE gradient download");
            for(size_t a=0;a<p.n();++a)ck(std::abs(expected[a]-observed[a])<3e-5,"TIGRE TV axis/boundary gradient mismatch");
            if(!adaptive && pattern==3){
                auto value=[&](const std::vector<float>& u){double total=0;for(int k=0;k<p.z;++k)for(int j=0;j<p.y;++j)for(int i=0;i<p.x;++i){size_t a=p.id(i,j,k);double dx=i?double(u[a])-u[a-1]:0,dy=j?double(u[a])-u[a-p.x]:0,dz=k?double(u[a])-u[a-size_t(p.x)*p.y]:0;double t=std::sqrt(dx*dx+dy*dy+dz*dz);total+=t-eps*std::log1p(t/eps);}return total;};
                for(size_t a=0;a<p.n();++a){auto plus=input,minus=input;plus[a]+=.0001f;minus[a]-=.0001f;double fd=(value(plus)-value(minus))/(double(plus[a])-minus[a]);ck(std::abs(fd-observed[a])<.002,"TIGRE TV finite difference mismatch");}
            }
        }
        cudaFree(x);cudaFree(g);
    }
}

void checkAutomaticProx(){
    Grid grid{16,12,8,3};grid.hx=grid.hy=grid.hz=1;grid.delta=5e-5;
    const auto p=params(grid);std::vector<float> input(grid.n(),.02f),observed(grid.n()),fixed(grid.n());
    float *x=nullptr,*out=nullptr;ck(cudaMalloc(&x,grid.n()*4)==cudaSuccess&&cudaMalloc(&out,grid.n()*4)==cudaSuccess,"auto buffers");
    AlgebraicRegularizationConfig c;c.plugin_id="huber_tv_prox";c.strength=1e-4;c.huber_delta=float(grid.delta);
    configureAutomaticHuberProx(c);ck(c.inner_iterations==1000&&c.step_size==0,"automatic policy");
    auto automatic=createRegularizerPlugin(c.plugin_id);ck(automatic->prepare(p,c,nullptr,0),"auto prepare");
    cudaMemcpy(x,input.data(),grid.n()*4,cudaMemcpyHostToDevice);ck(automatic->prox(x,out,c.strength),"constant auto prox");
    ck(automatic->proxStatistics().converged&&automatic->proxStatistics().iterations==0,"constant did not stop immediately");
    for(size_t i=0;i<input.size();++i)input[i]=.02f+.001f*float(std::sin(i*.71));
    cudaMemcpy(x,input.data(),grid.n()*4,cudaMemcpyHostToDevice);ck(automatic->prox(x,out,c.strength),"noisy auto prox");
    cudaMemcpy(observed.data(),out,grid.n()*4,cudaMemcpyDeviceToHost);
    const auto stats=automatic->proxStatistics();ck(stats.iterations>0&&stats.iterations<=1000&&stats.converged!=stats.iteration_limit,"auto termination flags");
    std::vector<double> u(observed.begin(),observed.end()),z(input.begin(),input.end()),grad;energy(u,grid,&grad);
    double norm=0,residual=0;for(size_t i=0;i<u.size();++i){double r=u[i]-std::max(0.,u[i]-(u[i]-z[i]+c.strength*grad[i]));residual+=r*r;norm+=z[i]*z[i];}
    ck(std::abs(std::sqrt(residual/norm)-stats.relative_stationarity)<2e-6,"GPU stationarity differs from independent CPU");
    auto fixed_config=c;fixed_config.prox_relative_tolerance=0;fixed_config.inner_iterations=stats.iterations;
    auto reference=createRegularizerPlugin(c.plugin_id);ck(reference->prepare(p,fixed_config,nullptr,0),"fixed reference prepare");ck(reference->prox(x,out,c.strength),"fixed reference prox");cudaMemcpy(fixed.data(),out,grid.n()*4,cudaMemcpyDeviceToHost);
    for(size_t i=0;i<u.size();++i)ck(std::abs(fixed[i]-observed[i])<1e-7,"auto stopping changes update formula");
    automatic->reset();ck(automatic->prox(x,out,c.strength),"auto reset");ck(automatic->proxStatistics().iterations==stats.iterations,"auto reset step count changed");
    // Hitting a strict one-step budget must not be reported as convergence.
    c.inner_iterations=1;c.prox_relative_tolerance=1e-8;ck(automatic->prepare(p,c,nullptr,0)&&automatic->prox(x,out,c.strength),"one-step prox");
    ck(automatic->proxStatistics().iteration_limit&&!automatic->proxStatistics().converged,"iteration cap falsely reports convergence");
    cudaFree(x);cudaFree(out);
}

void benchmarkProx(){
    Grid grid{256,256,200,3};float *x=nullptr,*out=nullptr;cudaMalloc(&x,grid.n()*4);cudaMalloc(&out,grid.n()*4);
    std::vector<float> input(grid.n());
    for(double spacing:{1.2,.5})for(bool noisy:{false,true}){
        grid.hx=grid.hy=grid.hz=spacing;grid.delta=5e-5;
        for(size_t i=0;i<input.size();++i)input[i]=.02f+(noisy?.002f*(float((unsigned(i)*1664525U+1013904223U)&1023)/1023-.5f):0);
        cudaMemcpy(x,input.data(),grid.n()*4,cudaMemcpyHostToDevice);
        for(int mode:{20,100,0}){
            AlgebraicRegularizationConfig c;c.plugin_id="huber_tv_prox";c.strength=1e-4;c.huber_delta=float(grid.delta);c.inner_iterations=mode?mode:100;
            if(!mode)configureAutomaticHuberProx(c);
            auto plugin=createRegularizerPlugin(c.plugin_id);ck(plugin->prepare(params(grid),c,nullptr,0),"benchmark prepare");
            double milliseconds=0;ProxStatistics stats;
            for(int repeat=0;repeat<4;++repeat){cudaDeviceSynchronize();auto started=std::chrono::steady_clock::now();ck(plugin->prox(x,out,c.strength),"benchmark prox");cudaDeviceSynchronize();double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();if(repeat)milliseconds+=elapsed;stats=plugin->proxStatistics();}
            std::cout<<"BENCH spacing="<<spacing<<" noisy="<<noisy<<" mode="<<(mode?std::to_string(mode):"auto")<<" ms="<<milliseconds/3<<" steps="<<stats.iterations<<" residual="<<stats.relative_stationarity<<" converged="<<stats.converged<<" limit="<<stats.iteration_limit<<'\n';
        }
    }
    cudaFree(x);cudaFree(out);
}
int main(int argc,char** argv)
{
    try {
        if(argc==2&&std::string(argv[1])=="--benchmark"){benchmarkProx();return 0;}
        for (Grid g : { Grid { 5, 4, 3, 3 }, Grid { 5, 4, 3, 2 }, Grid { 1, 4, 2, 3 }, Grid { 1, 1, 1, 3 } })
            checkGrid(g);
        checkTigreTvDirections();
        checkAutomaticProx();
        checkRegistryAndAbi();
        std::cout << "Huber TV gradient/prox, registry and full-sweep insertion passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
