#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <global/YkMacro.hpp>
#include <random>
#include "FDK/YkFdkPipeline.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

#include "common/YkProjectionOperators.hpp"
#include <cmath>
#include <common/YkVecGeo.hpp>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <global/YkCBCTParams.h>
#include <type_traits>
#include <util/YkCudaTimer.hpp>
#include <variant>
#include <vector_functions.hpp>
#include <vector_types.h>

// This header declares reusable geometry and buffer helpers at global scope.
// Resolve the library types here instead of relying on every test source to
// add `using namespace YK` after including this file.
using namespace YK;


static bool read_raw_float(const char* path, std::vector<float>& data)
{
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    size_t n = std::fread(data.data(), sizeof(float), data.size(), fp);
    std::fclose(fp);
    return n == data.size();
}

static bool read_raw_float(const char* path, float* data, uint64_t element_count)
{
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    size_t n = std::fread(data, sizeof(float), element_count, fp);
    std::fclose(fp);
    return n == element_count;
}

static bool write_raw_float(const char* path, const std::vector<float>& data)
{
    FILE* fp = std::fopen(path, "wb");
    if (!fp) return false;
    size_t n = std::fwrite(data.data(), sizeof(float), data.size(), fp);
    std::fclose(fp);
    return n == data.size();
}

static bool write_raw_float(const char* path, const float* data, uint64_t element_count)
{
    FILE* fp = std::fopen(path, "wb");
    if (!fp) return false;
    size_t n = std::fwrite(data, sizeof(float), element_count, fp);
    std::fclose(fp);
    return n == element_count;
}

static void printStats(const std::vector<float>& v, const char* tag)
{
    if (v.empty()) { printf("[%s] empty\n", tag); return; }
    float minv = v[0], maxv = v[0], sum = 0.f;
    for (auto x : v) {
        if (x < minv) minv = x;
        if (x > maxv) maxv = x;
        sum += x;
    }
    printf("[%s] min=%.4f  max=%.4f  mean=%.6f  n=%zu\n",
        tag, minv, maxv, sum / (float)v.size(), v.size());
}

struct datapath {
    inline static const std::string test_data_dir =
        R"(H:\Code\fanproj\fdk-test\TestData\)";
};

static std::string dataPath(const std::string& filename)
{
    return datapath::test_data_dir + "/" + filename;
}


YK_INLINE SCBCTParams make_default_params()
{
    SCBCTParams params;
    params.iPU = 1024;
    params.iPV = 1024;
    params.iPAng = 480;
    params.iPAngTotal = 480;
    params.tiltn_angle_rad = 0.f;
    params.tiltu_angle_rad = 0.f;
    params.tiltv_angle_rad = 0.f;
    params.iVX = 512;
    params.iVY = 512;
    params.iVZ = 400;
    params.bShortScan = true;
    params.scan_range_rad = (float)CUDA_PI * 4.0f / 3.0f;
    params.SID = 500.f;
    params.SDD = 1000.f;
    params.du_mm = 0.25f;
    params.dv_mm = 0.25f;
    params.vox_x_mm = 0.1f;
    params.vox_y_mm = 0.1f;
    params.vox_z_mm = 0.1f;
    params.offsetU_mm = 0.f;
    params.offsetV_mm = 0.f;
    params.vol_offset_x_mm = 0.f;
    params.vol_offset_y_mm = 0.f;
    params.vol_offset_z_mm = 0.f;

    params.angle_list.resize(params.iPAng);
    for (int i = 0; i < params.iPAng; ++i)
        params.angle_list[i] = i * 2.f * (float)CUDA_PI / 720;
    params.scan_start_angle_rad = params.angle_list[0];

    return params;
}



// ================================================================
// 仿真几何配置，描述每帧的抖动模式
// ================================================================
struct SimGeoConfig {
    // 基础扫描参数
    float SID = 500.f;
    float IDD = 500.f;  // SDD - SID
    int   Na = 480;
    int   Nu = 1024;
    int   Nv = 1024;
    float du = 0.25f;
    float dv = 0.25f;
    std::vector<float> angle_list;

    // per-frame offset，size==Na 或 size==1（全局常量）
    std::vector<float3> src_offsets = { make_float3(0,0,0) };
    std::vector<float3> det_offsets = { make_float3(0,0,0) };
    std::vector<float3> detTilt_degs = { make_float3(0,0,0) };
    std::vector<float3> srcCRTilt_degs = { make_float3(0,0,0) };
};

// ================================================================
// 几何生成工厂
// ================================================================
YK_INLINE std::vector<SConeProjGeomVec>
build_sim_geometry(const SimGeoConfig& cfg)
{
    std::vector<SConeProjGeomVec> h_views;
    build_circular_vec_geometry_perframe(
        h_views, cfg.angle_list,
        cfg.Na, cfg.Nu, cfg.Nv,
        cfg.du, cfg.dv,
        cfg.SID, cfg.IDD,
        cfg.det_offsets, cfg.src_offsets,
        cfg.detTilt_degs, cfg.srcCRTilt_degs);
    return h_views;
}


using GeoSource = std::variant<
SimGeoConfig,                        // 圆扫描几何，内部build
std::vector<SConeProjGeomVec>        // 外部预建几何
    > ;

    YK_INLINE std::vector<SConeProjGeomVec>
        resolve_geometry(const GeoSource& src)
    {
        return std::visit([](const auto& v) -> std::vector<SConeProjGeomVec> {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, SimGeoConfig>)
                return build_sim_geometry(v);
            else
                return v;
            }, src);
    }

// ================================================================
// 各种预设配置
// ================================================================

// 理想几何（无抖动）
YK_INLINE SimGeoConfig make_ideal_config(const SCBCTParams& p)
{
    SimGeoConfig cfg;
    cfg.SID = p.SID;
    cfg.IDD = p.SDD - p.SID;
    cfg.Na = p.iPAng;
    cfg.Nu = p.iPU;
    cfg.Nv = p.iPV;
    cfg.du = p.du_mm;
    cfg.dv = p.dv_mm;
    cfg.angle_list = p.angle_list;
    return cfg;
}

// 固定offset
YK_INLINE SimGeoConfig make_fixed_offset_config(
    const SCBCTParams& p,
    float3 src_offset,
    float3 det_offset,
    float3 detTilt_deg = make_float3(0, 0, 0))
{
    SimGeoConfig cfg = make_ideal_config(p);
    cfg.src_offsets = { src_offset };
    cfg.det_offsets = { det_offset };
    cfg.detTilt_degs = { detTilt_deg };
    return cfg;
}

// 周期抖动
YK_INLINE SimGeoConfig make_periodic_config(
    const SCBCTParams& p,
    float src_amp_x = 0.f,   // mm
    float src_amp_y = 0.f,   // mm，等效SID变化
    float src_amp_z = 0.f,
    float det_amp_x = 0.f,  // mm ,col offset
    float det_amp_y = 0.f,   // mm，等效IDD变化
    float det_amp_z = 0.f,  // mm, row offset
    float period_frames = 6.f)
{
    SimGeoConfig cfg = make_ideal_config(p);
    const int Na = p.iPAng;
    cfg.src_offsets.resize(Na);
    cfg.det_offsets.resize(Na);
    cfg.detTilt_degs.resize(Na);
    cfg.srcCRTilt_degs.resize(Na);
    for (int i = 0; i < Na; ++i) {
        const float phase = 2.f * CUDA_PI * i / period_frames;
        const float jit = std::sin(phase);
        cfg.src_offsets[i] = make_float3(src_amp_x * jit, src_amp_y * jit, src_amp_z * jit);
        cfg.det_offsets[i] = make_float3(det_amp_x * jit, det_amp_y * jit, det_amp_z * jit);
        cfg.detTilt_degs[i] = make_float3(0.f, 0.f, 0.f);
        cfg.srcCRTilt_degs[i] = make_float3(0.f, 0.f, 0.f);
    }
    return cfg;
}

// 随机抖动
YK_INLINE SimGeoConfig make_random_config(
    const SCBCTParams& p,
    float sigma_t = 0.25f,  // mm
    float sigma_r = 0.1f,   // deg
    uint32_t seed = 42)
{
    SimGeoConfig cfg = make_ideal_config(p);
    const int Na = p.iPAng;
    cfg.src_offsets.resize(Na);
    cfg.det_offsets.resize(Na);
    cfg.detTilt_degs.resize(Na);
    cfg.srcCRTilt_degs.resize(Na);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist_t(0.f, sigma_t);
    std::normal_distribution<float> dist_sid(0.f,1.f);
    std::normal_distribution<float> dist_idd(0.f, 1.f);
    std::normal_distribution<float> dist_r(0.f, sigma_r);
    for (int i = 0; i < Na; ++i) {
        cfg.src_offsets[i] = make_float3(dist_t(rng), dist_sid(rng), dist_t(rng));
        cfg.det_offsets[i] = make_float3(dist_t(rng), dist_idd(rng), dist_t(rng));
        cfg.detTilt_degs[i] = make_float3(dist_r(rng), dist_r(rng), 0.f);
        cfg.detTilt_degs[i] = make_float3(0, 0, 0.f);
        cfg.srcCRTilt_degs[i] = make_float3(0.f, 0.f, 0.f);
    }
    return cfg;
}



YK_INLINE void run_fp(
    const SCBCTParams& params,
    const GeoSource& geo_src,
    const std::string& sino_path,

    const std::string& phantom_path,
    cudaStream_t                    stream)
{
    const int Na = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;

    auto h_views = resolve_geometry(geo_src);

    Mem::MemoryController mc;
    auto h_vol = mc.allocateCpu3D<float>(Nx, Ny, Nz);
    if (!read_raw_float((datapath::test_data_dir + phantom_path).c_str(),
        h_vol.data(), 1LL * Nx * Ny * Nz)) {
        YK_LOGE("recon_raw_save.raw not found"); return;
    }
    auto d_vol = mc.allocateDevice3D<float>(Nx, Ny, Nz, 0);
    auto d_sino = mc.allocateDevice3D<float>(params.iPU, params.iPV, Na, 0);
    { auto borrow = mc.borrowCpu3D(h_vol.data(), Nx, Ny, Nz); mc.upload3D(d_vol, borrow); }
    h_vol.reset();

    GeometryContext geometry;
    ResourceContext resources;
    if (!geometry.initialize(params, h_views)) { YK_LOGE("FP geometry init failed"); return; }
    resources.attach(stream, 0);
    auto fp = makeForwardOperator(ETask::FP_Joseph);
    if (!fp->prepare(geometry, resources) ||
        !fp->apply(d_vol.data(), params, d_sino.data(), resources)) {
        YK_LOGE("FP operator execution failed");
        return;
    }
    fp->release();
    YK_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_sino(view_elems * Na);
    {
        auto borrow_sino = mc.borrowCpu3D(h_sino.data(), params.iPU, params.iPV, Na);
        mc.download3D(borrow_sino, d_sino);
    }
    write_raw_float((datapath::test_data_dir + sino_path).c_str(), h_sino.data(), h_sino.size());
    YK_LOGI("saved: {}", sino_path);
}

YK_INLINE void run_recon(
    const SCBCTParams& params,
    const GeoSource& geo_src,
    const std::string& sino_path,
    const std::string& vol_path,
    cudaStream_t                    stream)
{
    const int Na = params.iPAng;
    const int Nx = params.iVX, Ny = params.iVY, Nz = params.iVZ;
    const size_t view_elems = (size_t)params.iPU * params.iPV;
    const size_t vol_elems = (size_t)Nx * Ny * Nz;

    auto h_views_recon = resolve_geometry(geo_src);

    std::vector<float> h_sino(view_elems * Na);
    if (!read_raw_float((datapath::test_data_dir + sino_path).c_str(), h_sino)) {
        YK_LOGE("cannot read {}", sino_path); return;
    }

    Mem::MemoryController mc;
    auto d_vol_buf = mc.allocateDevice3D<float>(Nx, Ny, Nz, 0, false);
    {
        YK::Util::CudaTimer timer("recon", stream);
        // 外部 geometry 是唯一真源：在重建开始前一次性建立派生几何缓存，
        // 运行阶段只按视图顺序提交投影，不再传第二份角度或几何参数。
        FdkPipeline pipeline;
        if (!pipeline.prepareWithGeometry(params, h_views_recon, 64, stream)) {
            YK_LOGE("FDK geometry pipeline prepare failed");
            return;
        }
        const FdkProjectionBatch batch{ h_sino.data(), &h_views_recon, nullptr, Na };
        if (!pipeline.processBatch(batch, d_vol_buf.data(), true)) {
            YK_LOGE("FDK geometry pipeline execution failed");
            return;
        }
    }
    auto h_vol = mc.allocateCpu3D<float>(Nx, Ny, Nz, false);
    mc.download3D(h_vol, d_vol_buf);
    write_raw_float((datapath::test_data_dir + vol_path).c_str(), h_vol.cdata(), vol_elems);
    YK_LOGI("saved: {}", vol_path);
}

YK_INLINE void run_fp_and_recon(
    const SCBCTParams& params,
    const SimGeoConfig& geo_cfg,
    const SimGeoConfig& recon_cfg,
    const std::string& sino_path,
    const std::string& vol_path,
    cudaStream_t                    stream)
{
    run_fp(params, geo_cfg, sino_path, "recon_raw_save.raw", stream);
    run_recon(params, recon_cfg, sino_path, vol_path, stream);
}



YK_INLINE void generate_pcb_phantom(
    const std::string& save_path,
    int Nx = 512, int Ny = 100, int Nz = 512,
    float vox_x = 0.1f, float vox_y = 0.1f, float vox_z = 0.1f)
{
    constexpr float MU_AIR = 0.0f;
    constexpr float MU_PCB = 0.03f;
    constexpr float MU_COPPER = 0.6f;
    constexpr float MU_SOLDER = 0.4f;
    constexpr float MU_SILICON = 0.1f;


    std::vector<float> vol((size_t)Nz * Ny * Nx, MU_AIR);


    // 体素 -> 物理坐标（以原点为中心）
    auto px = [&](int i) { return (i - Nx * 0.5f + 0.5f) * vox_x; };
    auto py = [&](int j) { return (j - Ny * 0.5f + 0.5f) * vox_y; };
    auto pz = [&](int k) { return (k - Nz * 0.5f + 0.5f) * vox_z; };

    // 中心就是原点
    const float cx = 0.f;
    const float cy = 0.f;
    const float cz = 0.f;

    // [Nz][Ny][Nx]，Nx最快，Ny次之，Nz最慢
    auto idx = [&](int i, int j, int k) -> size_t {
        return (size_t)k * Ny * Nx + j * Nx + i;
        };

    // ----------------------------------------------------------------
    // 填充 lambdas
    // ----------------------------------------------------------------
    auto fill_box = [&](float x0, float x1,
        float y0, float y1,
        float z0, float z1, float mu)
        {
            for (int k = 0; k < Nz; ++k) {
                if (pz(k) < z0 || pz(k) > z1) continue;
                for (int j = 0; j < Ny; ++j) {
                    if (py(j) < y0 || py(j) > y1) continue;
                    for (int i = 0; i < Nx; ++i) {
                        if (px(i) >= x0 && px(i) <= x1)
                            vol[idx(i, j, k)] = mu;
                    }
                }
            }
        };

    // 圆柱沿Y轴（电路板法线方向）
    auto fill_cylinder_y = [&](float cx_, float cz_,
        float r, float y0, float y1, float mu)
        {
            for (int k = 0; k < Nz; ++k) {
                float dz = pz(k) - cz_;
                for (int j = 0; j < Ny; ++j) {
                    if (py(j) < y0 || py(j) > y1) continue;
                    for (int i = 0; i < Nx; ++i) {
                        float dx = px(i) - cx_;
                        if (dx * dx + dz * dz <= r * r)
                            vol[idx(i, j, k)] = mu;
                    }
                }
            }
        };

    // 圆环沿Y轴
    auto fill_ring_y = [&](float cx_, float cz_,
        float r_inner, float r_outer,
        float y0, float y1, float mu)
        {
            for (int k = 0; k < Nz; ++k) {
                float dz = pz(k) - cz_;
                for (int j = 0; j < Ny; ++j) {
                    if (py(j) < y0 || py(j) > y1) continue;
                    for (int i = 0; i < Nx; ++i) {
                        float dx = px(i) - cx_;
                        float r2 = dx * dx + dz * dz;
                        if (r2 >= r_inner * r_inner && r2 <= r_outer * r_outer)
                            vol[idx(i, j, k)] = mu;
                    }
                }
            }
        };

    // 球
    auto fill_sphere = [&](float cx_, float cy_, float cz_, float r, float mu)
        {
            for (int k = 0; k < Nz; ++k) {
                float dz = pz(k) - cz_;
                for (int j = 0; j < Ny; ++j) {
                    float dy = py(j) - cy_;
                    for (int i = 0; i < Nx; ++i) {
                        float dx = px(i) - cx_;
                        if (dx * dx + dy * dy + dz * dz <= r * r)
                            vol[idx(i, j, k)] = mu;
                    }
                }
            }
        };

    auto fill_groove = [&](float x0, float x1,
        float y0, float y1,
        float z0, float z1)
        {
            fill_box(x0, x1, y0, y1, z0, z1, MU_AIR);
        };

    // ----------------------------------------------------------------
    // PCB 模体，板面在XZ平面，法线方向为Y
    // 板厚2mm，居中放置在 cy
    // ----------------------------------------------------------------
    const float board_x0 = cx - 20.f, board_x1 = cx + 20.f;
    const float board_z0 = cz - 20.f, board_z1 = cz + 20.f;
    const float board_y0 = cy - 1.f, board_y1 = cy + 1.f;   // 2mm厚

    // 1. FR4 基板
    fill_box(board_x0, board_x1, board_y0, board_y1,
        board_z0, board_z1, MU_PCB);

    // 2. 铜焊盘 + 过孔（沿Y轴的圆柱/圆环）
    struct PadPos { float x, z; };
    std::vector<PadPos> pads = {
        { cx - 10.f, cz - 10.f },
        { cx + 10.f, cz - 10.f },
        { cx - 10.f, cz + 10.f },
        { cx + 10.f, cz + 10.f },
        { cx,        cz        },
    };
    for (auto& p : pads) {
        // 铜环（焊盘，在板面上下各0.1mm）
        fill_ring_y(p.x, p.z, 0.3f, 0.8f,
            board_y1, board_y1 + 0.1f, MU_COPPER);
        fill_ring_y(p.x, p.z, 0.3f, 0.8f,
            board_y0 - 0.1f, board_y0, MU_COPPER);
        // 过孔（贯穿基板）
        fill_cylinder_y(p.x, p.z, 0.3f, board_y0, board_y1, MU_COPPER);
    }

    // 3. 芯片封装（硅，贴在板面上）
    fill_box(cx - 5.f, cx + 5.f,
        board_y1, board_y1 + 2.f,
        cz - 4.f, cz + 4.f, MU_SILICON);

    // 4. BGA 焊球阵列（5x5，在芯片正下方贴板）
    for (int row = -2; row <= 2; ++row) {
        for (int col = -2; col <= 2; ++col) {
            fill_sphere(cx + col * 1.5f,
                board_y0 - 0.4f,
                cz + row * 1.5f,
                0.35f, MU_SOLDER);
        }
    }

    // 5. 铜走线（贴在板面，沿X和Z方向）
    fill_box(board_x0, board_x1,
        board_y1, board_y1 + 0.05f,
        cz - 0.2f, cz + 0.2f, MU_COPPER);   // X方向走线
    fill_box(cx - 0.2f, cx + 0.2f,
        board_y1, board_y1 + 0.05f,
        board_z0, board_z1, MU_COPPER);       // Z方向走线

    // 6. 边缘定位凹槽
    fill_groove(board_x0, board_x0 + 1.f,
        board_y0, board_y1,
        cz - 1.f, cz + 1.f);
    fill_groove(board_x1 - 1.f, board_x1,
        board_y0, board_y1,
        cz - 1.f, cz + 1.f);

    // 7. 定位通孔
    struct HolePos { float x, z; };
    std::vector<HolePos> holes = {
        { board_x0 + 2.f, board_z0 + 2.f },
        { board_x1 - 2.f, board_z0 + 2.f },
        { board_x0 + 2.f, board_z1 - 2.f },
        { board_x1 - 2.f, board_z1 - 2.f },
    };
    for (auto& h : holes)
        fill_cylinder_y(h.x, h.z, 0.5f, board_y0, board_y1, MU_AIR);

    write_raw_float(save_path.c_str(), vol.data(), vol.size());
    YK_LOGI("saved: {} ({}x{}x{})", save_path, Nx, Ny, Nz);
}
