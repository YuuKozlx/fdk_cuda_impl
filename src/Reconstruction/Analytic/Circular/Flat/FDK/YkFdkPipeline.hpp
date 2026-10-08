#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "Reconstruction/Analytic/Circular/Flat/FDK/YkBackProjectProcessor.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKGpuContext.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKFilterProcessor.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKParkerWeightProcessor.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKPreWeightProcessor.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKVecGeoDerived.hpp"
#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "YKCBCT/interface/YkSystemReconstruction.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"

namespace YK {

// 采集序列中的连续视图区间。当前公开 Session API 按 execute() 调用顺序
// 追加批次，调用者不需要也不能指定 offset；pipeline 保存并校验该区间。
struct FdkViewRange {
    int offset = 0;
    int count = 0;
};

// 校准 geometry 相对理想圆轨迹的统计，只用于报告近似程度。FDK 各阶段
// 始终消费从最终逐视图 geometry 派生的实际几何量。
struct FdkGeometryDiagnostics {
    float min_source_to_axis_mm = 0.f;
    float max_source_to_axis_mm = 0.f;
    float min_source_to_detector_mm = 0.f;
    float max_source_to_detector_mm = 0.f;
    float min_source_to_plane_mm = 0.f;
    float max_source_to_plane_mm = 0.f;
    float max_abs_source_axis_mm = 0.f;
    float max_detector_skew = 0.f;
    float max_abs_principal_u_pix = 0.f;
    float max_abs_principal_v_pix = 0.f;
    float min_principal_u_pix = 0.f;
    float max_principal_u_pix = 0.f;
    float min_principal_v_pix = 0.f;
    float max_principal_v_pix = 0.f;
    float min_relative_tilt_u_rad = 0.f;
    float max_relative_tilt_u_rad = 0.f;
    float min_relative_tilt_v_rad = 0.f;
    float max_relative_tilt_v_rad = 0.f;
    float min_relative_tilt_n_rad = 0.f;
    float max_relative_tilt_n_rad = 0.f;
    float max_normal_misalignment = 0.f;
    bool approximate = false;
};

// 一个在线 FDK 批次的主机侧输入。投影内存布局为 [view][v][u]。
// geometry 是执行阶段唯一的几何和角度来源；如果没有外部 geometry，调用方
// 可给出 circular_angles，仅用于构造同一份圆轨迹 geometry。
//
// projection 是异步 H2D 的源缓冲：调用 enqueueBatch()/processBatch()
// 成功返回后，它仍必须保持有效且内容不变，直到对应 FdkBatchFence
// 完成，或调用方同步 pipeline 使用的 CUDA stream。在线采集建议
// 使用项目 pinned host buffer，并以双缓冲 + fence 轮换。
struct FdkProjectionBatch {
    const float* projection = nullptr;
    const std::vector<SConeProjGeomVec>* geometry = nullptr;
    const float* circular_angles = nullptr;
    int count = 0;
};

// 一次异步 FDK 批次的完成栅栏。它拥有一个禁用计时的 CUDA event，
// 可重复记录，但在再次用于同一块主机输入缓冲前，调用方必须先 wait()。
// fence 记录在本批次最后一个 kernel 之后，因此它同时保证 H2D 已结束
// 且本批重建已累加到输出体。
class FdkBatchFence {
public:
    FdkBatchFence() = default;
    ~FdkBatchFence() { release_(); }

    FdkBatchFence(const FdkBatchFence&) = delete;
    FdkBatchFence& operator=(const FdkBatchFence&) = delete;

    FdkBatchFence(FdkBatchFence&& other) noexcept
        : event_(std::exchange(other.event_, nullptr)), recorded_(other.recorded_)
    {
        other.recorded_ = false;
    }

    FdkBatchFence& operator=(FdkBatchFence&& other) noexcept
    {
        if (this != &other) {
            release_();
            event_ = std::exchange(other.event_, nullptr);
            recorded_ = other.recorded_;
            other.recorded_ = false;
        }
        return *this;
    }

    bool valid() const { return event_ != nullptr && recorded_; }

    bool wait() const
    {
        if (!valid()) return true;
        const cudaError_t status = cudaEventSynchronize(event_);
        if (status != cudaSuccess) {
            YK_LOGE("[FdkBatchFence] 等待批次完成失败：{}", cudaGetErrorString(status));
            return false;
        }
        return true;
    }

    // 非阻塞查询。未记录过的 fence 不代表任何批次，故返回 false。
    bool ready() const
    {
        if (!valid()) return false;
        const cudaError_t status = cudaEventQuery(event_);
        if (status == cudaSuccess) return true;
        if (status == cudaErrorNotReady) return false;
        YK_LOGE("[FdkBatchFence] 查询批次状态失败：{}", cudaGetErrorString(status));
        return false;
    }

private:
    friend class FdkPipeline;

    bool record_(cudaStream_t stream)
    {
        if (!event_) {
            const cudaError_t status = cudaEventCreateWithFlags(
                &event_, cudaEventDisableTiming);
            if (status != cudaSuccess) {
                YK_LOGE("[FdkBatchFence] 创建批次 event 失败：{}",
                    cudaGetErrorString(status));
                return false;
            }
        }
        const cudaError_t status = cudaEventRecord(event_, stream);
        if (status != cudaSuccess) {
            YK_LOGE("[FdkBatchFence] 记录批次 event 失败：{}",
                cudaGetErrorString(status));
            recorded_ = false;
            return false;
        }
        recorded_ = true;
        return true;
    }

    void release_() noexcept
    {
        if (event_) cudaEventDestroy(event_);
        event_ = nullptr;
        recorded_ = false;
    }

    cudaEvent_t event_ = nullptr;
    bool recorded_ = false;
};

// CUDA stage 实际消费的一个 chunk。它只引用已经准备好的数据：不保存批次
// 生命周期，不拥有内存，也不携带初始化配置。
struct FdkChunkView {
    const SConeProjGeomVec* d_geometry = nullptr;
    const SFDKGeoParamPerView* d_derived_geometry = nullptr;
    const SFDKGeoParamPerView* h_derived_geometry = nullptr;
    const SConeProjGeomVec* h_geometry = nullptr;
    int count = 0;
};

// 最近一次完成的 chunk 的只读设备视图。它不是 dump 回调：pipeline 不会
// 同步、不回传主机数据、更不会决定文件格式。诊断程序可在同一 stream 上
// 自主拷贝这些缓冲，或以事件与后续工作建立同步关系。
//
// 指针归 pipeline 的工作区所有；下一次 processBatch()、reset() 或 release()
// 后该视图失效。该接口用于测试、性能分析和独立的可视化工具，不参与重建计算。
struct FdkStageView {
    int first_view = 0;
    int count = 0;
    const float* d_input = nullptr;        // 原始投影上传后的 chunk
    const float* d_preweighted = nullptr;  // 预加权（及 Parker 加权）后
    const float* d_filtered = nullptr;     // 滤波后，反投影使用的数据
    cudaStream_t stream = nullptr;

    bool valid() const { return count > 0 && d_input && d_preweighted && d_filtered; }
};

// 将在线采集状态从 CUDA 执行核心中剥离。螺旋重建以后可以有自己的调度器，
// 但仍可复用“连续、有上限、完成后关闭”的基本状态语义。
class FdkOnlineState {
public:
    void prepare(int total_views) { total_views_ = total_views; reset(); }
    void reset() { received_views_ = 0; finished_ = false; }

    // 先只校验并生成下一个区间；CUDA 工作提交完成后再 commit，避免同步
    // 参数错误让在线位置提前推进。
    bool nextRange(int count, FdkViewRange& range) const
    {
        if (finished_) {
            YK_LOGE("[FdkPipeline] 已完成采集，拒绝继续提交批次。");
            return false;
        }
        if (count <= 0 || count > total_views_ - received_views_) {
            YK_LOGE("[FdkPipeline] 批次数 {} 超过剩余容量 {}。",
                count, total_views_ - received_views_);
            return false;
        }
        range = { received_views_, count };
        return true;
    }

    void commit(const FdkViewRange& range)
    {
        // range 只能来自紧邻的 nextRange()；调用方是 pipeline 内部，故此处
        // 不重复执行对外输入校验。
        received_views_ += range.count;
        finished_ = received_views_ == total_views_;
    }

    int receivedViews() const { return received_views_; }
    bool complete() const { return finished_; }

private:
    int total_views_ = 0;
    int received_views_ = 0;
    bool finished_ = false;
};

// 普通圆轨迹 FDK 的唯一主流程。
//
// 参数流：prepare() 接收一次性的扫描/体数据/滤波配置，并创建固定工作区。
// 若完整逐视图 geometry 已知，则应传入 prepareWithGeometry()，在此一次性
// 预计算全局 dtheta 等派生量；这是任意几何 FDK 的首选路径。
// 数据流：processBatch() 只按已知范围取出预计算结果，切分为 chunk 后依次
// 执行“上传 -> 预加权 -> Parker 加权 -> 滤波 -> 反投影”。
// 各 stage 只接收强类型配置或 FdkChunkView，不在成员中暂存上一次调用的 chunk
// 数据，避免隐式状态跨批次泄漏。
class FdkPipeline {
public:
    // 新系统级入口：宏观 Flat CBCT 几何和 FDK 配置一次传入。扫描范围从
    // total_views/views_per_turn 派生，Parker 不再依赖调用方重复填写旧参数。
    bool prepare(const SSystemConfig& system,
        const SReconstructionSpec& reconstruction,
        int requested_chunk, cudaStream_t stream, int device_id = 0)
    {
        if (system.detector != EDetectorKind::Flat ||
            system.trajectory != ETrajectoryKind::Circular ||
            reconstruction.pipeline != EPipeline::FDK) {
            YK_LOGE("[FdkPipeline] 系统级入口只接受 FDK 重建请求。");
            return false;
        }
        std::vector<SConeProjGeomVec> geometry;
        SVolGeom volume{};
        std::vector<SCylConeProjGeomVec> unused;
        if (!buildSystemGeometry(system, geometry, unused, volume)) {
            YK_LOGE("[FdkPipeline] Flat 系统 geometry 构造失败。");
            return false;
        }

        SReconstructionParams params{};
        if (!buildParams_(system, reconstruction, volume, params)) return false;
        if (params.scan.short_scan &&
            !validateParkerCoverage_(geometry, system.flat_detector.channels,
                system.flat_detector.rows, params.scan.range_rad))
            return false;
        return prepareWithGeometry(params, geometry, requested_chunk,
            stream, device_id);
    }

    bool prepare(const SReconstructionParams& params, int requested_chunk,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!validateStatic_(params, requested_chunk, stream)) return false;

        params_ = params;
        chunk_size_ = std::min(requested_chunk, kMaxChunkAng);
        stream_ = stream;

        const SProjDims chunk_dims{ params_.scan.Nu, params_.scan.Nv, chunk_size_ };
        // 当前反投影为每个视图建立 pitch2D texture。除行 pitch 外，每个
        // slice 的起始地址也必须满足 textureAlignment；否则 CUDA runtime
        // 会在创建第二个 texture 时返回 invalid argument。这里提前拒绝，
        // 使 DLL 初始化得到可诊断的 false，而不是由底层检查宏终止进程。
        cudaDeviceProp device_property{};
        const cudaError_t property_status = cudaGetDeviceProperties(
            &device_property, device_id);
        const size_t row_bytes = static_cast<size_t>(params_.scan.Nu) * sizeof(float);
        const size_t view_bytes = row_bytes * params_.scan.Nv;
        if (property_status != cudaSuccess ||
            row_bytes % device_property.texturePitchAlignment != 0 ||
            view_bytes % device_property.textureAlignment != 0) {
            YK_LOGE("[FdkPipeline] 探测器尺寸 {}x{} 不满足 pitch2D texture "
                "对齐要求（pitch={} bytes, slice={} bytes, 要求 {}/{}）。",
                params_.scan.Nu, params_.scan.Nv, row_bytes, view_bytes,
                device_property.texturePitchAlignment,
                device_property.textureAlignment);
            release();
            return false;
        }
        gpu_.init(chunk_dims, params_.scan.totalViews, stream_, device_id);

        PreweightConfig preweight_config{};
        preweight_config.dims = chunk_dims;
        if (!preweight_.prepare(preweight_config)) { release(); return false; }

        FdkFilterConfig filter_config{};
        filter_config.dims = chunk_dims;
        filter_config.desc = params_.reconstruction.filter;
        filter_config.stream = stream_;
        if (!filter_.prepare(filter_config)) { release(); return false; }

        if (params_.scan.short_scan) {
            ParkerWeightConfig parker_config{};
            parker_config.dims = chunk_dims;
            parker_config.iPAnglesTotal = params_.scan.totalViews;
            parker_config.fScanRangeRad = params_.scan.range_rad;
            parker_config.fStartAngleRad = params_.scan.start_angle_rad;
            parker_config.nDirSign = params_.scan.direction;
            if (!parker_.prepare(parker_config)) { release(); return false; }
        }

        BpConfig bp_config{};
        bp_config.vol_geom = SVolGeom::make_centered(params_.volume.Nx, params_.volume.Ny,
            params_.volume.Nz, params_.volume.voxelX_mm, params_.volume.voxelY_mm, params_.volume.voxelZ_mm);
        bp_config.vol_geom.center = make_float3(params_.volume.centerX_mm,
            params_.volume.centerY_mm, params_.volume.centerZ_mm);
        bp_config.use_precomputed = true;
        bp_config.max_chunk_views = chunk_size_;
        if (!backproject_.prepare(bp_config)) { release(); return false; }

        state_.prepare(params_.scan.totalViews);
        prepared_ = true;
        return true;
    }

    // 任意几何的首选初始化入口。geometry 的顺序就是采集顺序，长度必须等于
    // iPAngTotal；派生几何在此一次性生成，保证跨 batch/chunk 的 dtheta 连续。
    bool prepareWithGeometry(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry, int requested_chunk,
        cudaStream_t stream, int device_id = 0)
    {
        if (static_cast<int>(geometry.size()) != params.scan.totalViews) {
            YK_LOGE("[FdkPipeline] 完整 geometry 数量 {} 与 iPAngTotal {} 不一致。",
                geometry.size(), params.scan.totalViews);
            return false;
        }
        std::vector<SFDKGeoParamPerView> derived;
        if (!GeoDerivedManagerVec{}.build_geo_params(params.scan.Nu, params.scan.Nv,
            params.scan.range_rad, geometry, derived)) {
            YK_LOGE("[FdkPipeline] 完整 geometry 的派生参数预计算失败。");
            return false;
        }
        FdkGeometryDiagnostics diagnostics{};
        if (!validateApproximateFdkGeometry_(geometry, derived, diagnostics)) return false;
        if (!prepare(params, requested_chunk, stream, device_id)) return false;
        static_geometry_ = geometry;
        static_derived_geometry_ = std::move(derived);
        geometry_diagnostics_ = diagnostics;
        return true;
    }

    // 圆轨迹只是 geometry 的一种来源。完整角度已知时同样在初始化阶段构造
    // 全量 geometry，后续与任意外部 geometry 共用同一预计算和执行路径。
    bool prepareWithAngles(const SReconstructionParams& params,
        const std::vector<float>& angles, int requested_chunk,
        cudaStream_t stream, int device_id = 0)
    {
        if (!validateCircularBuilderInput_(params)) return false;
        if (static_cast<int>(angles.size()) != params.scan.totalViews) {
            YK_LOGE("[FdkPipeline] 完整角度数量 {} 与 iPAngTotal {} 不一致。",
                angles.size(), params.scan.totalViews);
            return false;
        }
        std::vector<SConeProjGeomVec> geometry;
        buildCircularGeometry_(params, angles, geometry);
        return prepareWithGeometry(params, geometry, requested_chunk, stream, device_id);
    }

    // 高性能异步入口。返回 true 只表示本批 CUDA 工作已成功提交，
    // 不表示执行已结束。completion 非空时，在本批最后记录 event；
    // 调用方可在复用 batch.projection 或消费本批输出前 wait()。
    bool enqueueBatch(const FdkProjectionBatch& batch, float* d_volume,
        bool clear_output, FdkBatchFence* completion = nullptr)
    {
        if (!prepared_) {
            YK_LOGE("[FdkPipeline] processBatch 在 prepare 前调用。");
            return false;
        }
        if (!batch.projection || !d_volume || batch.count <= 0) {
            YK_LOGE("[FdkPipeline] 批次必须提供投影、输出体和正视图数。");
            return false;
        }
        if (clear_output && state_.receivedViews() != 0) {
            YK_LOGE("[FdkPipeline] clear_output 仅允许用于 reset 后的首批。");
            return false;
        }

        FdkViewRange range{};
        if (!state_.nextRange(batch.count, range)) return false;

        // 完整静态几何优先；动态 geometry 作为次优兼容路径；两者均未提供时
        // 才由角度构造圆轨迹几何。
        const bool use_static_geometry = !static_geometry_.empty();
        // 动态 geometry/angles 是兼容路径，内部 host vector 会被下一批覆盖。
        // 在改写前等待上一批完成，保证其异步几何上传和所有主机几何只读访问
        // 已经结束。正式在线路径应优先在 prepare 阶段提供完整静态 geometry。
        if (!use_static_geometry && !dynamic_geometry_fence_.wait()) return false;
        if (!use_static_geometry && batch.geometry) {
            if (static_cast<int>(batch.geometry->size()) != batch.count) {
                YK_LOGE("[FdkPipeline] 当前批次 geometry 数量与 count 不一致。");
                return false;
            }
            if (!buildDerivedGeometry_(*batch.geometry)) return false;
        }
        else if (!use_static_geometry) {
            if (!batch.circular_angles) {
                YK_LOGE("[FdkPipeline] 未提供 geometry 时必须提供圆轨迹角度数组。");
                return false;
            }
            if (!validateCircularBuilderInput_(params_)) return false;
            buildCircularGeometry_(params_, std::vector<float>(batch.circular_angles,
                batch.circular_angles + batch.count), host_geometry_);
            if (!buildDerivedGeometry_(host_geometry_)) return false;
        }

        if (clear_output) {
            const size_t volume_bytes = static_cast<size_t>(params_.volume.Nx) * params_.volume.Ny *
                params_.volume.Nz * sizeof(float);
            YK_CUDA_CHECK(cudaMemsetAsync(d_volume, 0, volume_bytes, stream_));
        }

        const size_t view_elements = static_cast<size_t>(params_.scan.Nu) * params_.scan.Nv;
        for (int base = 0; base < batch.count; base += chunk_size_) {
            const int count = std::min(chunk_size_, batch.count - base);
            const int global_offset = range.offset + base;
            const int source_offset = use_static_geometry ? global_offset : base;
            const SConeProjGeomVec* h_geometry = (use_static_geometry
                ? static_geometry_.data() : host_geometry_.data()) + source_offset;
            const SFDKGeoParamPerView* h_derived = (use_static_geometry
                ? static_derived_geometry_.data() : host_derived_geometry_.data()) + source_offset;
            gpu_.uploadGeoIncremental(h_geometry, h_derived, global_offset, count, stream_);
            const FdkChunkView chunk{
                gpu_.geo.d_geo() + global_offset,
                gpu_.geo.d_gv() + global_offset,
                h_derived,
                h_geometry,
                count
            };
            gpu_.geo.uploadCoeffsChunk(gpu_.geo.d_coeffs() + global_offset, count, stream_);
            gpu_.proj.uploadProjChunk(batch.projection + base * view_elements, count, stream_);

            if (!preweight_.apply(gpu_.proj.chunk_in.data(), gpu_.proj.chunk_pw.data(),
                { chunk.d_geometry, chunk.d_derived_geometry, chunk.count }, stream_)) return false;
            if (params_.scan.short_scan) {
                if (!parker_.apply(gpu_.proj.chunk_pw.data(),
                    { chunk.h_geometry, chunk.d_geometry,
                      chunk.d_derived_geometry, chunk.count }, stream_)) return false;
            }
            if (!filter_.apply(gpu_.proj.chunk_pw.data(), gpu_.proj.chunk_flt.data(),
                { chunk.h_derived_geometry, chunk.count }, stream_)) return false;
            if (!backproject_.apply(gpu_.proj.d_texObjs(), d_volume,
                { chunk.d_geometry, chunk.d_derived_geometry, chunk.count }, stream_)) return false;

            // 只记录设备工作区的观察视图；不进行回调、同步或复制。
            last_stage_ = { global_offset, count, gpu_.proj.chunk_in.data(),
                gpu_.proj.chunk_pw.data(), gpu_.proj.chunk_flt.data(), stream_ };
        }
        // event 必须在 commit 前成功提交；否则调用方无法证明主机
        // 输入的异步生命期，不应提前推进在线位置。
        if (!use_static_geometry && !dynamic_geometry_fence_.record_(stream_))
            return false;
        if (completion && !completion->record_(stream_)) return false;

        // 所有 CUDA 工作均已成功提交到本 session 的同一 stream；在此之后该
        // batch 才计入在线位置。异步 CUDA 执行错误由 fence/stream 同步时报告。
        state_.commit(range);
        return true;
    }

    // 兼容原有调用：仍是异步提交。如果调用方会立即复用主机输入，
    // 应改用 enqueueBatch(..., &fence) 或 processBatchSync()。
    bool processBatch(const FdkProjectionBatch& batch, float* d_volume,
        bool clear_output)
    {
        return enqueueBatch(batch, d_volume, clear_output, nullptr);
    }

    // 易用的同步入口：返回前保证主机输入可复用、本批输出可消费，
    // 并将异步 CUDA 错误转换为 false。在线高吞吐路径不应每批使用它。
    bool processBatchSync(const FdkProjectionBatch& batch, float* d_volume,
        bool clear_output)
    {
        if (!enqueueBatch(batch, d_volume, clear_output, nullptr)) return false;
        const cudaError_t status = cudaStreamSynchronize(stream_);
        if (status != cudaSuccess) {
            YK_LOGE("[FdkPipeline] 同步批次执行失败：{}", cudaGetErrorString(status));
            return false;
        }
        return true;
    }

    void reset()
    {
        // 工作区与 stage 配置可复用；仅清除在线位置，下一批重新从 offset 0 写入。
        state_.reset();
        last_stage_ = {};
    }

    void release()
    {
        // 动态几何的内部主机数组可能仍被异步 CUDA 工作引用，先闭合其事件；
        // 随后等待绑定 stream，保证所有静态/动态路径的工作区都不会在最后一批
        // kernel 尚未完成时被释放。正常 processBatch() 主路径不会因此同步。
        dynamic_geometry_fence_.wait();
        dynamic_geometry_fence_ = {};
        if (stream_)
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        preweight_.release();
        parker_.release();
        filter_.release();
        backproject_.release();
        gpu_.release();
        params_ = {};
        static_geometry_.clear();
        static_derived_geometry_.clear();
        host_geometry_.clear();
        host_derived_geometry_.clear();
        geometry_diagnostics_ = {};
        last_stage_ = {};
        chunk_size_ = 0;
        stream_ = nullptr;
        state_.prepare(0);
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }
    int receivedViews() const { return state_.receivedViews(); }
    bool complete() const { return state_.complete(); }
    const FdkStageView& lastStage() const { return last_stage_; }
    const FdkGeometryDiagnostics& geometryDiagnostics() const
    { return geometry_diagnostics_; }

private:
    static EFilterKernel mapFilter_(EFdkFilter filter)
    {
        switch (filter) {
        case EFdkFilter::None: return EFilterKernel::None;
        case EFdkFilter::RamLak: return EFilterKernel::RamLak;
        case EFdkFilter::SheppLogan: return EFilterKernel::SheppLogan;
        case EFdkFilter::Cosine: return EFilterKernel::Cosine;
        case EFdkFilter::Hann: return EFilterKernel::Hann;
        case EFdkFilter::Hamming: return EFilterKernel::Hamming;
        case EFdkFilter::Blackman: return EFilterKernel::Blackman;
        case EFdkFilter::Butterworth: return EFilterKernel::Butterworth;
        case EFdkFilter::Kaiser: return EFilterKernel::Kaiser;
        case EFdkFilter::Tukey: return EFilterKernel::Tukey;
        }
        return EFilterKernel::RamLak;
    }

    static bool buildParams_(const SSystemConfig& system,
        const SReconstructionSpec& reconstruction,
        const SVolGeom& volume, SReconstructionParams& params)
    {
        const auto& scan = system.circular;
        const auto& detector = system.flat_detector;
        const float range = regularScanRangeRad(scan);
        if (!std::isfinite(range) || range <= 0.f) return false;
        params = {};
        params.scan.Nu = detector.channels;
        params.scan.Nv = detector.rows;
        params.scan.NAng = scan.total_views;
        params.scan.totalViews = scan.total_views;
        params.scan.du_mm = detector.channel_size_mm;
        params.scan.dv_mm = detector.row_size_mm;
        params.scan.offsetU_mm = detector.pose.offset_unv_mm.x;
        params.scan.offsetV_mm = detector.pose.offset_unv_mm.z;
        params.scan.tiltU_rad = detector.pose.tilt_u_rad;
        params.scan.tiltV_rad = detector.pose.tilt_v_rad;
        params.scan.tiltN_rad = detector.pose.tilt_n_rad;
        params.scan.sourceOffsetX_mm = scan.source_offset_mm.x;
        params.scan.sourceOffsetY_mm = scan.source_offset_mm.y;
        params.scan.sourceOffsetZ_mm = scan.source_offset_mm.z;
        params.scan.range_rad = range;
        params.scan.start_angle_rad = scan.start_angle_rad;
        params.scan.direction = scan.rotation_direction;
        params.scan.sid_mm = scan.sid_mm;
        params.scan.sdd_mm = scan.sdd_mm;
        params.scan.short_scan = resolveParkerEnabled(system, reconstruction);
        params.volume.Nx = volume.Nx; params.volume.Ny = volume.Ny;
        params.volume.Nz = volume.Nz;
        params.volume.voxelX_mm = volume.vox_x;
        params.volume.voxelY_mm = volume.vox_y;
        params.volume.voxelZ_mm = volume.vox_z;
        params.volume.centerX_mm = volume.center.x;
        params.volume.centerY_mm = volume.center.y;
        params.volume.centerZ_mm = volume.center.z;
        const auto& input = reconstruction.fdk;
        auto& filter = params.reconstruction.filter;
        filter.kind = mapFilter_(input.filter);
        filter.source = EWeightsBuildSource::AnalyticFreq;
        filter.cutoff = input.cutoff;
        filter.gain = input.gain;
        filter.order = input.butterworth_order;
        filter.beta = input.kaiser_beta;
        filter.tukey_alpha = input.tukey_alpha;
        return true;
    }

    static bool validateParkerCoverage_(
        const std::vector<SConeProjGeomVec>& geometry, int detector_u,
        int detector_v, float range_rad)
    {
        if (geometry.empty() || range_rad >= 2.f * CUDA_PI - 1e-5f) {
            YK_LOGE("[FdkPipeline] Parker 只适用于小于 2π 的有效短扫描。");
            return false;
        }
        float max_fan = 0.f;
        for (const auto& geo : geometry) {
            const float cu = 0.5f * static_cast<float>(detector_u - 1);
            const float cv = 0.5f * static_cast<float>(detector_v - 1);
            const float3 center = make_float3(
                geo.detS.x + cu * geo.detU.x + cv * geo.detV.x,
                geo.detS.y + cu * geo.detU.y + cv * geo.detV.y,
                geo.detS.z + cu * geo.detU.z + cv * geo.detV.z);
            const float2 principal = make_float2(
                center.x - geo.src.x, center.y - geo.src.y);
            for (float u : {0.f, static_cast<float>(detector_u - 1)}) {
                const float3 edge = make_float3(
                    geo.detS.x + u * geo.detU.x + cv * geo.detV.x,
                    geo.detS.y + u * geo.detU.y + cv * geo.detV.y,
                    geo.detS.z + u * geo.detU.z + cv * geo.detV.z);
                const float2 ray = make_float2(
                    edge.x - geo.src.x, edge.y - geo.src.y);
                const float gamma = std::atan2(ray.x * principal.y -
                    ray.y * principal.x, ray.x * principal.x +
                    ray.y * principal.y);
                max_fan = std::max(max_fan, std::fabs(gamma));
            }
        }
        const float required = CUDA_PI + 2.f * max_fan;
        if (range_rad + 1e-5f < required) {
            YK_LOGE("[FdkPipeline] Parker 短扫范围不足：实际 {} rad，至少需要 {} rad（π+2Γ）。",
                range_rad, required);
            return false;
        }
        return true;
    }

    bool validateStatic_(const SReconstructionParams& p, int requested_chunk, cudaStream_t stream) const
    {
        if (!stream || requested_chunk <= 0 || p.scan.totalViews <= 0 || p.scan.Nu <= 0 ||
            p.scan.Nv <= 0 || p.volume.Nx <= 0 || p.volume.Ny <= 0 || p.volume.Nz <= 0 ||
            !std::isfinite(p.scan.range_rad) ||
            p.scan.range_rad <= 0.f ||
            (p.scan.direction != 1 && p.scan.direction != -1)) {
            YK_LOGE("[FdkPipeline] 静态 FDK 配置无效。");
            return false;
        }
        return true;
    }

    // 标称 SID/SDD 和像素尺寸只属于默认圆轨迹构造器。外部逐视图 geometry
    // 已经包含实际源点和探测器像素基向量，不应再受这些标称字段约束。
    static bool validateCircularBuilderInput_(const SReconstructionParams& p)
    {
        if (!std::isfinite(p.scan.du_mm) || !std::isfinite(p.scan.dv_mm) ||
            !std::isfinite(p.scan.sid_mm) || !std::isfinite(p.scan.sdd_mm) ||
            p.scan.du_mm <= 0.f || p.scan.dv_mm <= 0.f || p.scan.sid_mm <= 0.f || p.scan.sdd_mm <= p.scan.sid_mm) {
            YK_LOGE("[FdkPipeline] 默认圆轨迹构造器的标称 SID/SDD 或像素尺寸无效。");
            return false;
        }
        return true;
    }

    // 近似 FDK 接受校准后的非理想圆 geometry。这里仅拒绝数学上不可计算的
    // 输入；SOD/SDD、源轴向漂移和探测器姿态偏离通过 diagnostics 报告。
    bool validateApproximateFdkGeometry_(
        const std::vector<SConeProjGeomVec>& geometry,
        const std::vector<SFDKGeoParamPerView>& derived,
        FdkGeometryDiagnostics& diagnostics) const
    {
        constexpr float kRelativeTolerance = 1e-3f;
        constexpr float kAngleTolerance = 1e-5f;
        const auto length = [](const float4& v) {
            return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        };
        const auto dot = [](const float4& a, const float4& b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        };
        if (geometry.empty() || derived.size() != geometry.size()) return false;
        diagnostics.min_source_to_axis_mm = diagnostics.max_source_to_axis_mm =
            derived.front().source_to_axis_mm;
        diagnostics.min_source_to_detector_mm = diagnostics.max_source_to_detector_mm =
            derived.front().source_to_radial_detector_mm;
        diagnostics.min_source_to_plane_mm = diagnostics.max_source_to_plane_mm =
            derived.front().source_to_detector_plane_mm;
        diagnostics.min_principal_u_pix = diagnostics.max_principal_u_pix =
            derived.front().offsetU_pix;
        diagnostics.min_principal_v_pix = diagnostics.max_principal_v_pix =
            derived.front().offsetV_pix;
        const float du0 = derived.front().du_mm;
        const float dv0 = derived.front().dv_mm;
        if (du0 <= 0.f || dv0 <= 0.f) {
            YK_LOGE("[FdkPipeline] geometry 含有无效探测器像素轴。");
            return false;
        }
        const auto normalize = [&](const float4& value) {
            const float inv_length = 1.f / length(value);
            return make_float4(value.x * inv_length, value.y * inv_length,
                value.z * inv_length, 0.f);
        };
        const auto cross = [](const float4& a, const float4& b) {
            return make_float4(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x, 0.f);
        };
        const auto relativeDetectorTilt = [&](const SConeProjGeomVec& view,
            const SFDKGeoParamPerView& actual) {
            // The ideal detector follows the offset source: N points from the
            // detector toward the source, V follows the rotation axis, and
            // U x V = N. Decompose the measured detector basis using the same
            // intrinsic U/V/N rotation order as the geometry builder.
            const float4 ideal_n = make_float4(-actual.radial_ray.x,
                -actual.radial_ray.y, -actual.radial_ray.z, 0.f);
            const float4 axis = make_float4(0.f, 0.f, 1.f, 0.f);
            const float4 ideal_u = normalize(cross(axis, ideal_n));
            const float4 ideal_v = normalize(cross(ideal_n, ideal_u));
            const float4 measured_u = normalize(view.detU);
            const float4 measured_v = normalize(view.detV);
            float4 measured_n = normalize(cross(measured_u, measured_v));
            if (dot(measured_n, ideal_n) < 0.f) {
                measured_n.x = -measured_n.x;
                measured_n.y = -measured_n.y;
                measured_n.z = -measured_n.z;
            }

            const float r02 = std::clamp(dot(ideal_u, measured_n), -1.f, 1.f);
            const float tilt_v = std::asin(r02);
            const float tilt_u = std::atan2(-dot(ideal_v, measured_n),
                dot(ideal_n, measured_n));
            const float tilt_n = std::atan2(-dot(ideal_u, measured_v),
                dot(ideal_u, measured_u));
            return make_float3(tilt_u, tilt_v, tilt_n);
        };
        bool first_tilt = true;
        for (size_t i = 0; i < geometry.size(); ++i) {
            const auto& view = geometry[i];
            const auto& actual = derived[i];
            const float du = length(view.detU);
            const float dv = length(view.detV);
            if (du <= 0.f || dv <= 0.f || actual.source_to_axis_mm <= 0.f ||
                actual.source_to_radial_detector_mm <= 0.f ||
                actual.source_to_detector_plane_mm <= 0.f ||
                !std::isfinite(view.angle.x)) {
                YK_LOGE("[FdkPipeline] geometry 含有不可计算的逐视图参数。");
                return false;
            }
            diagnostics.min_source_to_axis_mm = std::min(
                diagnostics.min_source_to_axis_mm, actual.source_to_axis_mm);
            diagnostics.max_source_to_axis_mm = std::max(
                diagnostics.max_source_to_axis_mm, actual.source_to_axis_mm);
            diagnostics.min_source_to_detector_mm = std::min(
                diagnostics.min_source_to_detector_mm,
                actual.source_to_radial_detector_mm);
            diagnostics.max_source_to_detector_mm = std::max(
                diagnostics.max_source_to_detector_mm,
                actual.source_to_radial_detector_mm);
            diagnostics.min_source_to_plane_mm = std::min(
                diagnostics.min_source_to_plane_mm,
                actual.source_to_detector_plane_mm);
            diagnostics.max_source_to_plane_mm = std::max(
                diagnostics.max_source_to_plane_mm,
                actual.source_to_detector_plane_mm);
            diagnostics.max_abs_source_axis_mm = std::max(
                diagnostics.max_abs_source_axis_mm, std::abs(view.src.z));
            diagnostics.max_detector_skew = std::max(diagnostics.max_detector_skew,
                std::abs(dot(view.detU, view.detV)) / (du * dv));
            diagnostics.max_abs_principal_u_pix = std::max(
                diagnostics.max_abs_principal_u_pix, std::abs(actual.offsetU_pix));
            diagnostics.max_abs_principal_v_pix = std::max(
                diagnostics.max_abs_principal_v_pix, std::abs(actual.offsetV_pix));
            diagnostics.min_principal_u_pix = std::min(
                diagnostics.min_principal_u_pix, actual.offsetU_pix);
            diagnostics.max_principal_u_pix = std::max(
                diagnostics.max_principal_u_pix, actual.offsetU_pix);
            diagnostics.min_principal_v_pix = std::min(
                diagnostics.min_principal_v_pix, actual.offsetV_pix);
            diagnostics.max_principal_v_pix = std::max(
                diagnostics.max_principal_v_pix, actual.offsetV_pix);
            const float3 tilt = relativeDetectorTilt(view, actual);
            if (first_tilt) {
                diagnostics.min_relative_tilt_u_rad =
                    diagnostics.max_relative_tilt_u_rad = tilt.x;
                diagnostics.min_relative_tilt_v_rad =
                    diagnostics.max_relative_tilt_v_rad = tilt.y;
                diagnostics.min_relative_tilt_n_rad =
                    diagnostics.max_relative_tilt_n_rad = tilt.z;
                first_tilt = false;
            }
            else {
                diagnostics.min_relative_tilt_u_rad = std::min(
                    diagnostics.min_relative_tilt_u_rad, tilt.x);
                diagnostics.max_relative_tilt_u_rad = std::max(
                    diagnostics.max_relative_tilt_u_rad, tilt.x);
                diagnostics.min_relative_tilt_v_rad = std::min(
                    diagnostics.min_relative_tilt_v_rad, tilt.y);
                diagnostics.max_relative_tilt_v_rad = std::max(
                    diagnostics.max_relative_tilt_v_rad, tilt.y);
                diagnostics.min_relative_tilt_n_rad = std::min(
                    diagnostics.min_relative_tilt_n_rad, tilt.z);
                diagnostics.max_relative_tilt_n_rad = std::max(
                    diagnostics.max_relative_tilt_n_rad, tilt.z);
            }
            const float normalAlignment = std::abs(
                actual.radial_ray.x * actual.det_n.x +
                actual.radial_ray.y * actual.det_n.y +
                actual.radial_ray.z * actual.det_n.z);
            diagnostics.max_normal_misalignment = std::max(
                diagnostics.max_normal_misalignment, 1.f - normalAlignment);
            diagnostics.approximate = diagnostics.approximate ||
                std::abs(du - du0) > du0 * kRelativeTolerance ||
                std::abs(dv - dv0) > dv0 * kRelativeTolerance;
        }
        // 外部 geometry 的 angle.x 是唯一角度来源。相邻角度重复将使 dtheta
        // 不可定义，提前拒绝可避免后续滤波/反投影出现 NaN。
        for (size_t i = 1; i < geometry.size(); ++i) {
            if (std::abs(geometry[i].angle.x - geometry[i - 1].angle.x) < kAngleTolerance) {
                YK_LOGE("[FdkPipeline] geometry 存在重复相邻角度，无法计算 dtheta。");
                return false;
            }
        }
        const auto relativeSpread = [](float low, float high) {
            return high > 0.f ? (high - low) / high : 0.f;
        };
        diagnostics.approximate = diagnostics.approximate ||
            relativeSpread(diagnostics.min_source_to_axis_mm,
                diagnostics.max_source_to_axis_mm) > kRelativeTolerance ||
            relativeSpread(diagnostics.min_source_to_detector_mm,
                diagnostics.max_source_to_detector_mm) > kRelativeTolerance ||
            diagnostics.max_abs_source_axis_mm >
                diagnostics.max_source_to_axis_mm * kRelativeTolerance ||
            diagnostics.max_detector_skew > kRelativeTolerance ||
            diagnostics.max_abs_principal_u_pix > kRelativeTolerance ||
            diagnostics.max_abs_principal_v_pix > kRelativeTolerance ||
            diagnostics.max_normal_misalignment > kRelativeTolerance;
        constexpr float kRadToDeg = 57.29577951308232f;
        YK_LOGI("[FdkPipeline] 相对几何：principalU=[{:.4f}, {:.4f}] pix "
            "([{:.4f}, {:.4f}] mm)，principalV=[{:.4f}, {:.4f}] pix "
            "([{:.4f}, {:.4f}] mm)，tiltU=[{:.4f}, {:.4f}] deg，"
            "tiltV=[{:.4f}, {:.4f}] deg，tiltN=[{:.4f}, {:.4f}] deg。",
            diagnostics.min_principal_u_pix, diagnostics.max_principal_u_pix,
            diagnostics.min_principal_u_pix * du0,
            diagnostics.max_principal_u_pix * du0,
            diagnostics.min_principal_v_pix, diagnostics.max_principal_v_pix,
            diagnostics.min_principal_v_pix * dv0,
            diagnostics.max_principal_v_pix * dv0,
            diagnostics.min_relative_tilt_u_rad * kRadToDeg,
            diagnostics.max_relative_tilt_u_rad * kRadToDeg,
            diagnostics.min_relative_tilt_v_rad * kRadToDeg,
            diagnostics.max_relative_tilt_v_rad * kRadToDeg,
            diagnostics.min_relative_tilt_n_rad * kRadToDeg,
            diagnostics.max_relative_tilt_n_rad * kRadToDeg);
        if (diagnostics.approximate) {
            YK_LOGW("[FdkPipeline] 使用校准 geometry 的近似 FDK：SOD=[{:.4f}, {:.4f}] mm，"
                "SDD=[{:.4f}, {:.4f}] mm，plane=[{:.4f}, {:.4f}] mm，"
                "source-axis-z(max)={:.4f} mm，principal=({:.4f}, {:.4f}) pix，"
                "normal-error(max)={:.6f}，detector-skew(max)={:.6f}。",
                diagnostics.min_source_to_axis_mm, diagnostics.max_source_to_axis_mm,
                diagnostics.min_source_to_detector_mm, diagnostics.max_source_to_detector_mm,
                diagnostics.min_source_to_plane_mm, diagnostics.max_source_to_plane_mm,
                diagnostics.max_abs_source_axis_mm,
                diagnostics.max_abs_principal_u_pix,
                diagnostics.max_abs_principal_v_pix,
                diagnostics.max_normal_misalignment,
                diagnostics.max_detector_skew);
        }
        return true;
    }

    static void buildCircularGeometry_(const SReconstructionParams& params,
        const std::vector<float>& angles, std::vector<SConeProjGeomVec>& geometry)
    {
        SCircularTrajectorySpec trajectory{};
        trajectory.angles_rad = angles;
        trajectory.sid_mm = params.scan.sid_mm;
        trajectory.sdd_mm = params.scan.sdd_mm;
        trajectory.source_offset_mm = make_float3(
            params.scan.sourceOffsetX_mm, params.scan.sourceOffsetY_mm,
            params.scan.sourceOffsetZ_mm);
        SFlatDetectorSpec detector{};
        detector.channels = params.scan.Nu;
        detector.rows = params.scan.Nv;
        detector.channel_size_mm = params.scan.du_mm;
        detector.row_size_mm = params.scan.dv_mm;
        detector.pose.offset_unv_mm = make_float3(
            params.scan.offsetU_mm, 0.f, params.scan.offsetV_mm);
        detector.pose.tilt_u_rad = params.scan.tiltU_rad;
        detector.pose.tilt_v_rad = params.scan.tiltV_rad;
        detector.pose.tilt_n_rad = params.scan.tiltN_rad;
        if (!buildProjectionGeometry(trajectory, detector, geometry))
            geometry.clear();
    }

    bool buildDerivedGeometry_(const std::vector<SConeProjGeomVec>& geometry)
    {
        host_geometry_ = geometry;
        if (!GeoDerivedManagerVec{}.build_geo_params(params_.scan.Nu, params_.scan.Nv,
            params_.scan.range_rad, host_geometry_, host_derived_geometry_)) {
            YK_LOGE("[FdkPipeline] 当前批次 geometry 的派生参数预计算失败。");
            return false;
        }
        FdkGeometryDiagnostics diagnostics{};
        return validateApproximateFdkGeometry_(
            host_geometry_, host_derived_geometry_, diagnostics);
    }

    SReconstructionParams params_{};
    int chunk_size_ = 0;
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;
    FdkOnlineState state_{};
    FdkGpuContext gpu_{};
    Fdk::PreweightProcessor preweight_{};
    Fdk::ParkerWeightProcessor parker_{};
    Fdk::FilterProcessor filter_{};
    Fdk::BpProcessor backproject_{};
    std::vector<SConeProjGeomVec> host_geometry_{};
    std::vector<SFDKGeoParamPerView> host_derived_geometry_{};
    std::vector<SConeProjGeomVec> static_geometry_{};
    std::vector<SFDKGeoParamPerView> static_derived_geometry_{};
    FdkGeometryDiagnostics geometry_diagnostics_{};
    FdkBatchFence dynamic_geometry_fence_{};
    FdkStageView last_stage_{};
};

} // namespace YK
