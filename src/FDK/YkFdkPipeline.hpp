#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "FDK/YkBackProjectProcessor.hpp"
#include "FDK/YkFDKGpuContext.hpp"
#include "FDK/YkFDKFilterProcessor.hpp"
#include "FDK/YkFDKParkerWeightProcessor.hpp"
#include "FDK/YkFDKPreWeightProcessor.hpp"
#include "FDK/YkFDKVecGeoDerived.hpp"
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

// 一个在线 FDK 批次的主机侧输入。投影内存布局为 [view][v][u]。
// geometry 是执行阶段唯一的几何和角度来源；如果没有外部 geometry，调用方
// 可给出 circular_angles，仅用于构造同一份圆轨迹 geometry。
struct FdkProjectionBatch {
    const float* projection = nullptr;
    const std::vector<SConeProjGeomVec>* geometry = nullptr;
    const float* circular_angles = nullptr;
    int count = 0;
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
    bool prepare(const SCBCTParams& params, int requested_chunk,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!validateStatic_(params, requested_chunk, stream)) return false;

        params_ = params;
        chunk_size_ = std::min(requested_chunk, kMaxChunkAng);
        stream_ = stream;

        const SProjDims chunk_dims{ params_.iPU, params_.iPV, chunk_size_ };
        gpu_.init(chunk_dims, params_.iPAngTotal, stream_, device_id);

        PreweightConfig preweight_config{};
        preweight_config.dims = chunk_dims;
        if (!preweight_.prepare(preweight_config)) { release(); return false; }

        FdkFilterConfig filter_config{};
        filter_config.dims = chunk_dims;
        filter_config.desc = params_.desc;
        filter_config.stream = stream_;
        if (!filter_.prepare(filter_config)) { release(); return false; }

        if (params_.bShortScan) {
            ParkerWeightConfig parker_config{};
            parker_config.dims = chunk_dims;
            parker_config.fDetUSize = params_.du_mm;
            parker_config.fSrcOrigin = params_.SID;
            parker_config.fDetOrigin = params_.SDD - params_.SID;
            parker_config.iPAnglesTotal = params_.iPAngTotal;
            parker_config.fScanRangeRad = params_.scan_range_rad;
            parker_config.fStartAngleRad = params_.scan_start_angle_rad;
            parker_config.nDirSign = params_.nDirSign;
            if (!parker_.prepare(parker_config)) { release(); return false; }
        }

        BpConfig bp_config{};
        bp_config.vol_geom = SVolGeom::make_centered(params_.iVX, params_.iVY,
            params_.iVZ, params_.vox_x_mm, params_.vox_y_mm, params_.vox_z_mm);
        bp_config.vol_geom.center = make_float3(params_.vol_offset_x_mm,
            params_.vol_offset_y_mm, params_.vol_offset_z_mm);
        bp_config.use_precomputed = true;
        bp_config.max_chunk_views = chunk_size_;
        if (!backproject_.prepare(bp_config)) { release(); return false; }

        state_.prepare(params_.iPAngTotal);
        prepared_ = true;
        return true;
    }

    // 任意几何的首选初始化入口。geometry 的顺序就是采集顺序，长度必须等于
    // iPAngTotal；派生几何在此一次性生成，保证跨 batch/chunk 的 dtheta 连续。
    bool prepareWithGeometry(const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& geometry, int requested_chunk,
        cudaStream_t stream, int device_id = 0)
    {
        if (static_cast<int>(geometry.size()) != params.iPAngTotal) {
            YK_LOGE("[FdkPipeline] 完整 geometry 数量 {} 与 iPAngTotal {} 不一致。",
                geometry.size(), params.iPAngTotal);
            return false;
        }
        if (!validateCircularFdkGeometry_(geometry)) return false;
        if (!prepare(params, requested_chunk, stream, device_id)) return false;
        static_geometry_ = geometry;
        if (!GeoDerivedManagerVec{}.build_geo_params(params_.iPU, params_.iPV,
            params_.scan_range_rad, static_geometry_, static_derived_geometry_)) {
            YK_LOGE("[FdkPipeline] 完整 geometry 的派生参数预计算失败。");
            release();
            return false;
        }
        return true;
    }

    // 圆轨迹只是 geometry 的一种来源。完整角度已知时同样在初始化阶段构造
    // 全量 geometry，后续与任意外部 geometry 共用同一预计算和执行路径。
    bool prepareWithAngles(const SCBCTParams& params,
        const std::vector<float>& angles, int requested_chunk,
        cudaStream_t stream, int device_id = 0)
    {
        if (static_cast<int>(angles.size()) != params.iPAngTotal) {
            YK_LOGE("[FdkPipeline] 完整角度数量 {} 与 iPAngTotal {} 不一致。",
                angles.size(), params.iPAngTotal);
            return false;
        }
        std::vector<SConeProjGeomVec> geometry;
        buildCircularGeometry_(params, angles, geometry);
        return prepareWithGeometry(params, geometry, requested_chunk, stream, device_id);
    }

    bool processBatch(const FdkProjectionBatch& batch, float* d_volume,
        bool clear_output)
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
        if (!use_static_geometry && batch.geometry) {
            if (static_cast<int>(batch.geometry->size()) != batch.count) {
                YK_LOGE("[FdkPipeline] 当前批次 geometry 数量与 count 不一致。");
                return false;
            }
            buildDerivedGeometry_(*batch.geometry);
        }
        else if (!use_static_geometry) {
            if (!batch.circular_angles) {
                YK_LOGE("[FdkPipeline] 未提供 geometry 时必须提供圆轨迹角度数组。");
                return false;
            }
            buildCircularGeometry_(params_, std::vector<float>(batch.circular_angles,
                batch.circular_angles + batch.count), host_geometry_);
            if (!GeoDerivedManagerVec{}.build_geo_params(params_.iPU, params_.iPV,
                params_.scan_range_rad, host_geometry_, host_derived_geometry_)) {
                YK_LOGE("[FdkPipeline] 圆轨迹 geometry 的派生参数预计算失败。");
                return false;
            }
        }

        if (clear_output) {
            const size_t volume_bytes = static_cast<size_t>(params_.iVX) * params_.iVY *
                params_.iVZ * sizeof(float);
            YK_CUDA_CHECK(cudaMemsetAsync(d_volume, 0, volume_bytes, stream_));
        }

        const size_t view_elements = static_cast<size_t>(params_.iPU) * params_.iPV;
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
            if (params_.bShortScan) {
                if (!parker_.apply(gpu_.proj.chunk_pw.data(),
                    { chunk.h_geometry, chunk.count }, stream_)) return false;
            }
            if (!filter_.apply(gpu_.proj.chunk_pw.data(), gpu_.proj.chunk_flt.data(),
                { chunk.h_derived_geometry, chunk.count }, stream_)) return false;
            if (!backproject_.apply(gpu_.proj.d_texObjs(), d_volume,
                { chunk.d_geometry, chunk.d_derived_geometry, chunk.count }, stream_)) return false;

            // 只记录设备工作区的观察视图；不进行回调、同步或复制。
            last_stage_ = { global_offset, count, gpu_.proj.chunk_in.data(),
                gpu_.proj.chunk_pw.data(), gpu_.proj.chunk_flt.data(), stream_ };
        }
        // 所有 CUDA 工作均已成功提交到本 session 的同一 stream；在此之后该
        // batch 才计入在线位置。异步 CUDA 执行错误仍按现有 CUDA 错误策略处理。
        state_.commit(range);
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

private:
    bool validateStatic_(const SCBCTParams& p, int requested_chunk, cudaStream_t stream) const
    {
        if (!stream || requested_chunk <= 0 || p.iPAngTotal <= 0 || p.iPU <= 0 ||
            p.iPV <= 0 || p.iVX <= 0 || p.iVY <= 0 || p.iVZ <= 0 || p.du_mm <= 0.f ||
            p.dv_mm <= 0.f || p.SID <= 0.f || p.SDD <= p.SID ||
            (p.nDirSign != 1 && p.nDirSign != -1)) {
            YK_LOGE("[FdkPipeline] 静态 FDK 配置无效。");
            return false;
        }
        return true;
    }

    // 当前实现是圆轨迹、等像素尺寸的 FDK。向量 geometry 支持每视图的
    // 探测器平移和姿态校正，但尚未实现变 SID/SDD、变 du/dv 或任意轨迹的
    // 专用滤波/冗余权重模型，因此必须在初始化时明确拒绝这些输入。
    bool validateCircularFdkGeometry_(const std::vector<SConeProjGeomVec>& geometry) const
    {
        constexpr float kRelativeTolerance = 1e-3f;
        constexpr float kAngleTolerance = 1e-5f;
        const auto length = [](const float4& v) {
            return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        };
        const auto dot = [](const float4& a, const float4& b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        };
        const float du0 = length(geometry.front().detU);
        const float dv0 = length(geometry.front().detV);
        const float sid0 = std::sqrt(geometry.front().src.x * geometry.front().src.x +
            geometry.front().src.y * geometry.front().src.y);
        if (du0 <= 0.f || dv0 <= 0.f || sid0 <= 0.f) {
            YK_LOGE("[FdkPipeline] geometry 含有零长度探测器轴或无效 SID。");
            return false;
        }
        for (const auto& view : geometry) {
            const float du = length(view.detU);
            const float dv = length(view.detV);
            const float sid = std::sqrt(view.src.x * view.src.x + view.src.y * view.src.y);
            if (std::abs(du - du0) > du0 * kRelativeTolerance ||
                std::abs(dv - dv0) > dv0 * kRelativeTolerance ||
                std::abs(sid - sid0) > sid0 * kRelativeTolerance) {
                YK_LOGE("[FdkPipeline] 当前 FDK 不支持变 SID 或变 du/dv 的 geometry。");
                return false;
            }
            if (std::abs(dot(view.detU, view.detV)) > du * dv * kRelativeTolerance ||
                !std::isfinite(view.angle.x) || std::abs(view.src.z) > sid0 * kRelativeTolerance) {
                YK_LOGE("[FdkPipeline] geometry 不满足当前圆轨迹 FDK 的正交探测器或平面源轨迹约束。");
                return false;
            }
        }
        // 外部 geometry 的 angle.x 是唯一角度来源。相邻角度重复将使 dtheta
        // 不可定义，提前拒绝可避免后续滤波/反投影出现 NaN。
        for (size_t i = 1; i < geometry.size(); ++i) {
            if (std::abs(geometry[i].angle.x - geometry[i - 1].angle.x) < kAngleTolerance) {
                YK_LOGE("[FdkPipeline] geometry 存在重复相邻角度，无法计算 dtheta。");
                return false;
            }
        }
        return true;
    }

    static void buildCircularGeometry_(const SCBCTParams& params,
        const std::vector<float>& angles, std::vector<SConeProjGeomVec>& geometry)
    {
        const auto rad2deg = [](float radians) { return radians * 180.f / CUDA_PI; };
        geometry.resize(angles.size());
        build_circular_vec_geometry_from_theta(geometry, angles, static_cast<int>(angles.size()),
            params.iPU, params.iPV, params.du_mm, params.dv_mm, params.SID,
            params.SDD - params.SID, f3(params.offsetU_mm, 0.f, params.offsetV_mm),
            f3(rad2deg(params.tiltu_angle_rad), rad2deg(params.tiltn_angle_rad),
                rad2deg(params.tiltv_angle_rad)));
    }

    void buildDerivedGeometry_(const std::vector<SConeProjGeomVec>& geometry)
    {
        host_geometry_ = geometry;
        GeoDerivedManagerVec{}.build_geo_params(params_.iPU, params_.iPV,
            params_.scan_range_rad, host_geometry_, host_derived_geometry_);
    }

    SCBCTParams params_{};
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
    FdkStageView last_stage_{};
};

} // namespace YK
