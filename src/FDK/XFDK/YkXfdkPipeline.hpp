#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "FDK/YkFDKFilterProcessor.hpp"
#include "FDK/XFDK/kernels/YkXfdkLaunch.cuh"
#include "global/YkCBCTParams.h"
#include "global/YkCudaTextureController.hpp"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

namespace YK::Fdk {

struct XfdkBatchLayout {
    int theta_chunk_views = 0;
    int source_window_views = 0;
    int periodic_prefix_views = 0;
};

// 流式入口的一批连续投影。数据布局为 [view][v][u]，host 指针必须来自
// pinned memory；pipeline 会在函数返回前复制到自己的环形缓存，因此调用者
// 不需要把整圈投影保留在设备端。
struct XfdkProjectionBatch {
    const float* h_projection = nullptr;
    int count = 0;
};

// Grimmer 等（2009）提出的 extended-range FDK。xFDK 与 C-FDK 的关键区别是：
// xFDK 在平行束变量 xi 上重排并滤波，然后在每个体素/视图上使用由可见角区间
// 推导的平滑部分扫描权重。它只能在完整、等角、理想圆轨迹上成立；当前支持
// 全量设备输入和 pinned host 流式输入两种模式，不接受倾斜、源偏移或逐视图
// 校准 geometry。
class XfdkPipeline {
public:
    XfdkPipeline() = default;
    ~XfdkPipeline() { release(); }
    XfdkPipeline(const XfdkPipeline&) = delete;
    XfdkPipeline& operator=(const XfdkPipeline&) = delete;

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        // 32 只是默认调度粒度；原始投影缓存量不使用固定常数，而是由实际
        // beta_max/dtheta 和尾包范围在 prepareBatched() 中自动计算。
        return prepareBatched(params, geometry, 32, stream, device_id);
    }

    bool prepareBatched(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        int requested_theta_chunk, cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || params.scan.Nu < 2 || params.scan.Nv < 2 ||
            params.volume.Nx <= 0 || params.volume.Ny <= 0 ||
            params.volume.Nz <= 0 || geometry.size() < 3 ||
            static_cast<int>(geometry.size()) != params.scan.NAng ||
            requested_theta_chunk <= 0) {
            YK_LOGE("[XfdkPipeline] 参数尺寸、stream 或 geometry 数量无效。");
            return false;
        }
        if (!deriveGeometry_(params, geometry, geometry_)) return false;

        batch_layout_.theta_chunk_views = std::min(requested_theta_chunk,
            geometry_.views);
        // 浮点数恰好靠近整数视图时，不同 theta_begin 的 floor/ceil 可能
        // 相差一帧。遍历所有起点取最大值，保证工作区对每个批次都足够。
        for (int begin = 0; begin < geometry_.views; ++begin) {
            const int count = std::min(batch_layout_.theta_chunk_views,
                geometry_.views - begin);
            batch_layout_.source_window_views = std::max(
                batch_layout_.source_window_views,
                sourceWindow_(geometry_, begin, count).source_count);
        }
        for (int begin = 0; begin < geometry_.views; ++begin) {
            const int count = std::min(batch_layout_.theta_chunk_views,
                geometry_.views - begin);
            const auto window = sourceWindow_(geometry_, begin, count);
            batch_layout_.periodic_prefix_views = std::max(
                batch_layout_.periodic_prefix_views,
                std::max(0, window.source_begin + window.source_count -
                    geometry_.views));
            if (window.source_begin < 0) {
                const int ring_first_retained = std::max(0,
                    geometry_.views - batch_layout_.source_window_views);
                const int source_end = window.source_begin +
                    window.source_count - 1;
                batch_layout_.periodic_prefix_views = std::max(
                    batch_layout_.periodic_prefix_views,
                    std::min(source_end + 1, ring_first_retained));
            }
        }

        YK_CUDA_CHECK(cudaSetDevice(device_id));
        const size_t view_elements = static_cast<size_t>(geometry_.input_u) *
            geometry_.input_v;
        host_ring_ = memory_.allocatePinnedCpu3D<float>(
            geometry_.input_u, geometry_.input_v,
            batch_layout_.source_window_views);
        host_prefix_ = memory_.allocatePinnedCpu3D<float>(
            geometry_.input_u, geometry_.input_v,
            std::max(1, batch_layout_.periodic_prefix_views));
        host_window_ = memory_.allocatePinnedCpu3D<float>(
            geometry_.input_u, geometry_.input_v,
            batch_layout_.source_window_views);
        d_input_window_ = memory_.allocateDevice3D<float>(
            geometry_.input_u, geometry_.input_v,
            batch_layout_.source_window_views, device_id);
        d_rebinned_ = memory_.allocateDevice3D<float>(
            geometry_.output_xi, geometry_.output_gamma,
            batch_layout_.theta_chunk_views, device_id);
        d_filtered_ = memory_.allocateDevice3D<float>(
            geometry_.output_xi, geometry_.output_gamma,
            batch_layout_.theta_chunk_views, device_id);
        input_texture_ = Mem::TextureController::createEmptyTex3D(
            geometry_.input_u, geometry_.input_v,
            batch_layout_.source_window_views,
            cudaFilterModeLinear, cudaAddressModeBorder);
        filtered_texture_ = Mem::TextureController::createEmptyTex3D(
            geometry_.output_xi, geometry_.output_gamma,
            batch_layout_.theta_chunk_views,
            cudaFilterModeLinear, cudaAddressModeBorder);

        filter_geometry_.assign(batch_layout_.theta_chunk_views,
            SFDKGeoParamPerView{});
        for (auto& view : filter_geometry_) {
            view.Nu = geometry_.output_xi;
            view.Nv = geometry_.output_gamma;
            view.du_mm = geometry_.dxi_mm;
            view.inv_du_mm = 1.f / geometry_.dxi_mm;
            view.offsetU_pix = 0.f;
        }
        FdkFilterConfig filter_config{};
        filter_config.dims = { geometry_.output_xi, geometry_.output_gamma,
            batch_layout_.theta_chunk_views };
        filter_config.desc = params.reconstruction.filter;
        filter_config.stream = stream;
        if (!filter_.prepare(filter_config)) { release(); return false; }

        volume_geometry_ = SVolGeom::make_centered(params.volume.Nx,
            params.volume.Ny, params.volume.Nz, params.volume.voxelX_mm,
            params.volume.voxelY_mm, params.volume.voxelZ_mm);
        volume_geometry_.center = make_float3(params.volume.centerX_mm,
            params.volume.centerY_mm, params.volume.centerZ_mm);
        stream_ = stream;
        device_id_ = device_id;
        YK_CUDA_CHECK(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
        prepared_ = true;
        YK_LOGI("[XfdkPipeline] 分批工作区：theta chunk={}，原始投影窗口={}，"
            "周期首部缓存需求={}。", batch_layout_.theta_chunk_views,
            batch_layout_.source_window_views,
            batch_layout_.periodic_prefix_views);
        return true;
    }

    // d_projection 布局为 [view][v][u]。该入口提交全量 xFDK 工作，返回后
    // 仍需 wait() 或同步所属 stream 才能读取输出体。
    bool reconstruct(const float* d_projection, float* d_volume,
        bool clear_output = true)
    {
        if (!prepared_ || !d_projection || !d_volume) {
            YK_LOGE("[XfdkPipeline] reconstruct 在 prepare 前调用或指针为空。");
            return false;
        }
        if (clear_output) {
            const size_t bytes = static_cast<size_t>(volume_geometry_.Nx) *
                volume_geometry_.Ny * volume_geometry_.Nz * sizeof(float);
            YK_CUDA_CHECK(cudaMemsetAsync(d_volume, 0, bytes, stream_));
        }

        for (int theta_begin = 0; theta_begin < geometry_.views;
             theta_begin += batch_layout_.theta_chunk_views) {
            const int theta_count = std::min(batch_layout_.theta_chunk_views,
                geometry_.views - theta_begin);
            const detail::SXfdkChunk chunk = sourceWindow_(geometry_,
                theta_begin, theta_count);
            gatherSourceWindow_(d_projection, chunk);
            // cudaArray 按最大窗口创建。尾包只更新前 source_count 层，kernel
            // 由 chunk.source_count 限定访问范围，其余层内容不会参与计算。
            Mem::TextureController::updateTex3DFromDeviceAsync(input_texture_,
                d_input_window_.data(), geometry_.input_u, geometry_.input_v,
                batch_layout_.source_window_views, stream_);
            detail::launchXfdkRebin(input_texture_.tex, d_rebinned_.data(),
                geometry_, chunk, launch_policy_, stream_);
            if (!filter_.apply(d_rebinned_.data(), d_filtered_.data(),
                { filter_geometry_.data(), theta_count }, stream_)) return false;
            Mem::TextureController::updateTex3DFromDeviceAsync(filtered_texture_,
                d_filtered_.data(), geometry_.output_xi, geometry_.output_gamma,
                batch_layout_.theta_chunk_views, stream_);
            detail::launchXfdkBackprojection(filtered_texture_.tex, d_volume,
                geometry_, volume_geometry_, chunk, true, launch_policy_, stream_);
        }
        YK_CUDA_CHECK(cudaEventRecord(completion_, stream_));
        completion_recorded_ = true;
        return true;
    }

    // 开始真正的流式重建。随后必须按采集顺序调用 enqueueBatch()，最后调用
    // completeStreaming()。该模式只在设备上保留一个按几何自动计算的原始
    // 投影窗口，不需要完整 d_projection。
    bool beginStreaming(float* d_volume, bool clear_output = true)
    {
        if (!prepared_ || !d_volume || streaming_active_) return false;
        if (clear_output) {
            const size_t bytes = static_cast<size_t>(volume_geometry_.Nx) *
                volume_geometry_.Ny * volume_geometry_.Nz * sizeof(float);
            YK_CUDA_CHECK(cudaMemsetAsync(d_volume, 0, bytes, stream_));
        }
        streaming_volume_ = d_volume;
        streaming_received_ = 0;
        streaming_next_theta_ = 0;
        streaming_initial_theta_count_ = 0;
        streaming_active_ = true;
        streaming_complete_ = false;
        return true;
    }

    // 追加一批按采集顺序排列的 host 投影。输入批次可以任意大小，内部会
    // 按单视图写入环形缓存并尽可能早地消费已经满足 halo 的 theta 区间。
    bool enqueueBatch(const XfdkProjectionBatch& batch)
    {
        if (!streaming_active_ || !batch.h_projection || batch.count <= 0 ||
            batch.count > geometry_.views - streaming_received_)
            return false;
        const size_t view_elements = static_cast<size_t>(geometry_.input_u) *
            geometry_.input_v;
        for (int i = 0; i < batch.count; ++i) {
            const int global_view = streaming_received_++;
            const int slot = global_view % batch_layout_.source_window_views;
            std::memcpy(host_ring_.data() + static_cast<size_t>(slot) *
                view_elements, batch.h_projection + static_cast<size_t>(i) *
                view_elements, view_elements * sizeof(float));
            if (global_view < batch_layout_.periodic_prefix_views) {
                std::memcpy(host_prefix_.data() + static_cast<size_t>(global_view) *
                    view_elements, batch.h_projection + static_cast<size_t>(i) *
                    view_elements, view_elements * sizeof(float));
            }
            if (!drainStreaming_(false)) return false;
        }
        return true;
    }

    // 接收完整 views 后处理因周期边界而延迟的首尾 theta 区间。
    bool completeStreaming()
    {
        if (!streaming_active_ || streaming_received_ != geometry_.views)
            return false;
        // 首个跨周期块在采集期间没有足够的末尾 halo，放在整圈接收后
        // 使用环形缓存中的尾部视图补做；其余块已经在线累加完成。
        if (streaming_initial_theta_count_ > 0) {
            const int saved_next = streaming_next_theta_;
            streaming_next_theta_ = 0;
            const int initial_end = streaming_initial_theta_count_;
            while (streaming_next_theta_ < initial_end) {
                const int theta_count = std::min(batch_layout_.theta_chunk_views,
                    initial_end - streaming_next_theta_);
                const auto chunk = sourceWindow_(geometry_,
                    streaming_next_theta_, theta_count);
                if (!loadStreamingWindow_(chunk) || !processChunk_(chunk,
                    streaming_volume_, true)) return false;
                YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                streaming_next_theta_ += theta_count;
            }
            streaming_next_theta_ = saved_next;
            streaming_initial_theta_count_ = 0;
        }
        if (!drainStreaming_(true)) return false;
        YK_CUDA_CHECK(cudaEventRecord(completion_, stream_));
        completion_recorded_ = true;
        streaming_active_ = false;
        streaming_complete_ = true;
        return true;
    }

    bool streamingActive() const { return streaming_active_; }
    int receivedStreamingViews() const { return streaming_received_; }

    bool wait() const
    {
        if (!completion_recorded_) return true;
        return cudaEventSynchronize(completion_) == cudaSuccess;
    }

    void release()
    {
        if (completion_recorded_ && completion_) cudaEventSynchronize(completion_);
        filter_.release();
        input_texture_ = {};
        filtered_texture_ = {};
        d_input_window_ = {};
        host_ring_ = {};
        host_prefix_ = {};
        host_window_ = {};
        d_rebinned_ = {};
        d_filtered_ = {};
        filter_geometry_.clear();
        if (completion_) cudaEventDestroy(completion_);
        completion_ = nullptr;
        completion_recorded_ = false;
        geometry_ = {};
        batch_layout_ = {};
        volume_geometry_ = {};
        stream_ = nullptr;
        device_id_ = 0;
        prepared_ = false;
        streaming_volume_ = nullptr;
        streaming_received_ = 0;
        streaming_next_theta_ = 0;
        streaming_active_ = false;
        streaming_complete_ = false;
        streaming_initial_theta_count_ = 0;
    }

    bool isPrepared() const { return prepared_; }
    const detail::SXfdkGeometry& derivedGeometry() const { return geometry_; }
    const XfdkBatchLayout& batchLayout() const { return batch_layout_; }

private:
    static detail::SXfdkChunk sourceWindow_(
        const detail::SXfdkGeometry& geometry,
        int theta_begin, int theta_count)
    {
        const double shift = static_cast<double>(geometry.beta_max_rad) /
            static_cast<double>(geometry.dtheta_rad);
        const int source_begin = static_cast<int>(std::floor(
            static_cast<double>(theta_begin) - shift));
        const int source_end = static_cast<int>(std::ceil(
            static_cast<double>(theta_begin + theta_count - 1) + shift));
        return { theta_begin, theta_count, source_begin,
            source_end - source_begin + 1 };
    }

    static int wrapView_(int view, int total)
    {
        const int result = view % total;
        return result < 0 ? result + total : result;
    }

    bool loadStreamingWindow_(const detail::SXfdkChunk& chunk)
    {
        const size_t view_elements = static_cast<size_t>(geometry_.input_u) *
            geometry_.input_v;
        for (int i = 0; i < chunk.source_count; ++i) {
            const int logical = chunk.source_begin + i;
            const int view = wrapView_(logical, geometry_.views);
            const float* source = nullptr;
            if (logical >= 0 && logical < batch_layout_.periodic_prefix_views) {
                source = host_prefix_.cdata() + static_cast<size_t>(view) *
                    view_elements;
            } else if (logical < 0) {
                // 负逻辑编号对应扫描末尾视图。它们在整圈接收完成后仍
                // 保留在环形缓存中，不能误读为首部缓存。
                source = host_ring_.cdata() + static_cast<size_t>(view %
                    batch_layout_.source_window_views) * view_elements;
            } else if (logical >= geometry_.views) {
                // 超过末尾的逻辑编号回绕到扫描首部，首部视图从独立的
                // pinned 缓存读取，避免最后写入的环形槽位已被覆盖。
                source = host_prefix_.cdata() + static_cast<size_t>(view) *
                    view_elements;
            } else {
                source = host_ring_.cdata() + static_cast<size_t>(view %
                    batch_layout_.source_window_views) * view_elements;
            }
            std::memcpy(host_window_.data() + static_cast<size_t>(i) *
                view_elements, source, view_elements * sizeof(float));
        }
        // 窗口 staging 是 pinned memory；提交后立即同步 stream，保证下一次
        // 环形槽位复用时不会覆盖仍在进行的 H2D。kernel 本身仍按同一 stream
        // 异步排队，完成事件由 completeStreaming() 统一暴露。
        YK_CUDA_CHECK(cudaMemcpyAsync(d_input_window_.data(), host_window_.cdata(),
            static_cast<size_t>(chunk.source_count) * view_elements * sizeof(float),
            cudaMemcpyHostToDevice, stream_));
        Mem::TextureController::updateTex3DFromDeviceAsync(input_texture_,
            d_input_window_.data(), geometry_.input_u, geometry_.input_v,
            batch_layout_.source_window_views, stream_);
        return true;
    }

    bool processChunk_(const detail::SXfdkChunk& chunk, float* output,
        bool accumulate)
    {
        if (!output || chunk.theta_count <= 0) return false;
        detail::launchXfdkRebin(input_texture_.tex, d_rebinned_.data(),
            geometry_, chunk, launch_policy_, stream_);
        if (!filter_.apply(d_rebinned_.data(), d_filtered_.data(),
            { filter_geometry_.data(), chunk.theta_count }, stream_)) return false;
        Mem::TextureController::updateTex3DFromDeviceAsync(filtered_texture_,
            d_filtered_.data(), geometry_.output_xi, geometry_.output_gamma,
            batch_layout_.theta_chunk_views, stream_);
        detail::launchXfdkBackprojection(filtered_texture_.tex, output,
            geometry_, volume_geometry_, chunk, accumulate,
            launch_policy_, stream_);
        return true;
    }

    bool drainStreaming_(bool flush)
    {
        while (streaming_next_theta_ < geometry_.views) {
            const int theta_count = std::min(batch_layout_.theta_chunk_views,
                geometry_.views - streaming_next_theta_);
            const detail::SXfdkChunk chunk = sourceWindow_(geometry_,
                streaming_next_theta_, theta_count);
            const int source_end = chunk.source_begin + chunk.source_count - 1;
            if (!flush && chunk.source_begin < 0) {
                // 只挂起跨周期的起始块，继续尝试后续可在线消费的块。
                streaming_initial_theta_count_ = std::max(
                    streaming_initial_theta_count_,
                    streaming_next_theta_ + theta_count);
                streaming_next_theta_ += theta_count;
                continue;
            }
            if (!flush && source_end >= streaming_received_) break;
            if (!loadStreamingWindow_(chunk)) return false;
            if (!processChunk_(chunk, streaming_volume_, streaming_next_theta_ != 0))
                return false;
            // host_window_ 和 d_input_window_ 会在下一个块复用，先完成当前
            // 块的复制/纹理/反投影，避免单缓冲覆盖造成数据竞争。
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            streaming_next_theta_ += theta_count;
        }
        return true;
    }

    void gatherSourceWindow_(const float* d_projection,
        const detail::SXfdkChunk& chunk)
    {
        const size_t view_elements = static_cast<size_t>(geometry_.input_u) *
            geometry_.input_v;
        int copied = 0;
        while (copied < chunk.source_count) {
            const int source = wrapView_(chunk.source_begin + copied,
                geometry_.views);
            const int run = std::min(chunk.source_count - copied,
                geometry_.views - source);
            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_input_window_.data() + static_cast<size_t>(copied) *
                    view_elements,
                d_projection + static_cast<size_t>(source) * view_elements,
                static_cast<size_t>(run) * view_elements * sizeof(float),
                cudaMemcpyDeviceToDevice, stream_));
            copied += run;
        }
    }

    static float3 xyz_(const float4& value)
    { return make_float3(value.x, value.y, value.z); }
    static float dot_(const float3& a, const float3& b)
    { return a.x * b.x + a.y * b.y + a.z * b.z; }
    static float length_(const float3& a) { return std::sqrt(dot_(a, a)); }
    static float3 sub_(const float3& a, const float3& b)
    { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

    static bool fail_(const char* message)
    {
        YK_LOGE("[XfdkPipeline] 几何不满足论文适用条件：{}。", message);
        return false;
    }

    static bool deriveGeometry_(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        detail::SXfdkGeometry& result)
    {
        constexpr float pi = 3.14159265358979323846f;
        const int nu = params.scan.Nu, nv = params.scan.Nv;
        const float cu = 0.5f * (nu - 1), cv = 0.5f * (nv - 1);
        std::vector<float> angles;
        angles.reserve(geometry.size());
        float sid_sum = 0.f, sdd_sum = 0.f, du_sum = 0.f, dv_sum = 0.f;
        for (const auto& view : geometry) {
            const float3 source = xyz_(view.src);
            const float3 du = xyz_(view.detU), dv = xyz_(view.detV);
            const float3 center = make_float3(view.detS.x + cu * du.x + cv * dv.x,
                view.detS.y + cu * du.y + cv * dv.y,
                view.detS.z + cu * du.z + cv * dv.z);
            const float sid = std::hypot(source.x, source.y);
            const float sdd = length_(sub_(center, source));
            const float du_mm = length_(du), dv_mm = length_(dv);
            if (!(sid > 0.f && sdd > sid && du_mm > 0.f && dv_mm > 0.f))
                return fail_("存在退化的 SID、SDD 或探测器像素向量");
            const float scale = std::max({ sid, sdd, 1.f });
            const float tol = 3e-4f * scale;
            const float3 expected_u = make_float3(-source.y / sid, source.x / sid, 0.f);
            const float3 expected_center = make_float3(
                -source.x * (sdd - sid) / sid,
                -source.y * (sdd - sid) / sid, 0.f);
            if (std::fabs(source.z) > tol ||
                length_(sub_(center, expected_center)) > tol ||
                std::fabs(dot_(du, expected_u) / du_mm - 1.f) > 3e-4f ||
                std::fabs(du.z) > 3e-4f * du_mm ||
                std::fabs(dv.x) > 3e-4f * dv_mm ||
                std::fabs(dv.y) > 3e-4f * dv_mm ||
                dv.z < (1.f - 3e-4f) * dv_mm ||
                std::fabs(dot_(du, dv)) > 3e-4f * du_mm * dv_mm)
                return fail_("不支持源/探测器偏移、倾斜或 skew");
            // 论文坐标 s(alpha)=(R sin alpha,-R cos alpha,0)。因此这里的
            // alpha 不是源点的普通极角 atan2(y,x)，而是 atan2(x,-y)。
            float angle = std::atan2(source.x, -source.y);
            if (!angles.empty()) {
                float d = angle - angles.back();
                while (d <= -pi) d += 2.f * pi;
                while (d > pi) d -= 2.f * pi;
                angle = angles.back() + d;
            }
            angles.push_back(angle);
            sid_sum += sid; sdd_sum += sdd; du_sum += du_mm; dv_sum += dv_mm;
        }
        const int views = static_cast<int>(geometry.size());
        const float sid = sid_sum / views, sdd = sdd_sum / views;
        const float du = du_sum / views, dv = dv_sum / views;
        const float step = (angles.back() - angles.front()) / (views - 1);
        if (std::fabs(std::fabs(step) - 2.f * pi / views) > 4e-4f ||
            params.scan.short_scan)
            return fail_("只支持不重复终点的等角完整 2pi 扫描");
        for (int i = 0; i < views; ++i) {
            const float expected = angles.front() + step * i;
            const float3 source = xyz_(geometry[i].src);
            const float3 duv = xyz_(geometry[i].detU), dvv = xyz_(geometry[i].detV);
            const float current_sid = std::hypot(source.x, source.y);
            const float current_du = length_(duv), current_dv = length_(dvv);
            const float3 center = make_float3(
                geometry[i].detS.x + cu * duv.x + cv * dvv.x,
                geometry[i].detS.y + cu * duv.y + cv * dvv.y,
                geometry[i].detS.z + cu * duv.z + cv * dvv.z);
            const float current_sdd = length_(sub_(center, source));
            if (std::fabs(angles[i] - expected) > 4e-4f ||
                std::fabs(current_sid - sid) > 4e-4f * sid ||
                std::fabs(current_sdd - sdd) > 4e-4f * sdd ||
                std::fabs(current_du - du) > 4e-4f * du ||
                std::fabs(current_dv - dv) > 4e-4f * dv)
                return fail_("轨迹半径、像素间距或角度步长不恒定");
        }

        const float half_u = cu * du, half_v = cv * dv;
        const float beta_max = std::atan2(half_u, sdd);
        // 论文 Fig. 1 的 gamma_max 位于中心矢状面，是中心列到探测器
        // 纵向边缘的锥角。矩形平板角点在固定 gamma 网格上自然由 border-zero
        // 裁剪，不能用角点较小的锥角收缩整个可重建 z 范围。
        const float gamma_max = std::atan2(half_v, sdd);
        if (!(beta_max > 0.f && gamma_max > 0.f && beta_max < 0.5f * pi &&
            gamma_max < 0.5f * pi)) return fail_("探测器角度范围退化");

        result = {};
        result.input_u = result.output_xi = nu;
        result.input_v = result.output_gamma = nv;
        result.views = views;
        result.sid_mm = sid; result.sdd_mm = sdd;
        result.input_du_mm = du; result.input_dv_mm = dv;
        result.beta_max_rad = beta_max; result.gamma_max_rad = gamma_max;
        result.xi_min_mm = -sid * std::sin(beta_max);
        result.dxi_mm = 2.f * sid * std::sin(beta_max) / (nu - 1);
        result.gamma_min_rad = -gamma_max;
        result.dgamma_rad = 2.f * gamma_max / (nv - 1);
        result.alpha0_rad = angles.front();
        result.signed_dtheta_rad = step;
        result.dtheta_rad = std::fabs(step);
        result.reconstruction_radius_mm = std::min(
            0.5f * (nu - 1) * du, sid * std::sin(beta_max));
        result.c_cot_gamma = 1.f / std::tan(gamma_max);
        result.delta_r_mm = result.reconstruction_radius_mm *
            result.reconstruction_radius_mm / (2.f * sid);
        return result.dxi_mm > 0.f && result.dgamma_rad > 0.f;
    }

    bool prepared_ = false;
    bool completion_recorded_ = false;
    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t completion_ = nullptr;
    detail::SXfdkGeometry geometry_{};
    XfdkBatchLayout batch_layout_{};
    SVolGeom volume_geometry_{};
    SKernelLaunchPolicy launch_policy_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> d_input_window_{};
    Mem::HostPinnedBuffer3D<float> host_ring_{};
    Mem::HostPinnedBuffer3D<float> host_prefix_{};
    Mem::HostPinnedBuffer3D<float> host_window_{};
    Mem::DeviceLinearBuffer3D<float> d_rebinned_{};
    Mem::DeviceLinearBuffer3D<float> d_filtered_{};
    Mem::Tex3DHandle input_texture_{};
    Mem::Tex3DHandle filtered_texture_{};
    std::vector<SFDKGeoParamPerView> filter_geometry_{};
    FilterProcessor filter_{};
    float* streaming_volume_ = nullptr;
    int streaming_received_ = 0;
    int streaming_next_theta_ = 0;
    bool streaming_active_ = false;
    bool streaming_complete_ = false;
    int streaming_initial_theta_count_ = 0;
};

} // namespace YK::Fdk
