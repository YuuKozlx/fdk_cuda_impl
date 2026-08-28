#pragma once
#include "global/YkGlobals.h"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include <cassert>
#include <stdexcept>

using namespace YK;



struct SHeliCTParam {
    // ---- 探测器 ----
    int   iPU = 1024;
    int   iPV = 32;
    float du_mm = 0.25f;
    float dv_mm = 0.25f;
    float offsetU_mm = 0.f;
    float offsetV_mm = 0.f;   // 新增
    // 源端在机架初始局部坐标系中的校准偏移，由 geometry builder 烘焙。
    float sourceOffsetX_mm = 0.f;
    float sourceOffsetY_mm = 0.f;
    float sourceOffsetZ_mm = 0.f;

    // ---- 几何 ----
    float SID = 500.f;
    float SDD = 1000.f;

    // ---- 体积 ----
    int   iVX = 512;
    int   iVY = 512;
    int   iVZ = 400;
    float vox_x_mm = 0.1f;
    float vox_y_mm = 0.1f;
    float vox_z_mm = 0.1f;
    float vol_offset_x_mm = 0.f;
    float vol_offset_y_mm = 0.f;
    float vol_offset_z_mm = 0.f;

    // ---- 探测器倾斜 ----
    float tiltu_angle_rad = 0.f;
    float tiltn_angle_rad = 0.f;
    float tiltv_angle_rad = 0.f;

    // ---- 螺旋扫描 ----
    float              pitch_mm = 3.f;
    float              start_z_mm = 0.f;
    int               views_per_rot = 720;
    std::vector<float> angle_list;  // 新增：外部传入，仿真或真实数据均用此

    // ---- 算子选择 ----
    ETask fp_task = ETask::FP_Joseph;
};


#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace YK {

    // ════════════════════════════════════════════════════════════════
    //  §1  基础几何
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 几何放大比 M = SDD / SID
     *
     * 所有探测器物理尺寸换算到等中心均除以 M。
     *
     * @param SID  源–等中心距 (mm)
     * @param SDD  源–探测器距 (mm)
     * @return     放大比 M（无量纲，通常 > 1）
     */
    inline float magnification(float SID, float SDD)
    {
        assert(SID > 0.f && SDD > SID);
        return SDD / SID;
    }

    // ════════════════════════════════════════════════════════════════
    //  §2  探测器 Z 方向等中心覆盖
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 探测器在等中心轴线（r=0）处的 Z 方向总覆盖宽度
     *
     * 公式：
     *   dZ_axis = iPV * dv_mm / M
     *
     * 物理含义：iPV 排探测器投影到等中心平面后所张开的 Z 范围。
     * 仅在旋转轴上成立，用于 pitch 合法性校验和扫描圈数估算。
     *
     * @param iPV    探测器排数
     * @param dv_mm  探测器排间距 (mm)
     * @param M      放大比（由 magnification() 获得）
     * @return       轴线处 Z 覆盖 (mm)
     */
    inline float detectorZCoverageAxis(int iPV, float dv_mm, float M)
    {
        return static_cast<float>(iPV) * dv_mm / M;
    }

    /**
     * @brief 重建半径 R 处的有效 Z 覆盖宽度
     *
     * 由于锥束几何，距旋转轴 r=R 处的体素，其可用投影排数
     * 相比轴线处被压缩，有效 Z 覆盖随 R 增大而减小：
     *
     *   dZ_eff(R) = dZ_axis * (SID - R) / SID
     *             = iPV * dv_mm * (SID - R) / SDD
     *
     * 推导：源点到 r=R 体素的射线在探测器面上的 V 方向张角，
     * 等效回等中心平面时需乘以几何压缩因子 (SID - R) / SID。
     *
     * 此值可用于判断指定重建范围在给定 pitch 下是否具备足够的
     * z 向采样覆盖；具体 wFBP/PI-line 窗口应由对应算法单独定义。
     *
     * @param iPV    探测器排数
     * @param dv_mm  探测器排间距 (mm)
     * @param SID    源–等中心距 (mm)
     * @param SDD    源–探测器距 (mm)
     * @param R      重建半径 (mm)，须满足 0 <= R < SID
     * @return       r=R 处有效 Z 覆盖 (mm)
     */
    inline float detectorZCoverageAtR(int iPV, float dv_mm,
        float SID, float SDD, float R)
    {
        assert(R >= 0.f && R < SID);
        return static_cast<float>(iPV) * dv_mm * (SID - R) / SDD;
    }

    // ════════════════════════════════════════════════════════════════
    //  §3  Pitch 约束（双向推导）
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 给定探测器参数，计算最大允许螺旋 pitch
     *
     * 公式：
     *   h_max = pf * dZ_det = pf * (iPV * dv_mm / M)
     *
     * 物理含义：床进量不得超过探测器 Z 覆盖，否则相邻圈之间出现
     * 采样间隙（漏采样）。pf < 1 时相邻圈有重叠（冗余但鲁棒），
     * pf > 1 需 z 方向插值补偿。
     *
     * @param dZ_det      等中心处探测器 Z 覆盖 (mm)
     * @param pitch_ratio Pitch 比 pf，推荐 [0.75, 1.0]
     * @return            最大允许螺旋 pitch (mm)
     */
    inline float maxPitch(float dZ_det, float pitch_ratio = 0.75f)
    {
        return pitch_ratio * dZ_det;
    }

    /**
     * @brief 给定螺旋 pitch，计算所需最少探测器排数
     *
     * 公式：
     *   iPV >= ceil( h * M / (pf * dv_mm) )
     *
     * @param pitch_mm    螺旋床进量 (mm)
     * @param dv_mm       探测器排间距 (mm)
     * @param M           放大比
     * @param pitch_ratio Pitch 比 pf，推荐 [0.75, 1.0]
     * @return            所需最少探测器排数
     */
    inline int minDetectorRows(float pitch_mm, float dv_mm, float M,
        float pitch_ratio = 0.75f)
    {
        return static_cast<int>(
            std::ceil(pitch_mm * M / (pitch_ratio * dv_mm)));
    }

    // ════════════════════════════════════════════════════════════════
    //  §4  体积 Z 范围
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 体积 Z 方向物理范围
     *
     * 公式：
     *   Z_vol = iVZ * vox_z_mm
     *
     * @param iVZ      体积 Z 方向层数
     * @param vox_z_mm 体素 Z 尺寸 (mm)
     * @return         体积 Z 物理范围 (mm)
     */
    inline float volumeZRange(int iVZ, float vox_z_mm)
    {
        return static_cast<float>(iVZ) * vox_z_mm;
    }

    // ════════════════════════════════════════════════════════════════
    //  §5  扫描 ramp 余量
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 扫描起止端所需的 ramp 缓冲余量（单侧）
     *
     * 公式：
     *   Z_margin = dZ_det + h
     *
     * 物理含义：
     *   - dZ_det：前/后各留一个探测器覆盖，保证重建平面两侧均有
     *             完整的锥束投影数据（避免截断伪影）。
     *   - h     ：一个 pitch 作为螺旋入/出加速缓冲。
     *
     * @param dZ_det   等中心处探测器 Z 覆盖 (mm)
     * @param pitch_mm 螺旋床进量 (mm)
     * @return         单侧余量 (mm)
     */
    inline float scanMargin(float dZ_det, float pitch_mm)
    {
        return dZ_det + pitch_mm;
    }

    // ════════════════════════════════════════════════════════════════
    //  §6  扫描起始位置
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 计算螺旋扫描起始 Z 位置（体积中心在 z=0）
     *
     * 公式：
     *   z_start = -(Z_vol/2 + dZ_det + h)
     *           = -(iVZ * vox_z_mm / 2 + iPV * dv_mm / M + pitch_mm)
     *
     * @param Z_vol    体积 Z 范围 (mm)
     * @param dZ_det   等中心处探测器 Z 覆盖 (mm)
     * @param pitch_mm 螺旋床进量 (mm)
     * @return         扫描起始 Z 坐标 (mm)，通常为负值
     */
    inline float scanStartZ(float Z_vol, float dZ_det, float pitch_mm)
    {
        return -(Z_vol * 0.5f + scanMargin(dZ_det, pitch_mm));
    }

    // ════════════════════════════════════════════════════════════════
    //  §7  旋转圈数
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 计算覆盖完整体积所需的旋转圈数
     *
     * 公式：
     *   N_r = ceil( (Z_vol + 2 * dZ_det) / h ) + 1
     *
     * +1 保证末圈完整落在体积边界之外，防止边缘层投影不完整。
     *
     * @param Z_vol    体积 Z 范围 (mm)
     * @param dZ_det   等中心处探测器 Z 覆盖 (mm)
     * @param pitch_mm 螺旋床进量 (mm)
     * @return         所需旋转圈数
     */
    inline int numRotations(float Z_vol, float dZ_det, float pitch_mm)
    {
        return static_cast<int>(
            std::ceil((Z_vol + 2.f * dZ_det) / pitch_mm)) + 1;
    }

    // ════════════════════════════════════════════════════════════════
    //  §8  每圈投影数
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 扇角半角（探测器 U 方向 FOV 对应的最大扇角）
     *
     * 公式：
     *   gamma_max = atan( (iPU * du_mm) / (2 * M * SID) )
     *             = atan( W_u / SID )   其中 W_u = iPU * du_mm / (2M)
     *
     * @param iPU   探测器列数
     * @param du_mm 探测器列间距 (mm)
     * @param M     放大比
     * @param SID   源–等中心距 (mm)
     * @return      扇角半角 (rad)
     */
    inline float fanHalfAngle(int iPU, float du_mm, float M, float SID)
    {
        const float Wu = static_cast<float>(iPU) * du_mm / (2.f * M);
        return std::atan2(Wu, SID);
    }

    /**
     * @brief 单列角步长（精确 arcsin 公式）
     *
     * 公式：
     *   d_gamma = arcsin( du_mm / (M * SID) )
     *
     * 近似（du_mm << M * SID）：d_gamma ≈ du_mm / (M * SID)
     *
     * 注意：此函数给出理论 Nyquist 角步长下界，实际工程中
     * 每圈投影数由机械旋转速度和采集帧率决定，通常直接
     * 指定 views_per_rot（如 720），不必等于该理论值。
     *
     * @param du_mm 探测器列间距 (mm)
     * @param M     放大比
     * @param SID   源–等中心距 (mm)
     * @return      单列角步长 (rad)
     */
    inline float fanAngleStep(float du_mm, float M, float SID)
    {
        return std::asin(du_mm / (M * SID));
    }

    /**
     * @brief 按角度 Nyquist 准则计算每圈所需最少投影数（理论下界）
     *
     * 公式（短扫描，扫描范围 π + 2γ_max）：
     *   N_phi >= ceil( (π + 2 * gamma_max) / d_gamma )
     *
     * 完整扫描（360°）取 2π / d_gamma。
     *
     * 警告：对于像素尺寸较小（如 du/M ≈ 0.2mm）的探测器，
     * 该函数返回值可能达到数千甚至上万，远超实际需求。
     * 工程中建议直接指定 views_per_rot，此函数仅供参考。
     *
     * @param iPU        探测器列数
     * @param du_mm      探测器列间距 (mm)
     * @param M          放大比
     * @param SID        源–等中心距 (mm)
     * @param short_scan 是否短扫描（π + 扇角，默认 true）
     * @return           每圈最少投影数（理论 Nyquist 下界）
     */
     //inline int minViewsPerRotation(int iPU, float du_mm, float M, float SID,
     //    bool short_scan = true)
     //{
     //    const float gamma_max = fanHalfAngle(iPU, du_mm, M, SID);
     //    const float d_gamma = fanAngleStep(du_mm, M, SID);
     //    const float scan_range = short_scan
     //        ? static_cast<float>(M_PI) + 2.f * gamma_max
     //        : 2.f * static_cast<float>(M_PI);
     //    return static_cast<int>(std::ceil(scan_range / d_gamma));
     //}

     // ════════════════════════════════════════════════════════════════
     //  §9  总投影数
     // ════════════════════════════════════════════════════════════════

     /**
      * @brief 总投影数
      *
      * 公式：
      *   N_total = N_r * N_phi
      *
      * @param n_rot         旋转圈数
      * @param views_per_rot 每圈投影数
      * @return              总投影数
      */
    inline int totalViews(int n_rot, int views_per_rot)
    {
        return n_rot * views_per_rot;
    }

    // ════════════════════════════════════════════════════════════════
    //  §10  最大锥角（判断是否需要锥束修正算法）
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 最大锥角（等中心处，Z 方向半张角）
     *
     * 公式：
     *   alpha_max = atan( dZ_det / (2 * SID) )
     *
     * 判断准则：
     *   alpha_max < 2°         → FDK 直接适用
     *   2° <= alpha_max < 5°   → 加权 FDK（Parker + 锥角补偿）
     *   alpha_max >= 5°        → Katsevich / WEDGE / 迭代重建
     *
     * @param dZ_det 等中心处探测器 Z 覆盖 (mm)
     * @param SID    源–等中心距 (mm)
     * @return       最大锥角 (rad)
     */
    inline float maxConeAngle(float dZ_det, float SID)
    {
        return std::atan2(dZ_det * 0.5f, SID);
    }

    // ════════════════════════════════════════════════════════════════
    //  综合接口：正向填充扫描几何
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 根据公式体系自动计算并填充螺旋 CT 扫描参数
     *
     * 前提：调用前必须已设置以下字段：
     *   iPV, dv_mm, iPU, du_mm, SID, SDD,
     *   iVZ, vox_z_mm, pitch_mm
     *
     * 填充字段：
     *   start_z_mm  ← 由 scanStartZ() 计算
     *   angle_list  ← 大小 = n_rotations * views_per_rot，均匀角度序列
     *
     * views_per_rot 的选择：
     *   - 工程推荐：直接指定，如 720（0.5°/步）或 1440（0.25°/步）
     *   - 传入 -1  ：退回 Nyquist 理论下界（小像素时结果可能偏大）
     *
     * @tparam TParam        含上述字段的参数结构体类型
     * @param  p             参数结构体（就地修改）
     * @param  pf            Pitch 比（默认 0.75）
     * @param  views_per_rot 每圈投影数；-1 表示使用 Nyquist 理论值
     * @param  verbose       是否打印验证摘要
     */
    template <typename TParam>
    inline void fillHelicalScanGeometry(TParam& p,
        float pf = 0.75f,
        int   views_per_rot = 720,
        float R_check = -1.f,   // 新增：校验用半径，-1 表示用全 FOV
        bool  verbose = true)
    {
        // ── 基础几何 ──────────────────────────────────────────────
        const float M = magnification(p.SID, p.SDD);
        // R 取传入值，或退回全 FOV 半径
        const float R = (R_check > 0.f)
            ? R_check
            : p.iVX * p.vox_x_mm * 0.5f;
        const float dZ_axis = detectorZCoverageAxis(p.iPV, p.dv_mm, M);
        const float dZ_eff = detectorZCoverageAtR(p.iPV, p.dv_mm, p.SID, p.SDD, R);
        const float Z_vol = volumeZRange(p.iVZ, p.vox_z_mm);

        // ── 参数校验 ──────────────────────────────────────────────
        if (p.pitch_mm > pf * dZ_eff)   // 用 pf 留出冗余余量
            throw std::invalid_argument(
                "pitch_mm 超过 pf*dZ_eff(R)，r=R 处将产生漏采样风险");
        // ── 填充扫描几何 ──────────────────────────────────────────
        p.start_z_mm = scanStartZ(Z_vol, dZ_eff, p.pitch_mm);

        const int n_rot = numRotations(Z_vol, dZ_eff, p.pitch_mm);
        const int n_views = views_per_rot;

        p.angle_list.resize(static_cast<size_t>(n_rot) * n_views);
        const float d_angle = 2.f * static_cast<float>(M_PI) / n_views;
        for (int i = 0; i < static_cast<int>(p.angle_list.size()); ++i)
            p.angle_list[i] = i * d_angle;

        // ── verbose 输出 ──────────────────────────────────────────
        if (verbose) {
            const float alpha = maxConeAngle(dZ_axis, p.SID);
            const float alpha_deg = alpha * 180.f / static_cast<float>(M_PI);

            printf("--- Helical CT Param Summary ------------------------\n");
            printf("  M            = %.4f\n", M);
            printf("  R_recon      = %.1f mm\n", R);
            printf("  R_check      = %.1f mm  %s\n",
                R, (R_check > 0.f) ? "(user)" : "(full FOV)");
            printf("  dZ_axis      = %.2f mm  (r=0)\n", dZ_axis);
            printf("  dZ_eff(R)    = %.2f mm  (r=R)\n", dZ_eff);
            printf("  pitch_mm     = %.2f mm  ratio=%.3f  %s\n",
                p.pitch_mm,
                p.pitch_mm / dZ_eff,
                p.pitch_mm / dZ_eff <= 1.f ? "(OK)" : "(WARNING: >1)");
            printf("  Z_vol        = %.1f mm\n", Z_vol);
            printf("  z_start      = %.2f mm\n", p.start_z_mm);
            printf("  n_rotations  = %d\n", n_rot);
            printf("  views/rot    = %d\n", n_views);
            printf("  total_views  = %d\n", (int)p.angle_list.size());
            printf("  alpha_max    = %.3f deg  %s\n",
                alpha_deg,
                alpha_deg < 2.f ? "(FDK OK)" :
                alpha_deg < 5.f ? "(weighted FDK)" : "(need Katsevich)");
            printf("-----------------------------------------------------\n");
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  §11  探测器排数设计（反向：给定圈数上限求最少排数）
    // ════════════════════════════════════════════════════════════════

    /**
     * @brief 联动设计：pitch 由 iPV 决定（pitch = pf * dZ_det），
     *        求满足圈数上限的最少探测器排数及对应 pitch
     *
     * 设计逻辑：
     *   dZ_det = iPV * dv_mm / M
     *   pitch  = pf * dZ_det
     *   N_r    = ceil((Z_vol + 2*dZ_det) / pitch) + 1
     *
     * 对 iPV 以 step 步长递增搜索，直到 N_r <= Nr_max。
     * step 建议取探测器模组单元数（如 16 或 32）。
     *
     * 使用示例：
     * @code
     *   float pitch; int Nr;
     *   int iPV = YK::designDetectorRows(180.f, 0.417f, 2.f, 15, 0.75f, 16,
     *                                     pitch, Nr);
     *   // → iPV=112, pitch≈19mm, Nr=14；工程取2幂次后用 128 排
     * @endcode
     *
     * @param Z_vol        体积 Z 范围 (mm)
     * @param dv_mm        探测器排间距 (mm)
     * @param M            放大比
     * @param Nr_max       旋转圈数上限
     * @param pitch_ratio  Pitch 比 pf，推荐 [0.75, 1.0]
     * @param step         iPV 搜索步长（建议 = 探测器模组单元数）
     * @param out_pitch    [out] 对应的螺旋 pitch (mm)
     * @param out_Nr       [out] 实际旋转圈数；-1 表示无解
     * @return             满足约束的最少探测器排数；-1 表示无解
     */
    inline int designDetectorRows(float Z_vol, float dv_mm, float M,
        int Nr_max, float pitch_ratio,
        int step,
        float& out_pitch, int& out_Nr)
    {
        assert(step > 0);
        for (int iPV = step; iPV <= 1024; iPV += step) {
            const float dZ = detectorZCoverageAxis(iPV, dv_mm, M);
            const float pitch = maxPitch(dZ, pitch_ratio);
            const int   nr = numRotations(Z_vol, dZ, pitch);
            if (nr <= Nr_max) {
                out_pitch = pitch;
                out_Nr = nr;
                return iPV;
            }
        }
        out_pitch = 0.f;
        out_Nr = -1;
        return -1;
    }

    /**
     * @brief 打印探测器排数设计报告（枚举候选排数及关键指标）
     *
     * 输出示例：
     * @code
     *   iPV   dZ_det(mm)  pitch(mm)   N_r   alpha(deg)  status
     *    64      13.4        10.0      22      0.77       OVER
     *    96      20.1        15.0      16      1.15       OK
     *   128      26.7        20.0      12      1.53       OK  <- min
     *   144      30.1        22.5      11      1.72       OK
     * @endcode
     *
     * @param Z_vol   体积 Z 范围 (mm)
     * @param dv_mm   探测器排间距 (mm)
     * @param M       放大比
     * @param SID     源–等中心距 (mm)，用于计算锥角
     * @param Nr_max  旋转圈数上限
     * @param pf      Pitch 比（默认 0.75）
     * @param step    候选排数步长（默认 16）
     */
    inline void printDetectorRowDesign(float Z_vol, float dv_mm, float M,
        float SID, int Nr_max,
        float pf = 0.75f,
        int   step = 16)
    {
        float best_pitch = 0.f;
        int   best_Nr = 0;
        const int min_iPV = designDetectorRows(
            Z_vol, dv_mm, M, Nr_max, pf, step, best_pitch, best_Nr);

        printf("─── Detector Row Design  Nr_max=%d ─────────────────────\n", Nr_max);
        printf("  %-6s  %-12s %-12s %-6s %-12s %s\n",
            "iPV", "dZ_det(mm)", "pitch(mm)", "N_r", "alpha(deg)", "status");
        printf("  %-6s  %-12s %-12s %-6s %-12s %s\n",
            "────", "──────────", "─────────", "───", "──────────", "──────");

        const int show_until = (min_iPV > 0) ? min_iPV + step * 2 : step * 10;
        for (int iPV = step; iPV <= show_until && iPV <= 512; iPV += step) {
            const float dZ = detectorZCoverageAxis(iPV, dv_mm, M);
            const float pitch = maxPitch(dZ, pf);
            const int   nr = numRotations(Z_vol, dZ, pitch);
            const float alpha_deg = maxConeAngle(dZ, SID) * 180.f
                / static_cast<float>(M_PI);
            printf("  %-6d  %-12.1f %-12.1f %-6d %-12.2f %s%s\n",
                iPV, dZ, pitch, nr, alpha_deg,
                nr <= Nr_max ? "OK" : "OVER",
                iPV == min_iPV ? "  <- min" : "");
        }
        printf("────────────────────────────────────────────────────────\n");
        if (min_iPV > 0)
            printf("  推荐排数: %d  pitch=%.1f mm  N_r=%d\n",
                min_iPV, best_pitch, best_Nr);
        else
            printf("  无解：1024 排仍不满足 Nr_max=%d\n", Nr_max);
    }

} // namespace YK
