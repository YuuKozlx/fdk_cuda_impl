#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "Filter/YkCreateFilterKernel.cuh"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"


namespace YKTest {

    using namespace YK;

    static inline bool is_finite_vec(const std::vector<float>& v)
    {
        for (float x : v) {
            if (!std::isfinite(x)) return false;
        }
        return true;
    }

    static inline void dump_weights_compare(
        const float* hA, const float* hB,
        int n_complex, int N,
        int dump_bins)
    {
        std::printf("k   f=k/N     A(analytic)            B(discreteRL)          ratio(B/A)           diff(B-A)\n");
        std::printf("----------------------------------------------------------------------------------------------------\n");
        for (int k = 0; k < std::min(n_complex, dump_bins); ++k) {
            float f = (N > 0) ? (float)k / (float)N : 0.0f;
            float a = hA[k];
            float b = hB[k];
            float ratio = (std::fabs(a) > 1e-20f) ? (b / a) : 0.0f;
            float diff = b - a;
            std::printf("%4d %8.6f  % .8e  % .8e  % .8e  % .8e\n", k, f, a, b, ratio, diff);
        }
    }

    static inline void summarize_diff(
        const float* hA, const float* hB,
        int n_complex,
        bool ignore_dc,
        int& k_max_abs, float& max_abs,
        int& k_max_rel, float& max_rel)
    {
        k_max_abs = -1; max_abs = 0.0f;
        k_max_rel = -1; max_rel = 0.0f;

        int k0 = ignore_dc ? 1 : 0;

        for (int k = k0; k < n_complex; ++k) {
            float a = hA[k];
            float b = hB[k];
            float abs_err = std::fabs(b - a);
            float rel_err = abs_err / (std::fabs(a) + 1e-20f);

            if (abs_err > max_abs) { max_abs = abs_err; k_max_abs = k; }
            if (rel_err > max_rel) { max_rel = rel_err; k_max_rel = k; }
        }
    }

    static inline void summarize_identity(
        const std::vector<float>& w,
        int n_complex,
        int N,
        float expected,
        bool ignore_dc,
        int& k_max_abs, float& max_abs)
    {
        k_max_abs = -1;
        max_abs = 0.0f;
        int k0 = ignore_dc ? 1 : 0;

        for (int k = k0; k < n_complex; ++k) {
            float e = std::fabs(w[k] - expected);
            if (e > max_abs) { max_abs = e; k_max_abs = k; }
        }
    }

    // ------------------------------------------------------------
    // 重写测试：对比 AnalyticFreq vs DiscreteRLFFT（统一新接口）
    // ------------------------------------------------------------
    inline bool testFilterWeightsSpectra_RamLak(
        int Nu = 512,
        int dump_bins = 64,
        bool force_dc_zero = false,
        bool bake_invN = true,
        bool ignore_dc_in_stats = true,
        float pass_max_rel = 5e-3f,
        float pass_max_abs = 1e-6f)
    {
        int paddedN = 1;
        while (paddedN < 2 * Nu) paddedN <<= 1;
        const int n_complex = paddedN / 2 + 1;

        cudaStream_t stream = 0;

        std::printf("[testWeights] Nu=%d paddedN=%d n_complex=%d bake_invN=%d force_dc_zero=%d\n",
            Nu, paddedN, n_complex, (int)bake_invN, (int)force_dc_zero);

        // ---- 分配设备内存 ----
        float* d_w_A = nullptr;
        float* d_w_B = nullptr;
        float* d_w_I = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_w_A, (size_t)n_complex * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_w_B, (size_t)n_complex * sizeof(float)));
        YK_CUDA_CHECK(cudaMalloc(&d_w_I, (size_t)n_complex * sizeof(float)));

        Filter::CreateFilterKernelFromFFT kernel;
        kernel.prepare(paddedN, stream);

        // RAII cleanup，替代 goto
        auto cleanup = [&]() {
            cudaFree(d_w_A);
            cudaFree(d_w_B);
            cudaFree(d_w_I);
            kernel.release();
            };

        // ---- Identity (None) ----
        {
            SFilterKernelDesc descI{};
            descI.kind = EFilterKernel::None;
            descI.gain = 1.0f;
            descI.cutoff = 0.5f;
            descI.force_dc_zero = false;
            descI.source = EWeightsBuildSource::AnalyticFreq;
            kernel.build_weights(d_w_I, descI, bake_invN);
        }

        // ---- AnalyticFreq RamLak ----
        SFilterKernelDesc descA{};
        descA.kind = EFilterKernel::RamLak;
        descA.cutoff = 0.5f;
        descA.gain = 1.0f;
        descA.force_dc_zero = force_dc_zero;
        descA.source = EWeightsBuildSource::AnalyticFreq;
        descA.extract_mode = ERampExtractMode::Magnitude;
        kernel.build_weights(d_w_A, descA, bake_invN);

        // ---- DiscreteRLFFT RamLak ----
        SFilterKernelDesc descB = descA;
        descB.source = EWeightsBuildSource::DiscreteRLFFT;
        kernel.build_weights(d_w_B, descB, bake_invN);

        // ---- Copy back ----
        std::vector<float> hA(n_complex), hB(n_complex), hI(n_complex);
        YK_CUDA_CHECK(cudaMemcpyAsync(hA.data(), d_w_A,
            (size_t)n_complex * sizeof(float), cudaMemcpyDeviceToHost, stream));
        YK_CUDA_CHECK(cudaMemcpyAsync(hB.data(), d_w_B,
            (size_t)n_complex * sizeof(float), cudaMemcpyDeviceToHost, stream));
        YK_CUDA_CHECK(cudaMemcpyAsync(hI.data(), d_w_I,
            (size_t)n_complex * sizeof(float), cudaMemcpyDeviceToHost, stream));
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));

        // ---- Sanity checks ----
        if (!is_finite_vec(hA) || !is_finite_vec(hB) || !is_finite_vec(hI)) {
            std::printf("[testWeights][FAIL] NaN/Inf detected in weights.\n");
            cleanup();
            return false;
        }

        // ---- Identity check ----
        float expected_I = bake_invN ? (1.0f / (float)paddedN) : 1.0f;
        int kI = -1;
        float max_abs_I = 0.0f;
        summarize_identity(hI, n_complex, paddedN, expected_I,
            /*ignore_dc=*/false, kI, max_abs_I);
        std::printf("[testWeights] Identity(None) check: expected=%.8e max_abs=%.3e at k=%d\n",
            expected_I, max_abs_I, kI);

        // ---- Dump first bins ----
        std::printf("\n[testWeights] Dump first %d bins (A=AnalyticFreq, B=DiscreteRLFFT)\n",
            dump_bins);
        dump_weights_compare(hA.data(), hB.data(), n_complex, paddedN, dump_bins);

        // ---- Summary stats ----
        int k_abs = -1, k_rel = -1;
        float max_abs = 0.0f, max_rel = 0.0f;
        summarize_diff(hA.data(), hB.data(), n_complex,
            ignore_dc_in_stats, k_abs, max_abs, k_rel, max_rel);

        float f_abs = (paddedN > 0 && k_abs >= 0) ? (float)k_abs / (float)paddedN : 0.0f;
        float f_rel = (paddedN > 0 && k_rel >= 0) ? (float)k_rel / (float)paddedN : 0.0f;

        std::printf("\n[testWeights] Summary (ignore_dc=%d):\n", (int)ignore_dc_in_stats);
        std::printf("  max_abs_diff = %.6e at k=%d (f=%.6f)  A=%.6e  B=%.6e\n",
            max_abs, k_abs, f_abs,
            (k_abs >= 0 ? hA[k_abs] : 0.0f),
            (k_abs >= 0 ? hB[k_abs] : 0.0f));
        std::printf("  max_rel_diff = %.6e at k=%d (f=%.6f)  A=%.6e  B=%.6e\n",
            max_rel, k_rel, f_rel,
            (k_rel >= 0 ? hA[k_rel] : 0.0f),
            (k_rel >= 0 ? hB[k_rel] : 0.0f));

        // ---- PASS/FAIL ----
        bool pass = true;
        if (max_rel > pass_max_rel) {
            std::printf("[testWeights][WARN] max_rel_diff(%.3e) > threshold(%.3e)\n",
                max_rel, pass_max_rel);
            pass = false;
        }
        if (max_abs > pass_max_abs) {
            std::printf("[testWeights][WARN] max_abs_diff(%.3e) > threshold(%.3e)\n",
                max_abs, pass_max_abs);
            pass = false;
        }
        if (max_abs_I > 1e-6f) {
            std::printf("[testWeights][WARN] identity(None) max_abs(%.3e) too large (expected %.3e)\n",
                max_abs_I, expected_I);
            pass = false;
        }

        std::printf("[testWeights] %s\n", pass ? "PASS" : "FAIL");

        cleanup();
        return pass;
    }

} // namespace YKTest