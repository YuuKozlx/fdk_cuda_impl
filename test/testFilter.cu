#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "YkGlobals.h"
#include "YkFDKFilter.hpp"              // FilterManager (for paddedN/n_complex)
#include "YkFDKCreateFilterKernel.hpp"  // FilterKernelFFT + desc

static void write_raw(const char* path, const void* data, size_t bytes) {
    FILE* fp = std::fopen(path, "wb");
    if (!fp) { std::printf("cannot write %s\n", path); return; }
    std::fwrite(data, 1, bytes, fp);
    std::fclose(fp);
}
static void write_raw_f32(const char* path, const std::vector<float>& v) {
    write_raw(path, v.data(), v.size() * sizeof(float));
}

namespace {

    // ---------------------------
    // dump helper
    // ---------------------------
    static std::string kernel_name(YK::EFilterKernel k) {
        switch (k) {
        case YK::EFilterKernel::RamLak:     return "RamLak";
        case YK::EFilterKernel::SheppLogan: return "SheppLogan";
        case YK::EFilterKernel::Cosine:     return "Cosine";
        case YK::EFilterKernel::Hann:       return "Hann";
        case YK::EFilterKernel::Hamming:    return "Hamming";
        default:                            return "Unknown";
        }
    }

    // ---------------------------
    // (optional) dump spatial rl kernel du=1
    // ---------------------------
    // This matches your kernel_gen_spatial_rl_kernel_du1 in YkFDKCreateFilterKernel.hpp
    __global__ void dump_kernel_gen_spatial_rl_du1(float* h, int N, bool bake_invN) {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        if (u >= N) return;

        int n = (u <= N / 2) ? u : (u - N);
        int an = (n < 0) ? -n : n;

        float val = 0.0f;
        if (n == 0) {
            val = 1.0f / 4.0f;
        }
        else if (an & 1) {
            const float pi = 3.14159265358979323846f;
            val = -1.0f / (pi * pi * (float)(n * n));
        }
        else {
            val = 0.0f;
        }

        if (bake_invN && N > 0) val *= (1.0f / (float)N);
        h[u] = val;
    }

} // anon

int main() {
    // -------------------- config --------------------
    int Nu = 256;
    int Nv = 256;         // batch = Nv
    float du_mm = 1.0f;

    // IMPORTANT: in the latest convention
    // cutoff is in f=k/N in [0,0.5]
    // so "full RL" => cutoff=0.5
    const float cutoff_full = 0.5f;

    // -------------------- stream --------------------
    cudaStream_t s{};
    YK_CUDA_CHECK(cudaStreamCreate(&s));

    // -------------------- get paddedN/n_complex --------------------
    YK::FilterManager fm;
    fm.init(Nu, du_mm, /*batch=*/Nv, s);
    int paddedN = fm.getPaddedN();
    int nC = fm.getNComplex();

    std::printf("Nu=%d Nv=%d  paddedN=%d  n_complex=%d\n", Nu, Nv, paddedN, nC);

    // -------------------- freq axis dump (f = k/N) --------------------
    std::vector<float> hF(nC);
    for (int k = 0; k < nC; ++k) hF[k] = (float)k / (float)paddedN; // [0,0.5]
    write_raw_f32("freq_k_over_N.raw", hF);

    // -------------------- build weights via FilterKernelFFT --------------------
    YK::FilterKernelFFT kfft;
    kfft.prepare(paddedN, s);

    float* d_w = nullptr;
    YK_CUDA_CHECK(cudaMalloc(&d_w, (size_t)nC * sizeof(float)));

    std::vector<float> hW(nC);

    // ---- dump all analytic kernels ----
    const YK::EFilterKernel allKinds[] = {
        YK::EFilterKernel::RamLak,
        YK::EFilterKernel::SheppLogan,
        YK::EFilterKernel::Cosine,
        YK::EFilterKernel::Hann,
        YK::EFilterKernel::Hamming
    };

    for (auto kind : allKinds) {
        YK::FilterKernelDesc desc;
        desc.kind = kind;
        desc.cutoff = cutoff_full;          // [0,0.5]
        desc.gain = 1.0f;
        desc.normalized_ramp = true;        // ramp=f
        desc.force_dc_zero = true;          // align with discrete

        kfft.build_analytic(d_w, desc, /*bake_invN=*/true);

        YK_CUDA_CHECK(cudaMemcpyAsync(hW.data(), d_w, (size_t)nC * sizeof(float),
            cudaMemcpyDeviceToHost, s));
        YK_CUDA_CHECK(cudaStreamSynchronize(s));

        std::string fn = "weights_analytic_" + kernel_name(kind) + ".raw";
        write_raw_f32(fn.c_str(), hW);

        std::printf("\n[Analytic %s] first 16:\n", kernel_name(kind).c_str());
        for (int i = 0; i < 16 && i < nC; ++i) {
            std::printf("k=%2d  w=% .8e\n", i, hW[i]);
        }
    }

    // ---- dump discrete RamLak DU=1 weights (space RL -> FFT -> RealPart) ----
    {
        kfft.build_discrete_ramlak_du1(
            d_w,
            /*bake_invN=*/true,
            YK::FilterKernelFFT::EKernelToWeightsMode::RealPart,
            /*force_dc_zero=*/true);

        YK_CUDA_CHECK(cudaMemcpyAsync(hW.data(), d_w, (size_t)nC * sizeof(float),
            cudaMemcpyDeviceToHost, s));
        YK_CUDA_CHECK(cudaStreamSynchronize(s));

        write_raw_f32("weights_discrete_rl_du1.raw", hW);

        std::printf("\n[Discrete RL DU=1] first 16:\n");
        for (int i = 0; i < 16 && i < nC; ++i) {
            std::printf("k=%2d  w=% .8e\n", i, hW[i]);
        }
    }

    // ---- dump spatial RamLak DU=1 kernel itself (length paddedN) ----
    {
        float* d_h = nullptr;
        YK_CUDA_CHECK(cudaMalloc(&d_h, (size_t)paddedN * sizeof(float)));

        dim3 block(256, 1);
        dim3 grid((paddedN + block.x - 1) / block.x, 1);
        dump_kernel_gen_spatial_rl_du1 << <grid, block, 0, s >> > (d_h, paddedN, /*bake_invN=*/true);
        YK_CUDA_KERNEL_CHECK();

        std::vector<float> hH(paddedN);
        YK_CUDA_CHECK(cudaMemcpyAsync(hH.data(), d_h, (size_t)paddedN * sizeof(float),
            cudaMemcpyDeviceToHost, s));
        YK_CUDA_CHECK(cudaStreamSynchronize(s));

        write_raw_f32("spatial_rl_du1.raw", hH);

        YK_CUDA_CHECK(cudaFree(d_h));
    }

    // -------------------- cleanup --------------------
    YK_CUDA_CHECK(cudaFree(d_w));
    kfft.release();
    fm.release();

    YK_CUDA_CHECK(cudaStreamDestroy(s));
    std::printf("\nDone. Wrote:\n"
        "  freq_k_over_N.raw\n"
        "  weights_analytic_{RamLak,SheppLogan,Cosine,Hann,Hamming}.raw\n"
        "  weights_discrete_rl_du1.raw\n"
        "  spatial_rl_du1.raw\n");
    return 0;
}


//#include <cstdio>
//#include <vector>
//#include <cmath>
//#include <cuda_runtime.h>
//
//#include "YkGlobals.h"
//#include "YkFDKFilter.hpp"              // 你的 FilterManager（或不需要也行）
//#include "YkFDKCreateFilterKernel.hpp"  // FilterKernelFFT
//
//static void write_raw(const char* path, const std::vector<float>& v) {
//    FILE* fp = std::fopen(path, "wb");
//    if (!fp) { std::printf("cannot write %s\n", path); return; }
//    std::fwrite(v.data(), sizeof(float), v.size(), fp);
//    std::fclose(fp);
//}
//
//int main() {
//    int Nu = 256;
//    int Nv = 256;
//    float du_mm = 1.0f;
//
//    cudaStream_t s;
//    YK_CUDA_CHECK(cudaStreamCreate(&s));
//
//    // 只为拿 paddedN / n_complex
//    YK::FilterManager fm;
//    fm.init(Nu, du_mm, /*batch=*/Nv, s);
//    int paddedN = fm.getPaddedN();
//    int nC = fm.getNComplex();
//
//    // --- build weights via FilterKernelFFT directly ---
//    YK::FilterKernelFFT kfft;
//    kfft.prepare(paddedN, s);
//
//    float* d_wA = nullptr;
//    float* d_wD = nullptr;
//    YK_CUDA_CHECK(cudaMalloc(&d_wA, (size_t)nC * sizeof(float)));
//    YK_CUDA_CHECK(cudaMalloc(&d_wD, (size_t)nC * sizeof(float)));
//
//    YK::FilterKernelDesc desc;
//    desc.kind = YK::EFilterKernel::RamLak;
//    desc.cutoff = 1.0f;
//    desc.gain = 1.0f;
//    desc.normalized_ramp = true;
//
//    // analytic (DU=1, bake 1/N)
//    kfft.build_analytic(d_wA, desc, /*bake_invN=*/true);
//
//    // discrete RL (DU=1, bake 1/N) —— 注意：这里要求你实现 build_discrete_ramlak_du1
//    kfft.build_discrete_ramlak_du1(
//        d_wD,
//        /*bake_invN=*/true,
//        YK::FilterKernelFFT::EKernelToWeightsMode::RealPart);
//
//    std::vector<float> hA(nC), hD(nC);
//    YK_CUDA_CHECK(cudaMemcpyAsync(hA.data(), d_wA, (size_t)nC * sizeof(float), cudaMemcpyDeviceToHost, s));
//    YK_CUDA_CHECK(cudaMemcpyAsync(hD.data(), d_wD, (size_t)nC * sizeof(float), cudaMemcpyDeviceToHost, s));
//    YK_CUDA_CHECK(cudaStreamSynchronize(s));
//
//    // dump raw for plotting
//    write_raw("weights_analytic.raw", hA);
//    write_raw("weights_discrete_du1.raw", hD);
//
//    // print first 16
//    std::printf("paddedN=%d, n_complex=%d\n", paddedN, nC);
//    std::printf("First 16 weights (A=analytic, D=discrete_du1):\n");
//    for (int i = 0; i < 16 && i < nC; ++i) {
//        std::printf("k=%2d  A=% .8e  D=% .8e  diff=% .3e\n", i, hA[i], hD[i], (double)(hA[i] - hD[i]));
//    }
//
//    // stats
//    double maxAbs = 0.0;
//    double maxRel = 0.0;
//    double ratioSum = 0.0, ratioSum2 = 0.0;
//    int ratioCnt = 0;
//
//    for (int k = 0; k < nC; ++k) {
//        double a = hA[k];
//        double d = hD[k];
//        double diff = a - d;
//        maxAbs = std::max(maxAbs, std::fabs(diff));
//
//        double denom = std::fabs(d);
//        if (denom > 1e-12) {
//            maxRel = std::max(maxRel, std::fabs(diff) / denom);
//            double r = a / d;
//            ratioSum += r;
//            ratioSum2 += r * r;
//            ratioCnt++;
//        }
//    }
//
//    double ratioMean = (ratioCnt > 0) ? ratioSum / ratioCnt : 0.0;
//    double ratioVar = (ratioCnt > 0) ? (ratioSum2 / ratioCnt - ratioMean * ratioMean) : 0.0;
//
//    std::printf("\nmaxAbsDiff = %.6e\n", maxAbs);
//    std::printf("maxRelDiff = %.6e\n", maxRel);
//    std::printf("ratio mean(A/D)=%.6e, var=%.6e (if var~0 => mostly constant scale)\n",
//        ratioMean, ratioVar);
//
//    YK_CUDA_CHECK(cudaFree(d_wA));
//    YK_CUDA_CHECK(cudaFree(d_wD));
//    kfft.release();
//    fm.release();
//
//    YK_CUDA_CHECK(cudaStreamDestroy(s));
//    return 0;
//}
