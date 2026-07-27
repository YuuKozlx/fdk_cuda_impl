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
            YK::FilterKernelFFT::EKernelToWeightsMode::RealPart);

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
