//#include "Fp/kernels/YkFPHelpers.cuh"
//#include "Fp/YkFPPipelineContext.hpp"
//#include "global/YkCudaTextureController.hpp"
//#include "global/YkMacro.hpp"
//
//#include <vector>
//#include <cstdio>
//#include <cmath>
//#include <fstream>
//#include <string>
//
//
//template<typename DIR>
//__global__ void cone_fp_kernel2(
//    cudaTextureObject_t      volTex,
//    float3 src,
//    float3 detS,        // ����(0,0)���Ͻ���������
//    float3 detU,        // U���򲽳�����
//    float3 detV,        // V���򲽳�����
//    float3 vol_origin,  // ����(0,0,0)��������
//    float  vox0,        // �������سߴ�
//    float  vox1,
//    float  vox2,
//    int    nSlices,     // ����������
//    int    nDim1,
//    int    nDim2,
//    int    Nu, int Nv,
//    float* d_sino,      // [Nv][Nu]������ͶӰ
//    bool   accumulate)
//{
//    const int detU_idx = blockIdx.x * blockDim.x + threadIdx.x;
//    const int detV_idx = blockIdx.y * blockDim.y + threadIdx.y;
//    if (detU_idx >= Nu || detV_idx >= Nv) return;
//
//    // ̽�������������������꣨detS �����Ͻǣ�+0.5���������ģ�
//    const float3 detCenter = {
//        detS.x + (detU_idx + 0.5f) * detU.x + (detV_idx + 0.5f) * detV.x,
//        detS.y + (detU_idx + 0.5f) * detU.y + (detV_idx + 0.5f) * detV.y,
//        detS.z + (detU_idx + 0.5f) * detU.z + (detV_idx + 0.5f) * detV.z
//    };
//
//    // world �� ��������
//    // tex_coord = (world - vol_origin) / vox + 0.5
//    // �ϲ���tex_coord = world / vox - (vol_origin/vox - 0.5)
//    const float invVox0 = 1.f / vox0;
//    const float invVox1 = 1.f / vox1;
//    const float invVox2 = 1.f / vox2;
//
//    const float orig0 = DIR::c0(vol_origin.x, vol_origin.y, vol_origin.z) * invVox0 - 0.5f;
//    const float orig1 = DIR::c1(vol_origin.x, vol_origin.y, vol_origin.z) * invVox1 - 0.5f;
//    const float orig2 = DIR::c2(vol_origin.x, vol_origin.y, vol_origin.z) * invVox2 - 0.5f;
//
//    const float src0 = DIR::c0(src.x, src.y, src.z) * invVox0 - orig0;
//    const float src1 = DIR::c1(src.x, src.y, src.z) * invVox1 - orig1;
//    const float src2 = DIR::c2(src.x, src.y, src.z) * invVox2 - orig2;
//
//    const float det0 = DIR::c0(detCenter.x, detCenter.y, detCenter.z) * invVox0 - orig0;
//    const float det1 = DIR::c1(detCenter.x, detCenter.y, detCenter.z) * invVox1 - orig1;
//    const float det2 = DIR::c2(detCenter.x, detCenter.y, detCenter.z) * invVox2 - orig2;
//
//    // ��������ϵ�²����� ray�����᷽�򲽽�
//    const float inv_d0 = 1.f / (src0 - det0);
//    const float a1 = (src1 - det1) * inv_d0;
//    const float a2 = (src2 - det2) * inv_d0;
//    const float b1 = src1 - a1 * src0;
//    const float b2 = src2 - a2 * src0;
//
//    // ·����������
//    const float fDistCorr = sqrtf(a1 * a1 + a2 * a2 + 1.f) * vox0;
//
//    // ���������
//    float f0 = 0.5f;   // ��һ����������
//    float f1 = a1 * f0 + b1;
//    float f2 = a2 * f0 + b2;
//
//    float fVal = 0.f;
//    for (int s = 0; s < nSlices; ++s)
//    {
//        fVal += DIR::sample(volTex, f0, f1, f2);
//        f0 += 1.f;
//        f1 += a1;
//        f2 += a2;
//    }
//    fVal *= fDistCorr;
//
//    const size_t idx = (size_t)detV_idx * Nu + detU_idx;
//    if (accumulate)
//        d_sino[idx] += fVal;
//    else
//        d_sino[idx] = fVal;
//}
//
//namespace YK {
//    namespace Fp {
//        namespace Test {
//
//            // ----------------------------------------------------------------
//            // ����ѡ�� + kernel dispatch������ͶӰ��
//            // ----------------------------------------------------------------
//            static void dispatchOneView(
//                cudaTextureObject_t      volTex,
//                const SConeProjGeomVec& v,
//                const SVolGeom& g,
//                int Nu, int Nv,
//                float* d_sino_view,
//                bool   accumulate,
//                cudaStream_t stream)
//            {
//                const float3 vol_origin = g.origin();
//                const detail::MainAxis ax = detail::getMainAxis(v);
//
//                dim3 block(16, 16);
//                dim3 grid((Nu + 15) / 16, (Nv + 15) / 16);
//
//                switch (ax) {
//                case detail::MainAxis::X:
//                    cone_fp_kernel2<detail::DirX> << <grid, block, 0, stream >> > (
//                        volTex, v.src, v.detS, v.detU, v.detV, vol_origin,
//                        detail::DirX::vox0(g.vox_x, g.vox_y, g.vox_z),
//                        detail::DirX::vox1(g.vox_x, g.vox_y, g.vox_z),
//                        detail::DirX::vox2(g.vox_x, g.vox_y, g.vox_z),
//                        detail::DirX::nSlices(g.Nx, g.Ny, g.Nz),
//                        detail::DirX::nDim1(g.Nx, g.Ny, g.Nz),
//                        detail::DirX::nDim2(g.Nx, g.Ny, g.Nz),
//                        Nu, Nv, d_sino_view, accumulate);
//                    break;
//                case detail::MainAxis::Y:
//                    cone_fp_kernel2<detail::DirY> << <grid, block, 0, stream >> > (
//                        volTex, v.src, v.detS, v.detU, v.detV, vol_origin,
//                        detail::DirY::vox0(g.vox_x, g.vox_y, g.vox_z),
//                        detail::DirY::vox1(g.vox_x, g.vox_y, g.vox_z),
//                        detail::DirY::vox2(g.vox_x, g.vox_y, g.vox_z),
//                        detail::DirY::nSlices(g.Nx, g.Ny, g.Nz),
//                        detail::DirY::nDim1(g.Nx, g.Ny, g.Nz),
//                        detail::DirY::nDim2(g.Nx, g.Ny, g.Nz),
//                        Nu, Nv, d_sino_view, accumulate);
//                    break;
//                case detail::MainAxis::Z:
//                    cone_fp_kernel2<detail::DirZ> << <grid, block, 0, stream >> > (
//                        volTex, v.src, v.detS, v.detU, v.detV, vol_origin,
//                        detail::DirZ::vox0(g.vox_x, g.vox_y, g.vox_z),
//                        detail::DirZ::vox1(g.vox_x, g.vox_y, g.vox_z),
//                        detail::DirZ::vox2(g.vox_x, g.vox_y, g.vox_z),
//                        detail::DirZ::nSlices(g.Nx, g.Ny, g.Nz),
//                        detail::DirZ::nDim1(g.Nx, g.Ny, g.Nz),
//                        detail::DirZ::nDim2(g.Nx, g.Ny, g.Nz),
//                        Nu, Nv, d_sino_view, accumulate);
//                    break;
//                }
//            }
//
//            // ----------------------------------------------------------------
//            // ���ߣ����� raw float
//            // ----------------------------------------------------------------
//            static bool saveRaw(const std::string& path, const float* data, size_t count)
//            {
//                std::ofstream f(path, std::ios::binary);
//                if (!f) { fprintf(stderr, "[saveRaw] cannot open %s\n", path.c_str()); return false; }
//                f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
//                return true;
//            }
//
//            static bool loadRaw(const std::string& path, float* data, size_t count)
//            {
//                std::ifstream f(path, std::ios::binary);
//                if (!f) { fprintf(stderr, "[loadRaw] cannot open %s\n", path.c_str()); return false; }
//                f.read(reinterpret_cast<char*>(data), count * sizeof(float));
//                return (size_t)f.gcount() == count * sizeof(float);
//            }
//
//            static void test_uniform_volume_with_offset(cudaStream_t stream)
//            {
//                printf("\n[Test_offset] uniform volume with center offset\n");
//
//                constexpr int   Nx = 64, Ny = 64, Nz = 64;
//                constexpr float vox = 1.f;
//
//                constexpr int   Nu = 128, Nv = 128;
//                constexpr float du = 1.f, dv = 1.f;
//                constexpr float SID = 500.f, SDD = 1000.f;
//                const float mag = SDD / SID;   // �Ŵ��� = 2.0
//
//                // ���� view����ǰ����gantry=0��
//                SConeProjGeomVec v;
//                v.src = make_float3(0.f, -SID, 0.f);
//                v.srcCR = make_float3(0.f, 1.f, 0.f);
//                v.detU = make_float3(du, 0.f, 0.f);
//                v.detV = make_float3(0.f, 0.f, dv);
//                v.detS = make_float3(-Nu * 0.5f * du, SDD - SID, -Nv * 0.5f * dv);
//                v.angle = make_float3(0.f, 0.f, 0.f);
//
//                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);
//
//                // ��������ƫ�ã���֤ͶӰ��ֵλ���Ƿ����Ԥ��
//                struct OffsetCase {
//                    float ox, oy, oz;
//                    const char* label;
//                };
//
//                OffsetCase cases[] = {
//                    { 0.f,  0.f,  0.f, "no offset"     },
//                    { 10.f, 0.f,  0.f, "offset x=+10"  },  // U����ƫ�ƣ�ͶӰ��ֵӦ�� Nu/2 + 10*mag
//                    { 0.f,  0.f, 15.f, "offset z=+15"  },  // V����ƫ�ƣ�ͶӰ��ֵӦ�� Nv/2 + 15*mag
//                    { 10.f, 5.f, 15.f, "offset x=10 y=5 z=15" },
//                };
//
//                for (const auto& c : cases)
//                {
//                    SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);
//                    g.center = make_float3(c.ox, c.oy, c.oz);
//
//                    Mem::TextureController tc;
//                    auto volTex = tc.createTex3DFromHost(h_vol.data(), g);
//
//                    const size_t sino_elems = (size_t)Nu * Nv;
//                    float* d_sino = nullptr;
//                    YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
//                    YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));
//
//                    dispatchOneView(volTex.tex, v, g, Nu, Nv, d_sino, false, stream);
//                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
//
//                    std::vector<float> h_sino(sino_elems);
//                    YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
//                        sino_elems * sizeof(float), cudaMemcpyDeviceToHost));
//
//                    // �ҷ�ֵλ�ã�ͶӰ���������ģ�
//                    float sum_u = 0.f, sum_v = 0.f, total = 0.f;
//                    for (int iv = 0; iv < Nv; ++iv)
//                        for (int iu = 0; iu < Nu; ++iu) {
//                            float val = h_sino[iv * Nu + iu];
//                            sum_u += val * iu;
//                            sum_v += val * iv;
//                            total += val;
//                        }
//                    const float centroid_u = sum_u / total;
//                    const float centroid_v = sum_v / total;
//
//                    // �������ģ��������ͶӰ��̽����
//                    // x ƫ�� �� U ����det_u = Nu/2 + ox * mag / du
//                    // z ƫ�� �� V ����det_v = Nv/2 + oz * mag / dv
//                    // y ƫ��ֻӰ��·�����ȣ���Ӱ��ͶӰλ�ã������߷���
//                    const float expect_u = Nu * 0.5f + c.ox * mag / du;
//                    const float expect_v = Nv * 0.5f + c.oz * mag / dv;
//
//                    printf("  [%s]\n", c.label);
//                    printf("    centroid: (%.2f, %.2f)  expected: (%.2f, %.2f)  "
//                        "diff: (%.3f, %.3f)\n",
//                        centroid_u, centroid_v,
//                        expect_u, expect_v,
//                        fabsf(centroid_u - expect_u),
//                        fabsf(centroid_v - expect_v));
//
//                    // ����
//                    char fname[64];
//                    snprintf(fname, sizeof(fname), "test_offset_%s.raw", c.label);
//                    // �滻�ո�����ļ�������
//                    for (char* p = fname; *p; ++p) if (*p == ' ' || *p == '=') *p = '_';
//                    saveRaw(fname, h_sino.data(), sino_elems);
//                    printf("    saved: %s  [%d x %d]\n", fname, Nv, Nu);
//
//                    cudaFree(d_sino);
//                }
//            }
//
//            // ----------------------------------------------------------------
//            // ����1�������壨��������=1������֤����·�������Ƿ����
//            //
//            // Ԥ�ڣ����ڹ����ĵ����ߣ�����ֵ �� ��Խ�����ʵ��·�����ȣ�mm��
//            // ----------------------------------------------------------------
//            static void test_uniform_volume(cudaStream_t stream)
//            {
//                printf("\n[Test1] uniform volume (all voxels = 1.0)\n");
//
//                // С�����������֤
//                constexpr int   Nx = 64, Ny = 64, Nz = 64;
//                constexpr float vox = 1.f;   // 1mm ����
//
//                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);
//
//                // ���ȫ1
//                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);
//
//                // ����̽����
//                constexpr int   Nu = 64, Nv = 64;
//                constexpr float du = 1.f, dv = 1.f;
//                constexpr float SID = 500.f, SDD = 1000.f;
//
//                // ����һ����ǰ���� view��gantry angle = 0��
//                // src �� -y �ᣬ̽������ +y ����
//                SConeProjGeomVec v;
//                v.src = make_float3(0.f, -SID, 0.f);
//                v.srcCR = make_float3(0.f, 1.f, 0.f);   // �������߷��� +y
//                // ̽���������� +y ���� SDD ��
//                // detS = ̽����(0,0)���Ͻ�
//                const float detCX = 0.f;
//                const float detCY = SDD - SID;   // ̽�������ĵ� y����� isocenter��
//                const float detCZ = 0.f;
//                v.detU = make_float3(du, 0.f, 0.f);
//                v.detV = make_float3(0.f, 0.f, dv);
//                // detS = ̽�������� - Nu/2*detU - Nv/2*detV
//                v.detS = make_float3(
//                    detCX - Nu * 0.5f * du,
//                    detCY,
//                    detCZ - Nv * 0.5f * dv);
//                v.angle = make_float3(0.f, 0.f, 0.f);
//
//                // �ϴ����
//                Mem::TextureController tc;
//                auto volTex = tc.createTex3DFromHost(h_vol.data(), g);
//
//                // ���� sinogram�����ţ�
//                const size_t sino_elems = (size_t)Nu * Nv;
//                float* d_sino = nullptr;
//                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
//                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));
//
//                dispatchOneView(volTex.tex, v, g, Nu, Nv, d_sino, false, stream);
//                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
//
//                // ���ؽ��
//                std::vector<float> h_sino(sino_elems);
//                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
//                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));
//
//                // ��֤�������أ�Nu/2, Nv/2��
//                // ���������� y �ᴩԽ 64mm �����·������ = 64mm������=1mm������64����
//                const int cu = Nu / 2, cv = Nv / 2;
//                const float center_val = h_sino[cv * Nu + cu];
//                const float expected = (float)Ny * vox;   // 64mm
//                printf("  center pixel (%d,%d): got=%.4f  expected��%.4f  diff=%.4f\n",
//                    cu, cv, center_val, expected, fabsf(center_val - expected));
//
//                // ͳ�ƣ���Ե����Ӧ�ñ�����С��б�䴩Խ·��������������߽�ضϣ�
//                float sum = 0.f, maxv = 0.f;
//                for (auto x : h_sino) { sum += x; maxv = fmaxf(maxv, x); }
//                printf("  sino sum=%.1f  max=%.4f\n", sum, maxv);
//
//                saveRaw("test1_uniform_sino.raw", h_sino.data(), sino_elems);
//                printf("  saved: test1_uniform_sino.raw  [%d x %d]\n", Nv, Nu);
//
//                cudaFree(d_sino);
//            }
//
//            // ----------------------------------------------------------------
//            // ����2���������أ�����һ������=1������=0������֤ͶӰλ��
//            //
//            // Ԥ�ڣ�sinogram ��ֻ�й�������ĵ������з�����Ӧ
//            // ----------------------------------------------------------------
//            static void test_single_voxel(cudaStream_t stream)
//            {
//                printf("\n[Test2] single voxel at center\n");
//
//                constexpr int   Nx = 64, Ny = 64, Nz = 64;
//                constexpr float vox = 1.f;
//
//                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);
//
//                // ֻ����������Ϊ1
//                std::vector<float> h_vol(Nx * Ny * Nz, 0.f);
//                h_vol[(Nz / 2) * Ny * Nx + (Ny / 2) * Nx + (Nx / 2)] = 1.f;
//
//                constexpr int   Nu = 64, Nv = 64;
//                constexpr float du = 1.f, dv = 1.f;
//                constexpr float SID = 500.f, SDD = 1000.f;
//
//                // ͬ test1 �� view
//                SConeProjGeomVec v;
//                v.src = make_float3(0.f, -SID, 0.f);
//                v.srcCR = make_float3(0.f, 1.f, 0.f);
//                v.detU = make_float3(du, 0.f, 0.f);
//                v.detV = make_float3(0.f, 0.f, dv);
//                v.detS = make_float3(-Nu * 0.5f * du, SDD - SID, -Nv * 0.5f * dv);
//                v.angle = make_float3(0.f, 0.f, 0.f);
//
//                Mem::TextureController tc;
//                auto volTex = tc.createTex3DFromHost(h_vol.data(), g);
//
//                const size_t sino_elems = (size_t)Nu * Nv;
//                float* d_sino = nullptr;
//                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
//                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));
//
//                dispatchOneView(volTex.tex, v, g, Nu, Nv, d_sino, false, stream);
//                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
//
//                std::vector<float> h_sino(sino_elems);
//                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
//                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));
//
//                // �ҷ�������
//                int nonzero = 0;
//                int peak_u = -1, peak_v = -1;
//                float peak_val = 0.f;
//                for (int iv = 0; iv < Nv; ++iv)
//                    for (int iu = 0; iu < Nu; ++iu) {
//                        float val = h_sino[iv * Nu + iu];
//                        if (val > 1e-6f) {
//                            ++nonzero;
//                            if (val > peak_val) { peak_val = val; peak_u = iu; peak_v = iv; }
//                        }
//                    }
//
//                printf("  nonzero pixels: %d\n", nonzero);
//                printf("  peak at (%d,%d) = %.6f  (expected center (%d,%d))\n",
//                    peak_u, peak_v, peak_val, Nu / 2, Nv / 2);
//
//                saveRaw("test2_single_voxel_sino.raw", h_sino.data(), sino_elems);
//                printf("  saved: test2_single_voxel_sino.raw  [%d x %d]\n", Nv, Nu);
//
//                cudaFree(d_sino);
//            }
//
//
//            // ----------------------------------------------------------------
//            // ���������
//            // ----------------------------------------------------------------
//            inline void runAllTests()
//            {
//                cudaStream_t stream;
//                YK_CUDA_CHECK(cudaStreamCreate(&stream));
//
//                test_uniform_volume(stream);
//                test_single_voxel(stream);
//
//
//                YK_CUDA_CHECK(cudaStreamDestroy(stream));
//                printf("\n[Test] all done\n");
//            }
//
//        } // namespace Test
//    } // namespace Fp
//} // namespace YK
//
//
//int main() {
//    YK::Fp::Test::test_uniform_volume(0);
//    YK::Fp::Test::test_uniform_volume_with_offset(0);
//
//    return 0;
//}

//#include "Fp/kernels/YkFPHelpers.cuh"
//#include "Fp/kernels//YkFPLaunch.cuh"
//#include "global/YkCudaTextureController.hpp"
//#include "global/YkMacro.hpp"
//
//
//#include <vector>
//#include <cstdio>
//#include <cmath>
//#include <fstream>
//#include <algorithm>
//#include "FDK/YkVecGeo.hpp"
//

//
// 
// 
#include "Fp/kernels/YkFPSiddonLaunch.cuh"
#include "FP/kernels/YkFPLaunch.cuh"
#include "FP/kernels/YkFPHelpers.cuh"
#include "global/YkCudaTextureController.hpp"
#include "global/YkMacro.hpp"


#include <vector>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include "common/YkVecGeo.hpp"
#include "FP/kernels/YkFPCVPLaunch.cuh"
#include "util/YkCudatimer.hpp"

namespace YK {
    namespace Fp {
        namespace SiddonTest {

            // ----------------------------------------------------------------
            // ����
            // ----------------------------------------------------------------
            static bool saveRaw(const char* path, const float* data, size_t count)
            {
                std::ofstream f(path, std::ios::binary);
                if (!f) { fprintf(stderr, "[saveRaw] cannot open %s\n", path); return false; }
                f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
                return true;
            }

            static bool loadRaw(const char* path, std::vector<float>& buf, size_t count)
            {
                buf.resize(count);
                std::ifstream f(path, std::ios::binary);
                if (!f) { fprintf(stderr, "[loadRaw] cannot open %s\n", path); return false; }
                f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
                return (size_t)f.gcount() == count * sizeof(float);
            }

            static void printSinoStats(const std::vector<float>& sino,
                int Na, int Nv, int Nu,
                const char* tag)
            {
                float minv = 1e30f, maxv = -1e30f, sumv = 0.f;
                int   nnan = 0, ninf = 0, nneg = 0;
                for (float x : sino) {
                    if (std::isnan(x)) { ++nnan; continue; }
                    if (std::isinf(x)) { ++ninf; continue; }
                    if (x < 0.f) ++nneg;
                    minv = fminf(minv, x);
                    maxv = fmaxf(maxv, x);
                    sumv += x;
                }
                printf("  [%s] min=%.4f  max=%.4f  sum=%.3e  nan=%d  inf=%d  neg=%d\n",
                    tag, minv, maxv, sumv, nnan, ninf, nneg);
            }

            // ----------------------------------------------------------------
            // ����1�������壬���Ƕȣ���֤��������·������
            //
            // Ԥ�ڣ��������� �� Ny * vox_y����Խ�����·�����ȣ�
            // ----------------------------------------------------------------
            static void test1_uniform_single(cudaStream_t stream)
            {
                printf("\n[Siddon-Test1] uniform volume, single view\n");

                constexpr int   Nx = 64, Ny = 64, Nz = 64;
                constexpr float vox = 1.f;
                constexpr int   Nu = 64, Nv = 64;
                constexpr float du = 1.f, dv = 1.f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);

                // ������
                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);
                float* d_vol = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
                YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
                    h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));

                // ������ǰ�� view��src �� -y��
                SConeProjGeomVec view;
                view.src = make_float4(0.f, -SID, 0.f, 0.f);
                view.srcCR = make_float4(0.f, 1.f, 0.f, 0.f);
                view.detU = make_float4(du, 0.f, 0.f, 0.f);
                view.detV = make_float4(0.f, 0.f, dv, 0.f);
                view.detS = make_float4(-Nu * 0.5f * du, SDD - SID, -Nv * 0.5f * dv, 0.f);
                view.angle = make_float4(0.f, 0.f, 0.f, 0.f);

                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, &view,
                    sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));

                const size_t sino_elems = (size_t)1 * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                fp_siddon_launch(d_vol, d_sino, d_views,
                    g, Nu, Nv, 1, false, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                // ����������֤
                const float center = h_sino[(Nv / 2) * Nu + Nu / 2];
                const float expect = (float)Ny * vox;
                printf("  center pixel: got=%.4f  expected=%.4f  diff=%.6f\n",
                    center, expect, fabsf(center - expect));

                // б������Ӧ�ø���·��������
                const float corner = h_sino[0];
                printf("  corner pixel (0,0): %.4f  (should be > center for cone beam)\n",
                    corner);

                printSinoStats(h_sino, 1, Nv, Nu, "sino");
                saveRaw("siddon_test1_single.raw", h_sino.data(), sino_elems);
                printf("  saved: siddon_test1_single.raw [%d x %d]\n", Nv, Nu);

                cudaFree(d_vol);
                cudaFree(d_views);
                cudaFree(d_sino);
            }

            // ----------------------------------------------------------------
            // ����2���������أ����ģ�����֤ͶӰλ��
            //
            // Ԥ�ڣ�ֻ�й�������ĵ���������Ӧ����ֵ��̽��������
            // ----------------------------------------------------------------
            static void test2_single_voxel(cudaStream_t stream)
            {
                printf("\n[Siddon-Test2] single voxel at center\n");

                constexpr int   Nx = 64, Ny = 64, Nz = 64;
                constexpr float vox = 1.f;
                constexpr int   Nu = 64, Nv = 64;
                constexpr float du = 1.f, dv = 1.f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);

                std::vector<float> h_vol(Nx * Ny * Nz, 0.f);
                h_vol[(Nz / 2) * Ny * Nx + (Ny / 2) * Nx + (Nx / 2)] = 1.f;

                float* d_vol = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
                YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
                    h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));

                SConeProjGeomVec view;
                view.src = make_float4(0.f, -SID, 0.f, 0.f);
                view.srcCR = make_float4(0.f, 1.f, 0.f, 0.f);
                view.detU = make_float4(du, 0.f, 0.f, 0.f);
                view.detV = make_float4(0.f, 0.f, dv, 0.f);
                view.detS = make_float4(-Nu * 0.5f * du, SDD - SID, -Nv * 0.5f * dv, 0.f);
                view.angle = make_float4(0.f, 0.f, 0.f, 0.f);

                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, &view,
                    sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));

                const size_t sino_elems = (size_t)Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                fp_siddon_launch(d_vol, d_sino, d_views,
                    g, Nu, Nv, 1, false, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                // �ҷ�ֵλ��
                int peak_u = -1, peak_v = -1;
                float peak_val = 0.f;
                int nonzero = 0;
                for (int iv = 0; iv < Nv; ++iv)
                    for (int iu = 0; iu < Nu; ++iu) {
                        float val = h_sino[iv * Nu + iu];
                        if (val > 1e-8f) ++nonzero;
                        if (val > peak_val) {
                            peak_val = val;
                            peak_u = iu; peak_v = iv;
                        }
                    }

                printf("  nonzero pixels: %d\n", nonzero);
                printf("  peak at (%d,%d)=%.6f  expected center (%d,%d)\n",
                    peak_u, peak_v, peak_val, Nu / 2, Nv / 2);

                saveRaw("siddon_test2_single_voxel.raw", h_sino.data(), sino_elems);
                printf("  saved: siddon_test2_single_voxel.raw [%d x %d]\n", Nv, Nu);

                cudaFree(d_vol);
                cudaFree(d_views);
                cudaFree(d_sino);
            }

            // ----------------------------------------------------------------
            // ����3�������壬��Ƕȣ���֤���Ƕ�һ����
            //
            // Ԥ�ڣ����ξ���������Ƕ���������һ�£�
            //       �������򼸺���״����ͬ�Ƕ�·�������в���
            // ----------------------------------------------------------------
            static void test3_multi_view(cudaStream_t stream)
            {
                printf("\n[Siddon-Test3] uniform volume, 360 views\n");

                constexpr int   Nx = 64, Ny = 64, Nz = 64;
                constexpr float vox = 1.f;
                constexpr int   Na = 360;
                constexpr int   Nu = 64, Nv = 64;
                constexpr float du = 1.f, dv = 1.f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);
                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);

                float* d_vol = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
                YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
                    h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));

                // �������ȽǶ�
                std::vector<float> angles(Na);
                for (int i = 0; i < Na; ++i)
                    angles[i] = 2.f * CUDA_PI * i / Na;

                std::vector<SConeProjGeomVec>    h_views(Na);
                std::vector<SFDKGeoParamPerView> h_gv(Na);
                build_circular_vec_geometry_from_theta(
                    h_views, angles, Na, Nu, Nv, du, dv,
                    SID, SDD - SID,
                    f3(0.f, 0.f, 0.f),
                    f3(0.f, 0.f, 0.f));

                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, Na * sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, h_views.data(),
                    Na * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));

                const size_t sino_elems = (size_t)Na * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                fp_siddon_launch(d_vol, d_sino, d_views,
                    g, Nu, Nv, Na, false, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                // ���Ƕ���������ͳ��
                std::vector<float> center_vals(Na);
                for (int a = 0; a < Na; ++a)
                    center_vals[a] = h_sino[((size_t)a * Nv + Nv / 2) * Nu + Nu / 2];

                const float cmin = *std::min_element(center_vals.begin(), center_vals.end());
                const float cmax = *std::max_element(center_vals.begin(), center_vals.end());
                const float cmean = std::accumulate(center_vals.begin(),
                    center_vals.end(), 0.f) / Na;
                printf("  center pixel across %d angles:\n", Na);
                printf("    min=%.4f  max=%.4f  mean=%.4f  range=%.4f\n",
                    cmin, cmax, cmean, cmax - cmin);

                printSinoStats(h_sino, Na, Nv, Nu, "full sino");
                saveRaw("siddon_test3_multi.raw", h_sino.data(), sino_elems);
                printf("  saved: siddon_test3_multi.raw [%d x %d x %d]\n", Na, Nv, Nu);

                cudaFree(d_vol);
                cudaFree(d_views);
                cudaFree(d_sino);
            }

            // ----------------------------------------------------------------
            // ����4��ƫ����֤
            //
            // Ԥ�ڣ�ͶӰ�������������ƫ�������ƶ�
            // ----------------------------------------------------------------
            static void test4_offset(cudaStream_t stream)
            {
                printf("\n[Siddon-Test4] offset validation\n");

                constexpr int   Nx = 64, Ny = 64, Nz = 64;
                constexpr float vox = 1.f;
                constexpr int   Nu = 128, Nv = 128;
                constexpr float du = 1.f, dv = 1.f;
                constexpr float SID = 500.f, SDD = 1000.f;
                const float     mag = SDD / SID;

                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);

                SConeProjGeomVec view;
                view.src = make_float4(0.f, -SID, 0.f, 0.f);
                view.srcCR = make_float4(0.f, 1.f, 0.f, 0.f);
                view.detU = make_float4(du, 0.f, 0.f, 0.f);
                view.detV = make_float4(0.f, 0.f, dv, 0.f);
                view.detS = make_float4(-Nu * 0.5f * du, SDD - SID, -Nv * 0.5f * dv, 0.f);
                view.angle = make_float4(0.f, 0.f, 0.f, 0.f);

                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, &view,
                    sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));

                struct Case { float ox, oy, oz; const char* label; };
                Case cases[] = {
                    {  0.f,  0.f,  0.f, "no_offset"  },
                    { 10.f,  0.f,  0.f, "offset_x10" },
                    {  0.f,  0.f, 15.f, "offset_z15" },
                    { 10.f,  5.f, 15.f, "offset_xyz" },
                };

                const size_t sino_elems = (size_t)Nv * Nu;

                for (auto& c : cases)
                {
                    SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);
                    g.center = make_float3(c.ox, c.oy, c.oz);

                    float* d_vol = nullptr;
                    YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
                    YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
                        h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));

                    float* d_sino = nullptr;
                    YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                    YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                    fp_siddon_launch(d_vol, d_sino, d_views,
                        g, Nu, Nv, 1, false, stream);
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                    std::vector<float> h_sino(sino_elems);
                    YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                        sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                    // ���ļ���
                    float su = 0.f, sv = 0.f, total = 0.f;
                    for (int iv = 0; iv < Nv; ++iv)
                        for (int iu = 0; iu < Nu; ++iu) {
                            float val = h_sino[iv * Nu + iu];
                            su += val * (iu + 0.5f);
                            sv += val * (iv + 0.5f);
                            total += val;
                        }
                    const float cu = su / total;
                    const float cv = sv / total;

                    // ��������
                    const float eu = Nu * 0.5f + c.ox * mag / du;
                    const float ev = Nv * 0.5f + c.oz * mag / dv;

                    printf("  [%s] centroid=(%.2f,%.2f)  expected=(%.2f,%.2f)"
                        "  diff=(%.3f,%.3f)\n",
                        c.label, cu, cv, eu, ev,
                        fabsf(cu - eu), fabsf(cv - ev));

                    char fname[64];
                    snprintf(fname, sizeof(fname), "siddon_test4_%s.raw", c.label);
                    saveRaw(fname, h_sino.data(), sino_elems);

                    cudaFree(d_vol);
                    cudaFree(d_sino);
                }

                cudaFree(d_views);
            }

            // ----------------------------------------------------------------
            // ����5����ʵ���
            // ----------------------------------------------------------------
            static void test5_real_volume(cudaStream_t stream)
            {
                printf("\n[Siddon-Test5] real volume: fdk_vec_vol_online.raw\n");

                constexpr int   Nx = 512, Ny = 512, Nz = 400;
                constexpr float vox_xy = 0.25f, vox_z = 0.25f;
                constexpr int   Na = 360, Nu = 1024, Nv = 1024;
                constexpr float du = 0.25f, dv = 0.25f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox_xy, vox_z);
                g.center = make_float3(0.f, 0.f, 0.f);  // ʵ��ƫ��������

                std::vector<float> h_vol;
                if (!loadRaw("fdk_vec_vol_offline.raw", h_vol, (size_t)Nx * Ny * Nz)) return;
                printf("  volume loaded\n");

                float* d_vol = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
                YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
                    h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));
                h_vol.clear();

                std::vector<float> angles(Na);
                for (int i = 0; i < Na; ++i)
                    angles[i] = 2.f * CUDA_PI * i / Na;

                std::vector<SConeProjGeomVec>    h_views(Na);
                std::vector<SFDKGeoParamPerView> h_gv(Na);
                build_circular_vec_geometry_from_theta(
                    h_views, angles, Na, Nu, Nv, du, dv,
                    SID, SDD - SID,
                    f3(0.f, 0.f, 0.f),
                    f3(0.f, 0.f, 0.f));

                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, Na * sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, h_views.data(),
                    Na * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));
                printf("  geometry ready\n");

                const size_t sino_elems = (size_t)Na * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                printf("  launching Siddon FP: Na=%d Nu=%d Nv=%d...\n", Na, Nu, Nv);
                {
                    YK::Util::CudaTimer t{ "siddon",stream };
                    fp_siddon_launch(d_vol, d_sino, d_views,
                        g, Nu, Nv, Na, false, stream);
                }
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                printf("  done\n");

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                printSinoStats(h_sino, Na, Nv, Nu, "real sino");
                saveRaw("siddon_test5_real.raw", h_sino.data(), sino_elems);
                printf("  saved: siddon_test5_real.raw [%d x %d x %d]\n", Na, Nv, Nu);

                cudaFree(d_vol);
                cudaFree(d_views);
                cudaFree(d_sino);
            }

            // ----------------------------------------------------------------
            // �����
            // ----------------------------------------------------------------
            inline void runSiddonTests()
            {
                cudaStream_t stream;
                YK_CUDA_CHECK(cudaStreamCreate(&stream));

                //test1_uniform_single(stream);
                //test2_single_voxel(stream);
                //test3_multi_view(stream);
                //test4_offset(stream);
                test5_real_volume(stream);

                YK_CUDA_CHECK(cudaStreamDestroy(stream));
                printf("\n[Siddon] all tests done\n");
            }

        } // namespace SiddonTest
    } // namespace Fp
} // namespace YK


namespace YK {
    namespace Fp {
        namespace RawTest {
            using namespace YK::Fp;

            // ----------------------------------------------------------------
            // ���ߺ���
            // ----------------------------------------------------------------
            static bool saveRaw(const char* path, const float* data, size_t count)
            {
                std::ofstream f(path, std::ios::binary);
                if (!f) { fprintf(stderr, "[saveRaw] cannot open %s\n", path); return false; }
                f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
                return true;
            }

            static bool loadRaw(const char* path, std::vector<float>& buf, size_t count)
            {
                buf.resize(count);
                std::ifstream f(path, std::ios::binary);
                if (!f) { fprintf(stderr, "[loadRaw] cannot open %s\n", path); return false; }
                f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
                return (size_t)f.gcount() == count * sizeof(float);
            }


            // ----------------------------------------------------------------
            // ����1�������壬���Ƕȣ���֤��������·������
            // ----------------------------------------------------------------
            static void test_uniform_single_view(cudaStream_t stream)
            {
                printf("\n[RawTest1] uniform volume, single view\n");

                constexpr int   Nx = 64, Ny = 64, Nz = 64;
                constexpr float vox = 0.1f;
                constexpr int   Nu = 64, Nv = 64;
                constexpr float du = 1.f, dv = 1.f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);

                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);

                // ���� view��gantry=0��src �� -y
                SConeProjGeomVec view;
                view.src = make_float4(0.f, -SID, 0.f, 0.f);
                view.srcCR = make_float4(0.f, 1.f, 0.f, 0.f);
                view.detU = make_float4(du, 0.f, 0.f, 0.f);
                view.detV = make_float4(0.f, 0.f, dv, 0.f);
                view.detS = make_float4(-Nu * 0.5f * du, SDD - SID, -Nv * 0.5f * dv, 0.f);
                view.angle = make_float4(0.f, 0.f, 0.f, 0.f);

                std::vector<SConeProjGeomVec> h_views = { view };

                h_views = Fp::normalizeToVoxelBatch(h_views, g);

                // �ϴ� views
                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, h_views.data(),
                    sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));

                // �ϴ���� texture

                auto volTex = Mem::TextureController::createTex3DFromHost(h_vol.data(), g);

                // ���� sinogram [1][Nv][Nu]
                const size_t sino_elems = (size_t)1 * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                fp_joseph_launch(volTex.tex, h_views, d_views, d_sino, g,
                    1, Nu, Nv, false, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                // ��������Ӧ �� Ny * vox = 64mm
                const float center = h_sino[(Nv / 2) * Nu + Nu / 2];
                const float expect = (float)Ny * vox;
                printf("  center pixel: got=%.4f  expected=%.4f  diff=%.4f\n",
                    center, expect, fabsf(center - expect));

                saveRaw("rawtest1_single_view.raw", h_sino.data(), sino_elems);
                printf("  saved: rawtest1_single_view.raw [%d x %d]\n", Nv, Nu);

                cudaFree(d_views);
                cudaFree(d_sino);
            }

            // ----------------------------------------------------------------
            // ����2�������壬��Ƕȣ���֤���Ƕ�ͶӰһ����
            // ������������Ƕȵ���������Ӧ��ͬ��
            // ----------------------------------------------------------------
            static void test_uniform_multi_view(cudaStream_t stream)
            {
                printf("\n[RawTest2] uniform volume, multi view (360 angles)\n");

                constexpr int   Nx = 64, Ny = 64, Nz = 64;
                constexpr float vox = 0.1f;
                constexpr int   Na = 360;
                constexpr int   Nu = 64, Nv = 64;
                constexpr float du = 1.f, dv = 1.f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);
                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);

                // �������ȽǶ��б�
                std::vector<float> angles(Na);
                for (int i = 0; i < Na; ++i)
                    angles[i] = 2.f * CUDA_PI * i / Na;

                // ���� views
                std::vector<SConeProjGeomVec> h_views(Na);
                std::vector<SFDKGeoParamPerView> h_gv(Na);
                build_circular_vec_geometry_from_theta(
                    h_views, angles, Na, Nu, Nv, du, dv,
                    SID, SDD - SID,
                    f3(0.f, 0.f, 0.f),
                    f3(0.f, 0.f, 0.f));


                h_views = Fp::normalizeToVoxelBatch(h_views, g);

                // �ϴ� views
                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, Na * sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, h_views.data(),
                    Na * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));


                auto volTex = Mem::TextureController::createTex3DFromHost(h_vol.data(), g);

                const size_t sino_elems = (size_t)Na * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                fp_joseph_launch(volTex.tex, h_views, d_views, d_sino, g,
                    Na, Nu, Nv, false, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                // ��֤ÿ���Ƕ��������ص�һ����
                float minVal = 1e9f, maxVal = -1e9f, sumVal = 0.f;
                for (int a = 0; a < Na; ++a) {
                    float v = h_sino[((size_t)a * Nv + Nv / 2) * Nu + Nu / 2];
                    minVal = fminf(minVal, v);
                    maxVal = fmaxf(maxVal, v);
                    sumVal += v;
                }
                printf("  center pixel across %d angles: min=%.4f  max=%.4f  "
                    "mean=%.4f  range=%.4f\n",
                    Na, minVal, maxVal, sumVal / Na, maxVal - minVal);
                printf("  (range should be small for uniform sphere, "
                    "larger for cube due to geometry)\n");

                saveRaw("rawtest2_multi_view.raw", h_sino.data(), sino_elems);
                printf("  saved: rawtest2_multi_view.raw [%d x %d x %d]  Na x Nv x Nu\n",
                    Na, Nv, Nu);

                cudaFree(d_views);
                cudaFree(d_sino);
            }

            // ----------------------------------------------------------------
            // ����3��ƫ����֤������������ƫ�ƣ���֤ͶӰ����ƫ��
            // ----------------------------------------------------------------
            static void test_offset(cudaStream_t stream)
            {
                printf("\n[RawTest3] offset validation\n");

                constexpr int   Nx = 64, Ny = 64, Nz = 64;
                constexpr float vox = 0.1f;
                constexpr int   Nu = 128, Nv = 128;
                constexpr float du = 1.f, dv = 1.f;
                constexpr float SID = 500.f, SDD = 1000.f;
                const float     mag = SDD / SID;

                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);

                // ������ǰ�� view
                SConeProjGeomVec view;
                view.src = make_float4(0.f, -SID, 0.f, 0.f);
                view.srcCR = make_float4(0.f, 1.f, 0.f, 0.f);
                view.detU = make_float4(du, 0.f, 0.f, 0.f);
                view.detV = make_float4(0.f, 0.f, dv, 0.f);
                view.detS = make_float4(-Nu * 0.5f * du, SDD - SID, -Nv * 0.5f * dv, 0.f);
                view.angle = make_float4(0.f, 0.f, 0.f, 0.f);

                std::vector<SConeProjGeomVec> h_views = { view };



                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, h_views.data(),
                    sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));

                struct Case { float ox, oy, oz; const char* label; };
                Case cases[] = {
                    {  0.f,  0.f,  0.f, "no_offset"  },
                    { 10.f,  0.f,  0.f, "offset_x10" },
                    {  0.f,  0.f, 15.f, "offset_z15" },
                    { 10.f,  5.f, 15.f, "offset_xyz" },
                };

                const size_t sino_elems = (size_t)Nv * Nu;

                for (auto& c : cases)
                {
                    SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);
                    g.center = make_float3(c.ox, c.oy, c.oz);


                    auto volTex = Mem::TextureController::createTex3DFromHost(h_vol.data(), g);

                    float* d_sino = nullptr;
                    YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                    YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                    h_views = Fp::normalizeToVoxelBatch(h_views, g);

                    fp_joseph_launch(volTex.tex, h_views, d_views, d_sino, g,
                        1, Nu, Nv, false, stream);
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                    std::vector<float> h_sino(sino_elems);
                    YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                        sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                    // ��������
                    float su = 0.f, sv = 0.f, total = 0.f;
                    for (int iv = 0; iv < Nv; ++iv)
                        for (int iu = 0; iu < Nu; ++iu) {
                            float val = h_sino[iv * Nu + iu];
                            su += val * (iu + 0.5f);
                            sv += val * (iv + 0.5f);
                            total += val;
                        }
                    const float cu = su / total;
                    const float cv = sv / total;

                    // ��������
                    const float eu = Nu * 0.5f + c.ox * mag / du;
                    const float ev = Nv * 0.5f + c.oz * mag / dv;

                    printf("  [%s] centroid=(%.2f,%.2f) expected=(%.2f,%.2f) "
                        "diff=(%.3f,%.3f)\n",
                        c.label, cu, cv, eu, ev,
                        fabsf(cu - eu), fabsf(cv - ev));

                    char fname[64];
                    snprintf(fname, sizeof(fname), "rawtest3_%s.raw", c.label);
                    saveRaw(fname, h_sino.data(), sino_elems);

                    cudaFree(d_sino);
                }

                cudaFree(d_views);
            }

            // ----------------------------------------------------------------
            // ����4����ʵ�����ȫ�Ƕ�
            // ----------------------------------------------------------------
            static void test_real_volume(cudaStream_t stream)
            {
                printf("\n[RawTest4] real volume: fdk_vec_vol_online.raw\n");

                constexpr int   Nx = 512, Ny = 512, Nz = 400;
                constexpr float vox_xy = 0.25f, vox_z = 0.25f;
                constexpr int   Na = 360, Nu = 1024, Nv = 1024;
                constexpr float du = 0.25f, dv = 0.25f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox_xy, vox_z);
                g.center = make_float3(0.f, 0.f, 0.f);  // ʵ��ƫ��������

                std::vector<float> h_vol;
                if (!loadRaw("fdk_vec_vol_offline.raw", h_vol, (size_t)Nx * Ny * Nz)) return;
                printf("  volume loaded\n");

                std::vector<float> angles(Na);
                for (int i = 0; i < Na; ++i)
                    angles[i] = 2.f * CUDA_PI * i / 360;

                std::vector<SConeProjGeomVec>    h_views(Na);
                std::vector<SFDKGeoParamPerView> h_gv(Na);
                build_circular_vec_geometry_from_theta(
                    h_views, angles, Na, Nu, Nv, du, dv,
                    SID, SDD - SID,
                    f3(0.f, 0.f, 0.f),
                    f3(0.f, 0.f, 0.f));

                // build_circular_vec_geometry_from_theta ֮�󣬹�һ��֮ǰ��ӡ����
                for (int a : {0}) {
                    if (a < Na) continue;
                    printf("  view[%d] srcCR=(%.3f,%.3f,%.3f)\n",
                        a, h_views[a].srcCR.x, h_views[a].srcCR.y, h_views[a].srcCR.z);
                }
                // ��һ�����ٴ�ӡ
                h_views = Fp::normalizeToVoxelBatch(h_views, g);
                for (int a : {0}) {
                    if (a < Na) continue;
                    printf("  view_vox[%d] src=(%.2f,%.2f,%.2f)\n",
                        a, h_views[a].src.x, h_views[a].src.y, h_views[a].src.z);
                }

                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, Na * sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, h_views.data(),
                    Na * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));


                auto volTex = Mem::TextureController::createTex3DFromHost(h_vol.data(), g);
                h_vol.clear();
                printf("  texture bound\n");

                const size_t sino_elems = (size_t)Na * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                printf("  launching... Na=%d Nu=%d Nv=%d\n", Na, Nu, Nv);
                {
                    YK::Util::CudaTimer t{ "joseph",stream };
                    fp_joseph_launch(volTex.tex, h_views, d_views, d_sino, g,
                        Na, Nu, Nv, false, stream, FpStepSuperSample::x1);
                }

                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                printf("  done\n");

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                for (int a = 0; a < Na; ++a) {
                    float maxv = 0.f;
                    for (size_t k = 0; k < (size_t)Nv * Nu; ++k)
                        maxv = fmaxf(maxv, h_sino[a * (size_t)Nv * Nu + k]);
                    if (maxv > 1e-6f || a < 5 || a >= Na - 5) {
                        //printf("  angle %3d: max=%.4f\n", a, maxv);
                    }

                }

                float maxv = *std::max_element(h_sino.begin(), h_sino.end());
                float sumv = 0.f;
                for (auto x : h_sino) sumv += x;
                printf("  sino: max=%.4f  sum=%.3e\n", maxv, sumv);

                saveRaw("rawtest4_real_sino.raw", h_sino.data(), sino_elems);
                printf("  saved: rawtest4_real_sino.raw [%d x %d x %d]\n", Na, Nv, Nu);

                cudaFree(d_views);
                cudaFree(d_sino);
            }

            // ����4����ʵ�����ȫ�Ƕ�
           // ----------------------------------------------------------------
            static void test_real_volume_ss(cudaStream_t stream)
            {
                printf("\n[RawTest5] real volume: fdk_vec_vol_online.raw\n");

                constexpr int   Nx = 512, Ny = 512, Nz = 400;
                constexpr float vox_xy = 0.25f, vox_z = 0.25f;
                constexpr int   Na = 360, Nu = 1024, Nv = 1024;
                constexpr float du = 0.25f, dv = 0.25f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox_xy, vox_z);
                g.center = make_float3(0.f, 0.f, 0.f);  // ʵ��ƫ��������

                std::vector<float> h_vol;
                if (!loadRaw("fdk_vec_vol_offline.raw", h_vol, (size_t)Nx * Ny * Nz)) return;
                printf("  volume loaded\n");

                std::vector<float> angles(Na);
                for (int i = 0; i < Na; ++i)
                    angles[i] = 2.f * CUDA_PI * i / 360;

                std::vector<SConeProjGeomVec>    h_views(Na);
                std::vector<SFDKGeoParamPerView> h_gv(Na);
                build_circular_vec_geometry_from_theta(
                    h_views, angles, Na, Nu, Nv, du, dv,
                    SID, SDD - SID,
                    f3(0.f, 0.f, 0.f),
                    f3(0.f, 0.f, 0.f));

                // build_circular_vec_geometry_from_theta ֮�󣬹�һ��֮ǰ��ӡ����
                for (int a : {0}) {
                    if (a < Na) continue;
                    printf("  view[%d] srcCR=(%.3f,%.3f,%.3f)\n",
                        a, h_views[a].srcCR.x, h_views[a].srcCR.y, h_views[a].srcCR.z);
                }
                // ��һ�����ٴ�ӡ
                h_views = Fp::normalizeToVoxelBatch(h_views, g);
                for (int a : {0}) {
                    if (a < Na) continue;
                    printf("  view_vox[%d] src=(%.2f,%.2f,%.2f)\n",
                        a, h_views[a].src.x, h_views[a].src.y, h_views[a].src.z);
                }

                SConeProjGeomVec* d_views = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_views, Na * sizeof(SConeProjGeomVec)));
                YK_CUDA_CHECK(cudaMemcpy(d_views, h_views.data(),
                    Na * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));


                auto volTex = Mem::TextureController::createTex3DFromHost(h_vol.data(), g);
                h_vol.clear();
                printf("  texture bound\n");

                const size_t sino_elems = (size_t)Na * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                printf("  launching... Na=%d Nu=%d Nv=%d\n", Na, Nu, Nv);
                {
                    YK::Util::CudaTimer t{ "joseph",stream };
                    fp_joseph_ss_launch(volTex.tex, h_views, d_views, d_sino, g,
                        Na, Nu, Nv, false, stream, FpStepSuperSample::x4, FpDetSuperSample::x2);
                }

                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                printf("  done\n");

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                for (int a = 0; a < Na; ++a) {
                    float maxv = 0.f;
                    for (size_t k = 0; k < (size_t)Nv * Nu; ++k)
                        maxv = fmaxf(maxv, h_sino[a * (size_t)Nv * Nu + k]);
                    if (maxv > 1e-6f || a < 5 || a >= Na - 5) {
                        //printf("  angle %3d: max=%.4f\n", a, maxv);
                    }

                }

                float maxv = *std::max_element(h_sino.begin(), h_sino.end());
                float sumv = 0.f;
                for (auto x : h_sino) sumv += x;
                printf("  sino: max=%.4f  sum=%.3e\n", maxv, sumv);

                saveRaw("rawtest5_real_sino_x4_x2.raw", h_sino.data(), sino_elems);
                printf("  saved: rawtest4_real_sino.raw [%d x %d x %d]\n", Na, Nv, Nu);

                cudaFree(d_views);
                cudaFree(d_sino);
            }

            // ----------------------------------------------------------------
            // �����
            // ----------------------------------------------------------------
            inline void runRawTests()
            {
                cudaStream_t stream;
                YK_CUDA_CHECK(cudaStreamCreate(&stream));

                //test_uniform_single_view(stream);
                //test_uniform_multi_view(stream);
                //test_offset(stream);
                test_real_volume(stream);
                test_real_volume_ss(stream);

                YK_CUDA_CHECK(cudaStreamDestroy(stream));
                printf("\n[RawTest] all done\n");
            }

        } // namespace RawTest
    } // namespace Fp
} // namespace YK


namespace YK {
    namespace Fp {
        namespace CVPTest {


            // ----------------------------------------------------------------
            // ���ߺ���
            // ----------------------------------------------------------------
            static bool saveRaw(const char* path, const float* data, size_t count)
            {
                std::ofstream f(path, std::ios::binary);
                if (!f) { fprintf(stderr, "[saveRaw] cannot open %s\n", path); return false; }
                f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
                return true;
            }

            static bool loadRaw(const char* path, std::vector<float>& buf, size_t count)
            {
                buf.resize(count);
                std::ifstream f(path, std::ios::binary);
                if (!f) { fprintf(stderr, "[loadRaw] cannot open %s\n", path); return false; }
                f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
                return (size_t)f.gcount() == count * sizeof(float);
            }
            // ----------------------------------------------------------------
// CVP����1�������壬���Ƕȣ���֤��������·������
// ----------------------------------------------------------------
            static void test_cvp_uniform_single_view(cudaStream_t stream)
            {
                printf("\n[CVPTest1] uniform volume, single view\n");

                constexpr int   Nx = 64, Ny = 64, Nz = 64;
                constexpr float vox = 0.1f;
                constexpr int   Nu = 64, Nv = 64;
                constexpr float du = 1.f, dv = 1.f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);

                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);
                float* d_vol = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
                YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
                    h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));

                // ���� view��gantry=0��src �� -y���� RawTest1 ��ȫ��ͬ�ļ���
                SConeProjGeomVec view;
                view.src = make_float4(0.f, -SID, 0.f, 0.f);
                view.srcCR = make_float4(0.f, 1.f, 0.f, 0.f);
                view.detU = make_float4(du, 0.f, 0.f, 0.f);
                view.detV = make_float4(0.f, 0.f, dv, 0.f);
                view.detS = make_float4(-Nu * 0.5f * du, SDD - SID, -Nv * 0.5f * dv, 0.f);
                view.angle = make_float4(0.f, 0.f, 0.f, 0.f);

                // CVP ���������꣬����һ��
                std::vector<SConeProjGeomVec> h_views = { view };

                const size_t sino_elems = (size_t)1 * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                fp_cvp_launch(d_vol, d_sino, h_views.data(), g, 1, Nu, Nv, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                // ��������Ӧ �� Ny * vox = 6.4mm���� Joseph/Siddon ���룩
                const float center = h_sino[(Nv / 2) * Nu + Nu / 2];
                const float expect = (float)Ny * vox;
                printf("  center pixel: got=%.4f  expected=%.4f  diff=%.4f  rel=%.4f%%\n",
                    center, expect, fabsf(center - expect),
                    fabsf(center - expect) / expect * 100.f);

                // ��ӡ�����У�����۲� footprint ����
                printf("  center row (v=%d):", Nv / 2);
                for (int u = Nu / 2 - 4; u <= Nu / 2 + 4; ++u)
                    printf(" %.3f", h_sino[(Nv / 2) * Nu + u]);
                printf("\n");

                saveRaw("cvptest1_single_view.raw", h_sino.data(), sino_elems);
                printf("  saved: cvptest1_single_view.raw [%d x %d]\n", Nv, Nu);

                cudaFree(d_vol);
                cudaFree(d_sino);
            }

            // ----------------------------------------------------------------
            // CVP����2�������壬��Ƕȣ���֤���Ƕ���������һ����
            // ----------------------------------------------------------------
            static void test_cvp_uniform_multi_view(cudaStream_t stream)
            {
                printf("\n[CVPTest2] uniform volume, multi view (360 angles)\n");

                constexpr int   Nx = 64, Ny = 64, Nz = 64;
                constexpr float vox = 0.1f;
                constexpr int   Na = 360;
                constexpr int   Nu = 64, Nv = 64;
                constexpr float du = 1.f, dv = 1.f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);
                std::vector<float> h_vol(Nx * Ny * Nz, 1.f);

                float* d_vol = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
                YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
                    h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));

                std::vector<float> angles(Na);
                for (int i = 0; i < Na; ++i)
                    angles[i] = 2.f * CUDA_PI * i / Na;

                // CVP ���������꣬����һ��
                std::vector<SConeProjGeomVec> h_views(Na);
                std::vector<SFDKGeoParamPerView> h_gv(Na);
                build_circular_vec_geometry_from_theta(
                    h_views, angles, Na, Nu, Nv, du, dv,
                    SID, SDD - SID,
                    f3(0.f, 0.f, 0.f),
                    f3(0.f, 0.f, 0.f));

                const size_t sino_elems = (size_t)Na * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                fp_cvp_launch(d_vol, d_sino, h_views.data(), g, Na, Nu, Nv, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                // ��֤���Ƕ���������һ����
                float minVal = 1e9f, maxVal = -1e9f, sumVal = 0.f;
                for (int a = 0; a < Na; ++a) {
                    float v = h_sino[((size_t)a * Nv + Nv / 2) * Nu + Nu / 2];
                    minVal = fminf(minVal, v);
                    maxVal = fmaxf(maxVal, v);
                    sumVal += v;
                }
                const float mean = sumVal / Na;
                printf("  center pixel across %d angles:\n", Na);
                printf("    min=%.4f  max=%.4f  mean=%.4f  range=%.4f  range/mean=%.4f%%\n",
                    minVal, maxVal, mean, maxVal - minVal,
                    (maxVal - minVal) / mean * 100.f);
                printf("  expected mean �� %.4f (Ny * vox)\n", (float)Ny * vox);

                // �� Joseph ���ӽǽ���Աȣ�������ڣ�
                {
                    std::vector<float> h_ref;
                    if (loadRaw("rawtest2_multi_view.raw", h_ref, sino_elems)) {
                        double diff2 = 0.0, ref2 = 0.0;
                        for (size_t k = 0; k < sino_elems; ++k) {
                            double d = h_sino[k] - h_ref[k];
                            diff2 += d * d;
                            ref2 += (double)h_ref[k] * h_ref[k];
                        }
                        printf("  vs Joseph: rel_rms=%.3e\n", sqrt(diff2 / (ref2 + 1e-30)));
                    }
                }

                saveRaw("cvptest2_multi_view.raw", h_sino.data(), sino_elems);
                printf("  saved: cvptest2_multi_view.raw [%d x %d x %d]\n", Na, Nv, Nu);

                cudaFree(d_vol);
                cudaFree(d_sino);
            }




            static void test_cvp_real_volume(cudaStream_t stream)
            {
                printf("\n[CVPTest] real volume: fdk_vec_vol_offline.raw\n");

                constexpr int   Nx = 512, Ny = 512, Nz = 400;
                constexpr float vox_xy = 0.25f, vox_z = 0.25f;
                constexpr int   Na = 360, Nu = 1024, Nv = 1024;
                constexpr float du = 0.25f, dv = 0.25f;
                constexpr float SID = 500.f, SDD = 1000.f;

                SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox_xy, vox_z);
                g.center = make_float3(0.f, 0.f, 0.f);

                // ---------- ������� ----------
                std::vector<float> h_vol;
                if (!loadRaw("fdk_vec_vol_offline.raw", h_vol, (size_t)Nx * Ny * Nz)) return;
                printf("  volume loaded\n");

                float* d_vol = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)));
                YK_CUDA_CHECK(cudaMemcpy(d_vol, h_vol.data(),
                    h_vol.size() * sizeof(float), cudaMemcpyHostToDevice));
                h_vol.clear();

                // ---------- ����ͶӰ���Σ��������꣬����һ����----------
                std::vector<float> angles(Na);
                for (int i = 0; i < Na; ++i)
                    angles[i] = 2.f * CUDA_PI * i / Na;

                std::vector<SConeProjGeomVec> h_views(Na);
                build_circular_vec_geometry_from_theta(
                    h_views, angles, Na, Nu, Nv, du, dv,
                    SID, SDD - SID,
                    f3(0.f, 0.f, 0.f),
                    f3(0.f, 0.f, 0.f));

                // ��ӡ��0֡ȷ�ϼ�����ȷ
                printf("  view[0] src=(%.2f, %.2f, %.2f)\n",
                    h_views[0].src.x, h_views[0].src.y, h_views[0].src.z);
                printf("  view[0] srcCR=(%.4f, %.4f, %.4f)\n",
                    h_views[0].srcCR.x, h_views[0].srcCR.y, h_views[0].srcCR.z);

                // ---------- ��������ͼ ----------
                const size_t sino_elems = (size_t)Na * Nv * Nu;
                float* d_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino, sino_elems * sizeof(float)));
                YK_CUDA_CHECK(cudaMemset(d_sino, 0, sino_elems * sizeof(float)));

                // ---------- ���� CVP ----------
                printf("  launching CVP FP: Na=%d Nu=%d Nv=%d...\n", Na, Nu, Nv);
                {
                    YK::Util::CudaTimer t{ "cvp",stream };
                    fp_cvp_launch(d_vol, d_sino, h_views.data(), g, Na, Nu, Nv, stream);
                }

                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                printf("  done\n");

                // ---------- �ض� + ͳ�� ----------
                std::vector<float> h_sino(sino_elems);
                YK_CUDA_CHECK(cudaMemcpy(h_sino.data(), d_sino,
                    sino_elems * sizeof(float), cudaMemcpyDeviceToHost));

                for (int a = 0; a < Na; ++a) {
                    float maxv = 0.f;
                    for (size_t k = 0; k < (size_t)Nv * Nu; ++k)
                        maxv = fmaxf(maxv, h_sino[a * (size_t)Nv * Nu + k]);
                    if (maxv > 1e-6f || a < 5 || a >= Na - 5) {
                        //printf("  angle %3d: max=%.4f\n", a, maxv);
                    }

                }

                float maxv = *std::max_element(h_sino.begin(), h_sino.end());
                float sumv = 0.f;
                for (auto x : h_sino) sumv += x;
                printf("  sino: max=%.4f  sum=%.3e\n", maxv, sumv);

                // ---------- �� Joseph ����Աȣ�������ڣ�----------
                {
                    std::vector<float> h_ref;
                    if (loadRaw("rawtest4_real_sino.raw", h_ref, sino_elems)) {
                        double diff2 = 0.0, ref2 = 0.0;
                        for (size_t k = 0; k < sino_elems; ++k) {
                            double d = h_sino[k] - h_ref[k];
                            diff2 += d * d;
                            ref2 += (double)h_ref[k] * h_ref[k];
                        }
                        printf("  vs Joseph: rel_rms=%.3e\n", sqrt(diff2 / (ref2 + 1e-30)));
                    }
                    else {
                        printf("  (Joseph reference not found, skip diff)\n");
                    }
                }

                saveRaw("cvp_test_real_sino.raw", h_sino.data(), sino_elems);
                printf("  saved: cvp_test_real_sino.raw [%d x %d x %d]\n", Na, Nv, Nu);

                cudaFree(d_vol);
                cudaFree(d_sino);
            }

            inline void runCVPTests()
            {
                cudaStream_t stream;
                YK_CUDA_CHECK(cudaStreamCreate(&stream));
                //test_cvp_uniform_single_view(stream);
                //test_cvp_uniform_multi_view(stream);
                test_cvp_real_volume(stream);

                YK_CUDA_CHECK(cudaStreamDestroy(stream));
                printf("\n[CVPTest] all done\n");
            }
        };
    };
};


//
int main_fp() {
    YK::Fp::SiddonTest::runSiddonTests();
    YK::Fp::RawTest::runRawTests();
    YK::Fp::CVPTest::runCVPTests();

    return 0;
}