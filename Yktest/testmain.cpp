#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>
#include "global/YkLog.h"

// All manual integration entry points live in the existing module files.
// Keep registration here so adding a test never requires editing main().
int main_fdk_custom_filter(); int main_fdk_realdata(); int main_fdk();
int main_mcgpu_watercylinder_fdk(); int main_fdk_v2();
int main_fdk_zslab_bigdata(); int main_fdk_zslab();
void test_stdrecon_with_random_offset(cudaStream_t); void test_recon_with_random_offset(cudaStream_t);
void test_stdrecon_with_periodic_offset(cudaStream_t); void test_recon_with_periodic_offset(cudaStream_t);
void test_fdk_cylinder();

int main_bp_runner(); int main_bp_realdata_runner(); int main_fdkbp_vs_onlybp_verify();
int main_bp_verify(); int main_bp_zslab_verify(); int main_siddon_bp_runner();
int main_siddon_ray_adjoint_verify(); int main_siddon_voxel_adjoint_verify();
int main_joseph_adjoint_verify(); int main_joseph_v2_v3_adjoint_verify();
int main_siddon_bp_verify(); int main_siddon_zslab_verify(); int main_fp_bp_geometry_verify();

void test_fp_runner(cudaStream_t); void test_fp_runner_siddon_vs_joseph(cudaStream_t);
void test_fp_cylinder_siddon_joseph(cudaStream_t); void test_fp_runner_fixed_offset(cudaStream_t);
void test_fp_runner_random_offset(cudaStream_t); void test_fp_runner_periodic_offset(cudaStream_t);
void test_periodic_fp_ideal_recon(cudaStream_t); void test_periodic_fp_corrected_recon(cudaStream_t);
void test_random_fp_ideal_recon(cudaStream_t); void test_fixed_fp_ideal_recon(cudaStream_t);
void test_fixed_fp_corrected_recon(cudaStream_t); void test_insufficient_angle_fp_recon(cudaStream_t);
void test_flat_detector_roty_fp(cudaStream_t); void test_flat_detector_roty_fp_ellipse(cudaStream_t);
void test_generate_pcb_phantom(); int main_fp();

int main_ossart_test(); int main_ossart_ex_test(); int main_iter_recon_sim();
int main_iter_sirt_recon_sim(); int main_cgls_test(); int main_ossart_realdata_test();
int main_ossart_mcgpu_cylinder_test(); int main_cgls_realdata_test();
void test_flat_detector_roty_fp_ossart(cudaStream_t); void test_flat_detector_roty_fp_independent(cudaStream_t);

#ifdef YKCBCT_MANUAL_HELICAL
int main_helical_from_volume(); int main_helical_from_volume_cylinder();
int main_helical_online_from_volume();
int main_helical_from_volume_cylinder_ossart_independent();
int main_helical_from_volume_cylinder_ossart();
int main_helical_from_volume_cylinder_cgls();
#endif

namespace {
using TestFn = int (*)();

int runWithStream(void (*fn)(cudaStream_t)) {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    fn(stream);
    const cudaError_t status = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return status == cudaSuccess ? 0 : 1;
}
int runVoid(void (*fn)()) { fn(); return 0; }

#define TEST_INT(name, category, real_data, fn) {name, category, real_data, fn}
#define TEST_STREAM(name, category, real_data, fn) {name, category, real_data, []() { return runWithStream(fn); }}
#define TEST_VOID(name, category, real_data, fn) {name, category, real_data, []() { return runVoid(fn); }}

struct TestEntry { const char* name; const char* category; bool needs_real_data; TestFn run; };

const TestEntry kTests[] = {
    TEST_INT("fdk/basic", "fdk", false, main_fdk),
    TEST_INT("fdk/custom-filter", "fdk", false, main_fdk_custom_filter),
    TEST_INT("fdk/realdata", "fdk", true, main_fdk_realdata),
    TEST_INT("fdk/mcgpu-cylinder", "fdk", true, main_mcgpu_watercylinder_fdk),
    TEST_INT("fdk/v2", "fdk", false, main_fdk_v2),
    TEST_INT("fdk/zslab", "fdk", false, main_fdk_zslab),
    TEST_INT("fdk/zslab-bigdata", "fdk", true, main_fdk_zslab_bigdata),
    TEST_STREAM("fdk/random-offset-standard", "fdk", false, test_stdrecon_with_random_offset),
    TEST_STREAM("fdk/random-offset-corrected", "fdk", false, test_recon_with_random_offset),
    TEST_STREAM("fdk/periodic-offset-standard", "fdk", false, test_stdrecon_with_periodic_offset),
    TEST_STREAM("fdk/periodic-offset-corrected", "fdk", false, test_recon_with_periodic_offset),
    TEST_VOID("fdk/cylinder", "fdk", false, test_fdk_cylinder),

    TEST_INT("bp/runner", "bp", false, main_bp_runner),
    TEST_INT("bp/realdata-runner", "bp", true, main_bp_realdata_runner),
    TEST_INT("bp/fdk-vs-only", "bp", false, main_fdkbp_vs_onlybp_verify),
    TEST_INT("bp/verify", "bp", false, main_bp_verify),
    TEST_INT("bp/zslab", "bp", false, main_bp_zslab_verify),
    TEST_INT("bp/siddon-runner", "bp", false, main_siddon_bp_runner),
    TEST_INT("bp/adjoint-siddon-ray", "bp", false, main_siddon_ray_adjoint_verify),
    TEST_INT("bp/adjoint-siddon-voxel", "bp", false, main_siddon_voxel_adjoint_verify),
    TEST_INT("bp/adjoint-joseph", "bp", false, main_joseph_adjoint_verify),
    TEST_INT("bp/adjoint-joseph-v2-v3", "bp", false, main_joseph_v2_v3_adjoint_verify),
    TEST_INT("bp/siddon-verify", "bp", false, main_siddon_bp_verify),
    TEST_INT("bp/siddon-zslab", "bp", false, main_siddon_zslab_verify),
    TEST_INT("bp/fp-geometry", "bp", false, main_fp_bp_geometry_verify),

    TEST_INT("fp/basic", "fp", false, main_fp),
    TEST_STREAM("fp/runner", "fp", true, test_fp_runner),
    TEST_STREAM("fp/siddon-vs-joseph", "fp", true, test_fp_runner_siddon_vs_joseph),
    TEST_STREAM("fp/cylinder-siddon-vs-joseph", "fp", false, test_fp_cylinder_siddon_joseph),
    TEST_STREAM("fp/fixed-offset", "fp", true, test_fp_runner_fixed_offset),
    TEST_STREAM("fp/random-offset", "fp", true, test_fp_runner_random_offset),
    TEST_STREAM("fp/periodic-offset", "fp", true, test_fp_runner_periodic_offset),
    TEST_STREAM("fp/periodic-ideal-recon", "fp", true, test_periodic_fp_ideal_recon),
    TEST_STREAM("fp/periodic-corrected-recon", "fp", true, test_periodic_fp_corrected_recon),
    TEST_STREAM("fp/random-ideal-recon", "fp", true, test_random_fp_ideal_recon),
    TEST_STREAM("fp/fixed-ideal-recon", "fp", true, test_fixed_fp_ideal_recon),
    TEST_STREAM("fp/fixed-corrected-recon", "fp", true, test_fixed_fp_corrected_recon),
    TEST_STREAM("fp/insufficient-angle", "fp", true, test_insufficient_angle_fp_recon),
    TEST_STREAM("fp/flat-detector-roty", "fp", false, test_flat_detector_roty_fp),
    TEST_STREAM("fp/flat-detector-ellipse", "fp", false, test_flat_detector_roty_fp_ellipse),
    TEST_VOID("fp/generate-pcb-phantom", "fp", false, test_generate_pcb_phantom),

    TEST_INT("iter/ossart", "iter", false, main_ossart_test),
    TEST_INT("iter/ossart-ex", "iter", false, main_ossart_ex_test),
    TEST_INT("iter/sim", "iter", false, main_iter_recon_sim),
    TEST_INT("iter/sirt", "iter", false, main_iter_sirt_recon_sim),
    TEST_INT("iter/cgls", "iter", false, main_cgls_test),
    TEST_INT("iter/ossart-realdata", "iter", true, main_ossart_realdata_test),
    TEST_INT("iter/ossart-mcgpu-cylinder", "iter", true, main_ossart_mcgpu_cylinder_test),
    TEST_INT("iter/cgls-realdata", "iter", true, main_cgls_realdata_test),
    TEST_STREAM("iter/flat-detector-ossart", "iter", false, test_flat_detector_roty_fp_ossart),
    TEST_STREAM("iter/flat-detector-independent", "iter", false, test_flat_detector_roty_fp_independent),
#ifdef YKCBCT_MANUAL_HELICAL
    TEST_INT("helical/from-volume", "helical", true, main_helical_from_volume),
    TEST_INT("helical/cylinder", "helical", false, main_helical_from_volume_cylinder),
    TEST_INT("helical/online", "helical", true, main_helical_online_from_volume),
    TEST_INT("helical/ossart-independent", "helical", false, main_helical_from_volume_cylinder_ossart_independent),
    TEST_INT("helical/ossart", "helical", false, main_helical_from_volume_cylinder_ossart),
    TEST_INT("helical/cgls", "helical", false, main_helical_from_volume_cylinder_cgls),
#endif
};

void listTests() {
    for (const auto& test : kTests)
        std::printf("%-38s %-6s %s\n", test.name, test.category,
            test.needs_real_data ? "real-data" : "synthetic/local");
}
}

int main(int argc, char** argv)
{
    YK::Logger::instance().set_level(YK::LogLevel::Debug);
    if (argc != 2 || std::strcmp(argv[1], "list") == 0) {
        std::printf("Usage: %s list | <test-name>\n", argv[0]);
        listTests();
        return argc == 2 ? 0 : 2;
    }
    for (const auto& test : kTests)
        if (std::strcmp(argv[1], test.name) == 0)
            return test.run();
    std::fprintf(stderr, "Unknown manual test: %s\n", argv[1]);
    listTests();
    return 2;
}
