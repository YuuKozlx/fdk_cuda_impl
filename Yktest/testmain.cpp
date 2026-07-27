#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>
#include "global/YkLog.h"

// All manual integration entry points live in the existing module files.
// Keep registration here so adding a test never requires editing main().
int main_fp();
int main_operator_roundtrip_smoke();
int main_external_geometry_operator_smoke();
int main_fdk_batch_consistency_smoke();
int main_catphan_phantom_smoke();
int main_filter_spatial_ramp_validation();
int main_sart_smoke(); int main_sirt_smoke();
int main_operator_matrix_smoke(); int main_planar_geometry_operator_smoke();
int main_ossart_tigre_smoke(); int main_ossart_smoke(); int main_ossart_ex_smoke();
int main_cgls_smoke(); int main_cgls_astra_smoke(); int main_cgls_ex_smoke();
int main_helical_icd_smoke();
int main_helical_wfbp_smoke();
int main_helical_wfbp_ffs_smoke();
int main_helical_wfbp_comparison();
int main_fpcyl_adjoint();
int main_fpcyl_wfbp_comparison();

int main_ossart_test(); int main_ossart_ex_test(); int main_iter_recon_sim();
int main_iter_sirt_recon_sim(); int main_cgls_test(); int main_ossart_realdata_test();
int main_ossart_mcgpu_cylinder_test(); int main_cgls_realdata_test();
void test_flat_detector_roty_fp_ossart(cudaStream_t); void test_flat_detector_roty_fp_independent(cudaStream_t);


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
    TEST_INT("framework/operator-roundtrip", "framework", false, main_operator_roundtrip_smoke),
    TEST_INT("framework/external-geometry", "framework", false, main_external_geometry_operator_smoke),
    TEST_INT("fdk/batch-consistency", "fdk", false, main_fdk_batch_consistency_smoke),
    TEST_INT("phantom/catphan-like", "phantom", false, main_catphan_phantom_smoke),
    TEST_INT("filter/spatial-ramp", "filter", false, main_filter_spatial_ramp_validation),
    TEST_INT("recon/sart", "recon", false, main_sart_smoke),
    TEST_INT("recon/sirt", "recon", false, main_sirt_smoke),
    TEST_INT("recon/ossart-tigre", "recon", false, main_ossart_tigre_smoke),
    TEST_INT("recon/ossart", "recon", false, main_ossart_smoke),
    TEST_INT("recon/ossart-ex", "recon", false, main_ossart_ex_smoke),
    TEST_INT("recon/cgls", "recon", false, main_cgls_smoke),
    TEST_INT("recon/cgls-astra", "recon", false, main_cgls_astra_smoke),
    TEST_INT("recon/cgls-ex", "recon", false, main_cgls_ex_smoke),
    TEST_INT("operator/matrix-circular", "operator", false, main_operator_matrix_smoke),
    TEST_INT("geometry/planar-fp-bp", "geometry", false, main_planar_geometry_operator_smoke),
    TEST_INT("helical/icd", "helical", false, main_helical_icd_smoke),
    TEST_INT("helical/wfbp", "helical", false, main_helical_wfbp_smoke),
    TEST_INT("helical/wfbp-ffs", "helical", false, main_helical_wfbp_ffs_smoke),
    TEST_INT("helical/wfbp-compare", "helical", false, main_helical_wfbp_comparison),
    TEST_INT("fpcyl/adjoint", "fpcyl", false, main_fpcyl_adjoint),
    TEST_INT("fpcyl/wfbp-compare", "fpcyl", false, main_fpcyl_wfbp_comparison),
    TEST_INT("fp/basic", "fp", false, main_fp),

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
