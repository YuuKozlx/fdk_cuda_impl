#include <cstdio>
#include <cstring>
#include <string>

#include <cuda_runtime.h>
#include "CLI11/CLI11.hpp"
#include "global/YkLog.h"

// All manual integration entry points live in the existing module files.
// Keep registration here so adding a test never requires editing main().
int main_fp();
int main_operator_roundtrip_smoke();
int main_external_geometry_operator_smoke();
int main_fdk_batch_consistency_smoke();
int main_catphan_phantom_smoke();
int main_filter_spatial_ramp_validation();
int main_filter_discrete_ramlak_dc_zero();
int main_fp_siddon_uniform_center_length();
int main_fp_siddon_single_voxel_peak();
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

YK::LogLevel parseLogLevel(const std::string& value)
{
    if (value == "trace") return YK::LogLevel::Trace;
    if (value == "debug") return YK::LogLevel::Debug;
    if (value == "info") return YK::LogLevel::Info;
    if (value == "warn") return YK::LogLevel::Warn;
    if (value == "error") return YK::LogLevel::Error;
    if (value == "critical") return YK::LogLevel::Critical;
    return YK::LogLevel::Off;
}

const TestEntry kTests[] = {
    // Framework and public data-flow contracts.
    TEST_INT("framework/operator-roundtrip", "framework", false, main_operator_roundtrip_smoke),
    TEST_INT("framework/external-geometry", "framework", false, main_external_geometry_operator_smoke),

    // FDK, filter and phantom numerical regressions.
    TEST_INT("fdk/batch-consistency", "fdk", false, main_fdk_batch_consistency_smoke),
    TEST_INT("phantom/catphan-like", "phantom", false, main_catphan_phantom_smoke),
    TEST_INT("filter/spatial-ramp", "filter", false, main_filter_spatial_ramp_validation),
    TEST_INT("filter/discrete-ramlak-dc", "filter", false, main_filter_discrete_ramlak_dc_zero),

    // Forward/back projector properties.
    TEST_INT("fp/siddon-uniform-center", "fp", false, main_fp_siddon_uniform_center_length),
    TEST_INT("fp/siddon-single-voxel", "fp", false, main_fp_siddon_single_voxel_peak),
    TEST_INT("operator/matrix-circular", "operator", false, main_operator_matrix_smoke),
    TEST_INT("geometry/planar-fp-bp", "geometry", false, main_planar_geometry_operator_smoke),

    // Synthetic reconstruction smoke tests.
    TEST_INT("recon/sart", "recon", false, main_sart_smoke),
    TEST_INT("recon/sirt", "recon", false, main_sirt_smoke),
    TEST_INT("recon/ossart-tigre", "recon", false, main_ossart_tigre_smoke),
    TEST_INT("recon/ossart", "recon", false, main_ossart_smoke),
    TEST_INT("recon/ossart-ex", "recon", false, main_ossart_ex_smoke),
    TEST_INT("recon/cgls", "recon", false, main_cgls_smoke),
    TEST_INT("recon/cgls-astra", "recon", false, main_cgls_astra_smoke),
    TEST_INT("recon/cgls-ex", "recon", false, main_cgls_ex_smoke),

    // Helical and cylindrical detector algorithms.
    TEST_INT("helical/icd", "helical", false, main_helical_icd_smoke),
    TEST_INT("helical/wfbp", "helical", false, main_helical_wfbp_smoke),
    TEST_INT("helical/wfbp-ffs", "helical", false, main_helical_wfbp_ffs_smoke),
    TEST_INT("helical/wfbp-compare", "helical", false, main_helical_wfbp_comparison),
    TEST_INT("fpcyl/adjoint", "fpcyl", false, main_fpcyl_adjoint),
    TEST_INT("fpcyl/wfbp-compare", "fpcyl", false, main_fpcyl_wfbp_comparison),
    // Legacy diagnostic suites.  These consume files from a developer-local
    // data directory and are excluded from category/all-local runs.
    TEST_INT("fp/legacy-realdata", "fp", true, main_fp),

    // Iterative algorithm diagnostics.
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

int runSelection(const char* selection)
{
    const bool all_local = std::strcmp(selection, "all-local") == 0;
    int selected = 0;
    int failed = 0;
    for (const auto& test : kTests) {
        const bool category_match = std::strcmp(selection, test.category) == 0;
        if ((!all_local && !category_match) || test.needs_real_data) continue;
        ++selected;
        std::printf("\n=== RUN  %s ===\n", test.name);
        const int status = test.run();
        failed += status != 0;
        std::printf("=== %s %s ===\n", status == 0 ? "PASS" : "FAIL", test.name);
    }
    if (selected == 0) return -1;
    std::printf("\nSummary: selected=%d passed=%d failed=%d\n",
        selected, selected - failed, failed);
    return failed == 0 ? 0 : 1;
}
}

int main(int argc, char** argv)
{
    CLI::App app{"YKCBCT CUDA、算法集成和数值回归测试"};
    std::string selection = "list";
    std::string log_level = "debug";
    bool show_list = false;
    app.add_option("selection", selection, "测试名、测试分类、all-local 或 list");
    app.add_flag("-l,--list", show_list, "列出全部测试");
    app.add_option("--log-level", log_level, "日志等级")
        ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}));
    app.footer(
        "示例:\n"
        "  ykcbct_manual_tests filter/discrete-ramlak-dc\n"
        "  ykcbct_manual_tests filter\n"
        "  ykcbct_manual_tests all-local\n"
        "真实数据测试只能通过完整测试名显式运行。");
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    YK::Logger::instance().set_level(parseLogLevel(log_level));
    if (show_list || selection == "list") {
        listTests();
        return 0;
    }
    for (const auto& test : kTests)
        if (selection == test.name)
            return test.run();
    if (const int status = runSelection(selection.c_str()); status >= 0)
        return status;
    std::fprintf(stderr, "Unknown test or category: %s\n", selection.c_str());
    listTests();
    return 2;
}
