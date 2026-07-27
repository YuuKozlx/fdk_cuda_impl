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
int main_algebraic_ex_smoke();
int main_algebraic_smoke();
int main_iterative_convergence_smoke();
int main_ossart_tv_smoke();
int main_tigre_gradient_family_smoke();
int main_cgls_smoke(); int main_cgls_astra_smoke(); int main_cgls_ex_smoke();
int main_cgls_unified_smoke();
int main_large_water_pwls();
int main_large_water_ossart();
int main_large_water_fdk();
int main_large_water_fdk_iterative();
void configure_large_water_fdk_iterative_test(const std::string& method,
    int iterations, int subsets, float relaxation,
    float relative_residual_tolerance, int minimum_iterations,
    int convergence_check_interval, int convergence_patience);
int main_large_arrow_tigre();
void configure_large_arrow_test(const std::string& method, int iterations,
    int block_size, float lambda, int tv_iterations);
#if YKCBCT_TEST_HAS_HELICAL
int main_helical_icd_smoke();
int main_helical_wfbp_smoke();
int main_helical_wfbp_ffs_smoke();
int main_helical_wfbp_comparison();
int main_helical_large_volume();
int main_helical_large_volume_icd();
int main_fpcyl_adjoint();
int main_fpcyl_wfbp_comparison();
#endif

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
    TEST_INT("recon/algebraic-ex", "recon", false, main_algebraic_ex_smoke),
    TEST_INT("recon/algebraic", "recon", false, main_algebraic_smoke),
    TEST_INT("recon/convergence", "recon", false, main_iterative_convergence_smoke),
    TEST_INT("recon/ossart-tv", "recon", false, main_ossart_tv_smoke),
    TEST_INT("recon/tigre-gradient-family", "recon", false,
        main_tigre_gradient_family_smoke),
    TEST_INT("recon/cgls", "recon", false, main_cgls_smoke),
    TEST_INT("recon/cgls-astra", "recon", false, main_cgls_astra_smoke),
    TEST_INT("recon/cgls-ex", "recon", false, main_cgls_ex_smoke),
    TEST_INT("recon/cgls-unified", "recon", false, main_cgls_unified_smoke),

    // 大体积性能与显存测试，单独分类避免混入常规重建冒烟项。
    TEST_INT("large/water-pwls", "large", false, main_large_water_pwls),
    TEST_INT("large/water-ossart", "large", false, main_large_water_ossart),
    TEST_INT("large/water-fdk", "large", false, main_large_water_fdk),
    TEST_INT("large/water-fdk-iterative", "large", false,
        main_large_water_fdk_iterative),
    TEST_INT("large/arrow-tigre", "large", false, main_large_arrow_tigre),

    // Helical and cylindrical detector algorithms.
#if YKCBCT_TEST_HAS_HELICAL
    TEST_INT("helical/icd", "helical", false, main_helical_icd_smoke),
    TEST_INT("helical/wfbp", "helical", false, main_helical_wfbp_smoke),
    TEST_INT("helical/wfbp-ffs", "helical", false, main_helical_wfbp_ffs_smoke),
    TEST_INT("helical/wfbp-compare", "helical", false, main_helical_wfbp_comparison),
    TEST_INT("helical/large-volume", "helical", false, main_helical_large_volume),
    TEST_INT("helical/large-volume-icd", "helical", false, main_helical_large_volume_icd),
    TEST_INT("fpcyl/adjoint", "fpcyl", false, main_fpcyl_adjoint),
    TEST_INT("fpcyl/wfbp-compare", "fpcyl", false, main_fpcyl_wfbp_comparison),
#endif
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
    std::string arrow_method = "os-asd-pocs";
    int arrow_iterations = 6;
    int arrow_block_size = 20;
    float arrow_lambda = 0.25f;
    int arrow_tv_iterations = 5;
    std::string water_iterative_method = "ossart";
    int water_iterative_iterations = 10;
    int water_iterative_subsets = 10;
    float water_iterative_relaxation = 0.25f;
    float water_iterative_relative_residual = 0.f;
    int water_iterative_minimum_iterations = 1;
    int water_iterative_check_interval = 1;
    int water_iterative_patience = 1;
    bool show_list = false;
    app.add_option("selection", selection, "测试名、测试分类、all-local 或 list");
    app.add_flag("-l,--list", show_list, "列出全部测试");
    app.add_option("--log-level", log_level, "日志等级")
        ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}));
    app.add_option("--arrow-method", arrow_method, "大箭头模体重建方法")
        ->check(CLI::IsMember({
            "sart", "os-sart", "sirt",
            "asd-pocs", "os-asd-pocs", "b-asd-pocs-beta",
            "pcsd", "os-pcsd", "aw-pcsd", "os-aw-pcsd",
            "aw-asd-pocs", "os-aw-asd-pocs"}));
    app.add_option("--arrow-iterations", arrow_iterations,
        "大箭头模体外循环次数")->check(CLI::PositiveNumber);
    app.add_option("--arrow-block-size", arrow_block_size,
        "OS 方法每个子集的视角数")->check(CLI::PositiveNumber);
    app.add_option("--arrow-lambda", arrow_lambda,
        "大箭头模体数据更新初始步长")->check(CLI::PositiveNumber);
    app.add_option("--arrow-tv-iterations", arrow_tv_iterations,
        "POCS 方法每轮 TV 内迭代次数")->check(CLI::PositiveNumber);
    app.add_option("--water-iterative-method", water_iterative_method,
        "水模 FDK 初值后的迭代方法")
        ->check(CLI::IsMember({"ossart", "cgls"}));
    app.add_option("--water-iterative-iterations", water_iterative_iterations,
        "水模 FDK 初值后的迭代次数")->check(CLI::PositiveNumber);
    app.add_option("--water-iterative-subsets", water_iterative_subsets,
        "水模 OSSART 子集数量")->check(CLI::PositiveNumber);
    app.add_option("--water-iterative-relaxation", water_iterative_relaxation,
        "水模 OSSART 松弛因子")->check(CLI::PositiveNumber);
    app.add_option("--water-iterative-relative-residual",
        water_iterative_relative_residual,
        "水模相对投影残差阈值；0 表示关闭提前停止")
        ->check(CLI::NonNegativeNumber);
    app.add_option("--water-iterative-minimum-iterations",
        water_iterative_minimum_iterations, "水模收敛前最少外循环数")
        ->check(CLI::NonNegativeNumber);
    app.add_option("--water-iterative-check-interval",
        water_iterative_check_interval, "水模每隔几轮检查收敛")
        ->check(CLI::PositiveNumber);
    app.add_option("--water-iterative-patience", water_iterative_patience,
        "水模连续满足次数")->check(CLI::PositiveNumber);
    app.footer(
        "示例:\n"
        "  ykcbct_manual_tests filter/discrete-ramlak-dc\n"
        "  ykcbct_manual_tests filter\n"
        "  ykcbct_manual_tests all-local\n"
        "  ykcbct_manual_tests large/arrow-tigre "
        "--arrow-method os-sart --arrow-iterations 20\n"
        "真实数据测试只能通过完整测试名显式运行。");
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    YK::Logger::instance().set_level(parseLogLevel(log_level));
    configure_large_arrow_test(arrow_method, arrow_iterations,
        arrow_block_size, arrow_lambda, arrow_tv_iterations);
    configure_large_water_fdk_iterative_test(water_iterative_method,
        water_iterative_iterations, water_iterative_subsets,
        water_iterative_relaxation, water_iterative_relative_residual,
        water_iterative_minimum_iterations, water_iterative_check_interval,
        water_iterative_patience);
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
