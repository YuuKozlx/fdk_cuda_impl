#include <cstdio>
#include <cstring>

#include "global/YkLog.h"

// These are manual integration tests.  They may require local raw data, so no
// test is run implicitly.  Invoke one explicit group from the command line.
int main_fdk();
int main_fdk_custom_filter();
int main_bp_verify();
int main_siddon_ray_adjoint_verify();
int main_fp();
int main_ossart_test();
int main_cgls_test();
int main_iter_sirt_recon_sim();

namespace {
void printUsage(const char* executable)
{
    std::printf("Usage: %s <group>\n", executable);
    std::printf("  fdk       standard FDK integration test\n");
    std::printf("  filter    custom FDK filter test\n");
    std::printf("  fp        forward-projection test\n");
    std::printf("  bp        back-projection verification\n");
    std::printf("  adjoint   Siddon adjointness verification\n");
    std::printf("  sirt | ossart | cgls   iterative integration tests\n");
}
}

int main(int argc, char** argv)
{
    YK::Logger::instance().set_level(YK::LogLevel::Debug);
    if (argc != 2) { printUsage(argv[0]); return 0; }

    const char* group = argv[1];
    if (std::strcmp(group, "fdk") == 0) return main_fdk();
    if (std::strcmp(group, "filter") == 0) return main_fdk_custom_filter();
    if (std::strcmp(group, "fp") == 0) return main_fp();
    if (std::strcmp(group, "bp") == 0) return main_bp_verify();
    if (std::strcmp(group, "adjoint") == 0) return main_siddon_ray_adjoint_verify();
    if (std::strcmp(group, "sirt") == 0) return main_iter_sirt_recon_sim();
    if (std::strcmp(group, "ossart") == 0) return main_ossart_test();
    if (std::strcmp(group, "cgls") == 0) return main_cgls_test();

    std::fprintf(stderr, "Unknown manual test group: %s\n", group);
    printUsage(argv[0]);
    return 2;
}
