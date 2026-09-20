#include "SimConfig.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
    const auto path = std::filesystem::current_path() / "config-validation-test.toml";
    auto test = [&](const std::string& reconstruction, bool valid) {
        {
            std::ofstream f(path);
            f << "[simulation]\ngeometry='cyl_helical'\nuse_library_fp=true\n"
                 "[[materials]]\nlabel=1\nname='water'\nformula='H2O'\n"
                 "[reconstruction]\n" << reconstruction;
        }
        bool accepted = false;
        try { yk::spectral::loadConfig(path); accepted = true; }
        catch (const std::runtime_error&) {}
        if (accepted != valid) throw std::runtime_error("unexpected config validation result: " + reconstruction);
    };
    try {
        test("pipeline='wfbp'\nfilter='ramlak'\nenabled=true\n", true);
        test("pipeline='wfbp'\nfilter='hann'\n", false);
        test("pipeline='fdk'\nfilter='ramlka'\n", false);
        test("pipeline='fdk'\nfilter=123\n", false);
        test("pipeline='wfbp'\nwfbp_cutoff='1.0'\n", false);
        test("pipeline='wfbp'\nwfbp_cutoff=nan\n", false);
        test("pipeline='wfbp'\nwfbp_apodization=1.1\n", false);
        test("pipeline='fdk'\nwfbp_cutoff=1.0\n", false);
        test("pipeline='fdk'\nenabled=true\n", false);
        test("pipeline='unknown'\n", false);
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
