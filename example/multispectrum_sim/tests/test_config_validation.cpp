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
        std::string diagnostic;
        try { yk::spectral::loadConfig(path); accepted = true; }
        catch (const std::runtime_error& e) { diagnostic = e.what(); }
        if (accepted != valid) throw std::runtime_error(
            "unexpected config validation result: " + reconstruction +
            (diagnostic.empty() ? std::string{} : "\nreason: " + diagnostic));
    };
    try {
        test("type='analytic'\nenabled=true\n[reconstruction.analytic]\npipeline='wfbp'\nfilter='ramlak'\n", true);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='wfbp'\nfilter='hann'\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='fdk'\nfilter='ramlka'\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='fdk'\nfilter=123\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='wfbp'\nwfbp_cutoff='1.0'\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='wfbp'\nwfbp_cutoff=nan\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='wfbp'\nwfbp_apodization=1.1\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='fdk'\nwfbp_cutoff=1.0\n", false);
        test("type='analytic'\nenabled=true\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='unknown'\n", false);
        test("type='iterative'\nenabled=true\n[reconstruction.iterative]\nalgorithm='tigre_sart'\niterations=5\n", true);
        test("type='iterative'\nenabled=true\n[reconstruction.iterative]\nalgorithm='tigre_sirt'\niterations=5\n", true);
        test("type='iterative'\nenabled=true\n[reconstruction.iterative]\nalgorithm='tigre_os_sart'\nsubsets=8\n", true);
        test("type='iterative'\nenabled=true\n[reconstruction.iterative]\nalgorithm='tigre_sart_tv'\niterations=5\nrelaxation=0.8\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='tigre_sart_tv'\ntv_alpha=0\n", false);

        test("type='iterative'\nenabled=true\n[reconstruction.iterative]\nalgorithm='tigre_os_sart_tv'\nsubsets=8\nback_projector='joseph_v3'\n", true);
        test("type='iterative'\nenabled=true\n[reconstruction.iterative]\nalgorithm='sirt'\niterations=5\n", true);
        test("type='iterative'\nenabled=true\n[reconstruction.iterative]\nalgorithm='ossart'\nsubsets=8\n", true);
        test("type='iterative'\nenabled=true\n[reconstruction.iterative]\nalgorithm='cgls'\niterations=5\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\niterations=0\n", false);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\nrelaxation=0\n", false);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\nforward_projector='bad'\n", false);
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
