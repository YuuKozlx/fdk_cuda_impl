#include "config/SimConfig.hpp"
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    const auto path = std::filesystem::current_path() / "config-validation-test.toml";
    auto test = [&](const std::string& reconstruction, bool valid) {
        {
            std::ofstream f(path);
            f << "schema_version=2\n[workflow]\nmode='reconstruct'\n"
                 "[geometry]\nkind='cyl_helical'\n"
                 "[reconstruction]\ninput_projection_file='input.raw'\n" << reconstruction;
        }
        bool accepted = false;
        std::string diagnostic;
        try { yk::spectral::loadConfig(path); accepted = true; }
        catch (const std::exception& e) { diagnostic = e.what(); }
        if (accepted != valid) throw std::runtime_error(
            "unexpected config validation result: " + reconstruction +
            (diagnostic.empty() ? std::string{} : "\nreason: " + diagnostic));
    };
    auto testV2 = [&](const std::string& content, bool valid) {
        {
            std::ofstream f(path);
            f << "schema_version=2\n" << content;
        }
        bool accepted = false;
        std::string diagnostic;
        try { yk::spectral::loadConfig(path); accepted = true; }
        catch (const std::runtime_error& e) { diagnostic = e.what(); }
        if (accepted != valid) throw std::runtime_error(
            "unexpected v2 config validation result: " + content +
            (diagnostic.empty() ? std::string{} : "\nreason: " + diagnostic));
    };
    try {
        {
            const auto spectrum_path = std::filesystem::current_path() /
                "spectrum-normalization-test.csv";
            {
                std::ofstream spectrum(spectrum_path);
                spectrum << "Energy[keV]  N[keV cm^2 mAs]^-1 @ 1 meter\n20 2\n40 3\n";
            }
            const auto spectrum = yk::spectral::loadSpectrum(spectrum_path);
            std::filesystem::remove(spectrum_path);
            if (spectrum.size() != 2 ||
                std::abs(spectrum[0].fluence_per_keV_cm2_mAs_at_1m - 2.0) > 1e-12 ||
                std::abs(spectrum[1].fluence_per_keV_cm2_mAs_at_1m - 3.0) > 1e-12 ||
                std::abs(spectrum[0].bin_width_keV - 20.0) > 1e-12)
                throw std::runtime_error("absolute spectrum was not parsed correctly");
        }
        test("type='analytic'\n[reconstruction.analytic]\npipeline='wfbp'\nfilter='ramlak'\n", true);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='wfbp'\nfilter='hann'\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='fdk'\nfilter='ramlka'\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='fdk'\nfilter=123\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='wfbp'\nwfbp_cutoff='1.0'\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='wfbp'\nwfbp_cutoff=nan\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='wfbp'\nwfbp_apodization=1.1\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='fdk'\nwfbp_cutoff=1.0\n", false);
        test("type='analytic'\n", false);
        test("type='analytic'\n[reconstruction.analytic]\npipeline='unknown'\n", false);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='tigre_sart'\niterations=5\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='tigre_sirt'\niterations=5\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='tigre_os_sart'\nsubsets=8\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='tigre_sart'\nsubsets=8\n", false);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='tigre_sart_tv'\niterations=5\nrelaxation=0.8\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='tigre_sart_tv'\ntv_alpha=0\n", false);

        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='tigre_os_sart_tv'\nsubsets=8\nprojection_model='joseph'\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\niterations=5\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='ossart'\nsubsets=8\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\nsubsets=8\n", false);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='cgls'\niterations=5\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\niterations=0\n", false);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\nrelaxation=0\n", false);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\nforward_projector='bad'\n", false);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\nprojection_model='siddon'\n", true);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\nprojection_model='bad'\n", false);
        test("type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\nprojection_model='joseph'\nback_projector='joseph_v3'\n", false);
        testV2(
            "[workflow]\nmode='reconstruct'\n"
            "[geometry]\nkind='flat_cbct'\nviews=1\nviews_per_turn=1\n"
            "detector_u=1\ndetector_v=1\nvolume_x=1\nvolume_y=1\nvolume_z=1\n"
            "[reconstruction]\ntype='analytic'\ninput_projection_file='input.raw'\n"
            "[reconstruction.analytic]\npipeline='fdk'\n", true);
        testV2(
            "[simulation]\ngeometry='flat_cbct'\n", false);
        testV2(
            "[workflow]\nmode='reconstruct'\n[geometry]\nkind='flat_cbct'\n"
            "[reconstruction]\nenabled = true\ninput_projection_file='input.raw'\n"
            "type='analytic'\n[reconstruction.analytic]\npipeline='fdk'\n", false);
        testV2(
            "[workflow]\nmode='project'\n[geometry]\nkind='flat_cbct'\n"
            "[projection]\noutput_file='projection.raw'\n", false);
        testV2(
            "[workflow]\nmode='project_and_reconstruct'\n"
            "[geometry]\nkind='flat_cbct'\nviews=1\nviews_per_turn=1\n"
            "detector_u=1\ndetector_v=1\nvolume_x=1\nvolume_y=1\nvolume_z=1\n"
            "[projection]\noutput_file='projection.raw'\nengine='pixel_local_random'\n"
            "[projection.sampling]\nsamples_per_pixel=1\n"
            "[[projection.materials]]\nlabel=1\nname='water'\nformula='H2O'\n"
            "[reconstruction]\ntype='analytic'\n"
            "[reconstruction.analytic]\npipeline='fdk'\n", true);
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}

