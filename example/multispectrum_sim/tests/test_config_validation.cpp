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
        const std::string display_base="[workflow]\nmode='reconstruct'\n[geometry]\nkind='flat_cbct'\n"
            "[reconstruction]\ninput_projection_file='input.raw'\n[reconstruction.analytic]\npipeline='fdk'\n[visualization]\n";
        testV2(display_base+"enabled=false\ntexture_max_size=512\nmax_fps=15\nprojection='signal'\n",true);
        testV2(display_base+"enabled=true\n",false);
        testV2(display_base+"texture_max_size=0\n",false);
        testV2(display_base+"max_fps=nan\n",false);
        testV2(display_base+"max_fps=0\n",false);
        testV2(display_base+"projection='unknown'\n",false);
        const std::string preprocess_base = "type='iterative'\n[reconstruction.iterative]\nalgorithm='sirt'\n";
        test(preprocess_base + "[reconstruction.preprocessing]\nenabled=true\n"
             "[reconstruction.preprocessing.beam_hardening]\nenabled=true\norder=2\ncoefficients=[0,1,0.1]\n", true);
        test(preprocess_base + "[reconstruction.preprocessing]\nenabled=true\n"
             "[reconstruction.preprocessing.beam_hardening]\nenabled=true\norder=3\ncoefficients=[0,1]\n", false);
        test(preprocess_base + "[reconstruction.preprocessing]\nenabled=true\n"
             "[reconstruction.preprocessing.scatter]\nenabled=true\n", false);
        const std::string kernel_base = preprocess_base +
            "[reconstruction.preprocessing]\nenabled=true\n"
            "[reconstruction.preprocessing.beam_hardening]\nenabled=true\norder=1\ncoefficients=[0,1]\n";
        test(kernel_base + "[reconstruction.preprocessing.scatter.kernel]\nmode='all'\n", true);
        test(kernel_base + "[reconstruction.preprocessing.scatter.kernel]\nmode='segmented'\nsegments=3\n", true);
        test(kernel_base + "[reconstruction.preprocessing.scatter.kernel]\nmode='unknown'\n", false);
        test(kernel_base + "[reconstruction.preprocessing.scatter.kernel]\nmode='segmented'\nsegments=1\n", false);
        test(kernel_base + "[reconstruction.preprocessing.scatter.kernel]\nmode='all'\nsegments=3\n", false);
        test(kernel_base + "[reconstruction.preprocessing.scatter.solver]\ntype='fixed'\n", true);
        const std::string adaptive_base = kernel_base +
            "[reconstruction.preprocessing.scatter.solver]\ntype='adaptive_experimental'\n"
            "[reconstruction.preprocessing.scatter.solver.adaptive]\n";
        test(adaptive_base + "iterations=10\ninitial_damping=0.7\nmin_damping=0.1\n", true);
        test(adaptive_base + "min_damping=0.9\n", false);
        test(adaptive_base + "damping_decay=0.9\n[reconstruction.preprocessing.scatter.guide]\nenabled=true\nsigma_u_pixels=2\nsigma_v_pixels=2\n", true);
        test(adaptive_base + "damping_decay=0\n", false);
        test(adaptive_base + "damping_decay=1.1\n", false);
        test(adaptive_base + "[reconstruction.preprocessing.scatter.guide]\nenabled=true\nsigma_u_pixels=0\n", false);
        test(adaptive_base + "[reconstruction.preprocessing.scatter.solver.adaptive.guide]\nenabled=true\n", false);
        test(kernel_base + "[reconstruction.preprocessing.scatter.solver]\ntype='fixed'\n"
             "[reconstruction.preprocessing.scatter.guide]\nenabled=true\nsigma_u_pixels=2\nsigma_v_pixels=2\n", true);
        test(kernel_base + "[reconstruction.preprocessing.scatter.guide]\nenabled=true\nsigma_v_pixels=33\n", false);
        test(adaptive_base + "iterations=0\n", false);
        test(adaptive_base + "relative_tolerance=0.001\n", false);
        test(adaptive_base + "max_iterations=10\n", false);
        test(adaptive_base + "[reconstruction.preprocessing.scatter.solver.adaptive.transmission_guard]\n"
             "low_quantile=0.01\nrisk_threshold=0.02\npositivity_margin=0.9\n", true);
        test(adaptive_base + "[reconstruction.preprocessing.scatter.solver.adaptive.transmission_guard]\n"
             "risk_threshold=0\n", false);
        test(adaptive_base + "[reconstruction.preprocessing.scatter.solver.adaptive.transmission_guard]\n"
             "positivity_margin=1\n", false);
        test(adaptive_base + "[reconstruction.preprocessing.scatter.correction]\ndamping=0.6\n", false);
        test(kernel_base + "[reconstruction.preprocessing.scatter.solver]\ntype='fixed'\n"
             "[reconstruction.preprocessing.scatter.solver.adaptive]\niterations=10\n", false);
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
            "[projection.geometry_flux]\nenabled=true\n"
            "[projection.quantum_noise]\nenabled=true\nseed=67890\n"
            "[projection.sampling]\nsamples_per_pixel=1\n"
            "[[projection.materials]]\nlabel=1\nname='water'\nformula='H2O'\n"
            "[reconstruction]\ntype='analytic'\n"
            "[reconstruction.analytic]\npipeline='fdk'\n", true);
        testV2(
            "[workflow]\nmode='project'\n"
            "[geometry]\nkind='flat_cbct'\nviews=1\nviews_per_turn=1\n"
            "detector_u=1\ndetector_v=1\nvolume_x=1\nvolume_y=1\nvolume_z=1\n"
            "[projection]\noutput_file='projection.raw'\n"
            "[projection.geometry_flux]\nenabled=true\n"
            "[projection.quantum_noise]\nenabled=false\nseed=67890\n"
            "[projection.focal_spot]\nenabled=true\n"
            "normal_size_u_mm=0.4\nnormal_size_v_mm=0.4\n"
            "defocus_size_u_mm=3.0\ndefocus_size_v_mm=2.0\nepsilon=0.01\n"
            "[projection.sampling]\nsamples_per_pixel=8\n"
            "[[projection.materials]]\nlabel=1\nname='water'\nformula='H2O'\n", true);
        testV2(
            "[workflow]\nmode='project'\n"
            "[geometry]\nkind='flat_cbct'\n"
            "[projection]\noutput_file='projection.raw'\n"
            "[projection.geometry_flux]\nenabled=true\n"
            "[projection.quantum_noise]\nenabled=true\n"
            "[projection.focal_spot]\nenabled=true\n"
            "normal_size_u_mm=0.4\nnormal_size_v_mm=0.4\n"
            "defocus_size_u_mm=3.0\ndefocus_size_v_mm=2.0\nepsilon=1.01\n"
            "[projection.sampling]\nsamples_per_pixel=8\n"
            "[[projection.materials]]\nlabel=1\nname='water'\nformula='H2O'\n", false);
        const std::string focal_base=
            "[workflow]\nmode='project'\n[geometry]\nkind='flat_cbct'\nviews=1\nviews_per_turn=1\n"
            "detector_u=1\ndetector_v=1\nvolume_x=1\nvolume_y=1\nvolume_z=1\n"
            "[projection]\noutput_file='projection.raw'\n"
            "[projection.geometry_flux]\nenabled=true\n[projection.quantum_noise]\nenabled=false\nseed=67890\n[projection.sampling]\nsamples_per_pixel=8\n"
            "[[projection.materials]]\nlabel=1\nname='water'\nformula='H2O'\n"
            "[projection.focal_spot]\nenabled=true\n";
        testV2(focal_base+"normal_size_u_mm=0.4\nnormal_size_v_mm=0.4\ndefocus_size_u_mm=0.2\ndefocus_size_v_mm=0.8\nepsilon=0.1\n",false);
        testV2(focal_base+"normal_size_u_mm=0.4\nnormal_size_v_mm=0.4\ndefocus_size_u_mm=0.4\ndefocus_size_v_mm=0.4\nepsilon=0.25\n",true);
        testV2(focal_base+"normal_size_u_mm=0.4\nnormal_size_v_mm=0.4\ndefocus_size_u_mm=0.4\ndefocus_size_v_mm=0.8\nepsilon=0.25\n",true);
        testV2(focal_base+"normal_size_u_mm=0.0\nnormal_size_v_mm=0.0\ndefocus_size_u_mm=1.0\ndefocus_size_v_mm=1.0\nepsilon=1.0\n",true);
        const auto independent_base=focal_base.substr(0,focal_base.find("[projection.focal_spot]"))+"[projection.focal_spot]\n";
        testV2(independent_base+"enabled=true\ndefocus_enabled=false\nnormal_size_u_mm=0.4\nnormal_size_v_mm=0.4\ndefocus_size_u_mm=0.2\ndefocus_size_v_mm=0.2\nepsilon=1.0\n",true);
        testV2(independent_base+"enabled=false\ndefocus_enabled=true\nnormal_size_u_mm=0.4\nnormal_size_v_mm=0.4\ndefocus_size_u_mm=0.2\ndefocus_size_v_mm=0.2\nepsilon=0.25\n",true);
        testV2(independent_base+"enabled=false\ndefocus_enabled=false\nnormal_size_u_mm=0.4\nnormal_size_v_mm=0.4\ndefocus_size_u_mm=0.2\ndefocus_size_v_mm=0.2\nepsilon=0.25\n",true);
        testV2(focal_base+"normal_size_u_mm=0.4\nnormal_size_v_mm=0.4\ndefocus_size_u_mm=1.0\ndefocus_size_v_mm=1.0\nepsilon=0.25\n",true);
        if(!yk::spectral::loadConfig(path).projection.focal_spot.defocus_enabled)throw std::runtime_error("legacy enabled focus lost defocus");
        testV2(independent_base+"enabled=false\nnormal_size_u_mm=0.4\nnormal_size_v_mm=0.4\ndefocus_size_u_mm=1.0\ndefocus_size_v_mm=1.0\nepsilon=0.25\n",true);
        if(yk::spectral::loadConfig(path).projection.focal_spot.defocus_enabled)throw std::runtime_error("legacy disabled focus unexpectedly enabled defocus");
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}

