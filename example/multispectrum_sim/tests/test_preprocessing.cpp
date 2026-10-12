#include "ReconstructionPreprocessor.hpp"
#include "XcomAttenuationProvider.hpp"
#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace yk::spectral;

void testPreprocessState(const std::filesystem::path&);

int main()
{
    const auto root = std::filesystem::temp_directory_path() /
        "multispectrum-preprocessing-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    try {
        SimulationConfig config;
        config.geometry.parameters.views = 2;
        config.geometry.parameters.detector_u = 3;
        config.geometry.parameters.detector_v = 2;
        config.reconstruction.preprocessing.enabled = true;
        config.reconstruction.preprocessing.beam_hardening.enabled = true;
        config.reconstruction.preprocessing.beam_hardening.order = 2;
        config.reconstruction.preprocessing.beam_hardening.coefficients = {0.0, 1.0, 1.0};
        config.reconstruction.preprocessing.beam_hardening.clamp_min = 0.0;
        config.reconstruction.preprocessing.beam_hardening.clamp_max = 20.0;

        const auto input_path = root / "input.raw";
        const auto output_path = root / "output.raw";
        std::vector<float> input = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f,
                                    3.0f, 3.5f, 4.0f, 0.25f, 0.75f, 1.25f};
        {
            std::ofstream file(input_path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(input.data()),
                       static_cast<std::streamsize>(input.size() * sizeof(float)));
        }

        std::string diagnostic;
        if (!prepareReconstructionProjection(config, input_path, output_path, diagnostic))
            throw std::runtime_error(diagnostic);

        std::vector<float> output(input.size());
        std::ifstream file(output_path, std::ios::binary);
        file.read(reinterpret_cast<char*>(output.data()),
                  static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!file) throw std::runtime_error("could not read preprocessing output");
        for (size_t i = 0; i < input.size(); ++i) {
            const float expected = input[i] + input[i] * input[i];
            if (std::abs(output[i] - expected) > 1.0e-6f)
                throw std::runtime_error("beam hardening polynomial result mismatch");
        }
        file.close();
        testPreprocessState(root);
        // Verify the fitted EID water curve with two separated energy bins.
        // Photon-count weighting produces a different attenuation curve.
        const auto spectrum_path = root / "eid-spectrum.csv";
        {
            std::ofstream spectrum(spectrum_path);
            spectrum << "Energy[keV]  N[keV cm^2 mAs]^-1 @ 1 meter\n40 1\n80 1\n";
        }
        config.projection.spectrum_file = spectrum_path;
        auto& bhc = config.reconstruction.preprocessing.beam_hardening;
        bhc.fit_from_water_plate = true;
        bhc.order = 3;
        bhc.water_plate_max_mm = 400;
        bhc.water_plate_samples = 81;
        bhc.reference_energy_keV = 0; // Photon mean of the equal bins is 60 keV.
        MaterialSpec water;
        water.label = 1; water.name = "water"; water.formula = "H2O"; water.density_g_cm3 = 1;
        XcomAttenuationProvider attenuation({});
        std::vector<double> mass;
        std::string error;
        if (!attenuation.query(water, {40, 80, 60}, mass, error))
            throw std::runtime_error(error);
        std::vector<float> expected(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            const double thickness_cm = 40.0 * i / (input.size() - 1);
            const double transmission = (40 * std::exp(-mass[0] * thickness_cm)
                + 80 * std::exp(-mass[1] * thickness_cm)) / 120;
            input[i] = static_cast<float>(-std::log(transmission));
            expected[i] = static_cast<float>(mass[2] * thickness_cm);
        }
        {
            std::ofstream signal(input_path, std::ios::binary);
            signal.write(reinterpret_cast<const char*>(input.data()), input.size() * sizeof(float));
        }
        if (!prepareReconstructionProjection(config, input_path, output_path, diagnostic))
            throw std::runtime_error(diagnostic);
        std::ifstream fitted(output_path, std::ios::binary);
        fitted.read(reinterpret_cast<char*>(output.data()), output.size() * sizeof(float));
        if (!fitted) throw std::runtime_error("could not read EID fit output");
        for (size_t i = 0; i < output.size(); ++i)
            if (std::abs(output[i] - expected[i]) > 0.01f)
                throw std::runtime_error("EID water plate fit mismatch at sample " + std::to_string(i) +
                    ": actual=" + std::to_string(output[i]) + " expected=" + std::to_string(expected[i]));
        fitted.close();
        // Generated and exported/imported spectra must produce identical BHC fits.
        config.projection.spectrum_file.clear();
        config.projection.spectrum.source = "generated";
        config.projection.spectrum.filters = {{"Al", 2.0}};
        config.projection.spectrum.export_file = root / "generated-bhc.spec";
        if (!prepareReconstructionProjection(config, input_path, output_path, diagnostic))
            throw std::runtime_error("generated BHC: " + diagnostic);
        std::vector<float> generated_output(input.size());
        {
            std::ifstream generated_file(output_path, std::ios::binary);
            generated_file.read(reinterpret_cast<char*>(generated_output.data()), generated_output.size()*sizeof(float));
            if (!generated_file) throw std::runtime_error("cannot read generated BHC output");
        }
        config.projection.spectrum_file = config.projection.spectrum.export_file;
        config.projection.spectrum.export_file.clear();
        config.projection.spectrum.source = "file";
        if (!prepareReconstructionProjection(config, input_path, output_path, diagnostic))
            throw std::runtime_error("imported BHC: " + diagnostic);
        {
            std::ifstream imported_file(output_path, std::ios::binary);
            imported_file.read(reinterpret_cast<char*>(output.data()), output.size()*sizeof(float));
            if (!imported_file) throw std::runtime_error("cannot read imported BHC output");
        }
        for (size_t i=0; i<output.size(); ++i)
            if (std::abs(output[i]-generated_output[i]) > 1e-6f)
                throw std::runtime_error("generated/imported water-plate fit differs");
        config.projection.spectrum_file = spectrum_path;
        // Fixed guidance must estimate scatter from the smoothed copy without
        // replacing the original projection, including across multiple frames.
        std::vector<double> radii(256), density(256,1.0);
        for (size_t i=0;i<radii.size();++i) radii[i]=i*.1;
        nlohmann::json zero_model = {
            {"format","fasks-runtime-model/v4"},{"schema_version",4},
            {"signal_domain","energy_deposition_per_pixel"},
            {"validity",{{"transmission_min",.1},{"transmission_max",.9}}},
            {"kernel_sampling",{{"representation","discrete_radial_table"},{"support_u_mm",1.0},{"support_v_mm",1.0}}},
            {"center_kernel",{
                {"amplitude",{{"interpolation","linear-log-attenuation"},{"log_attenuation_nodes",{.1,3.0}},{"values",{0,0}}}},
                {"shape",{{"family","discrete-radial-kernel"},{"radial_positions_mm",radii},{"radial_density_nodes_mm2",{density,density}}}}}},
            {"water_equivalent_thickness",{{"interpolation","pchip"},{"log_attenuation_nodes",{0,3.0}},{"thickness_mm_nodes",{0,300}}}},
            {"fasks",{{"gamma_per_mm",0},{"wet_smooth_sigma_mm",0}}},
            {"edge_amplitude_window",{{"model","paper-linear-distance-to-boundary"},{"boundary_scale",1},{"ramp_mm",80}}},
            {"detector_response",{{"model","air-pencil-beam-scatter-per-primary"},{"scatter_kernel_per_primary",{{0}}}}}
        };
        const auto model_path=root/"zero-scatter.json", air_path=root/"air.raw";
        { std::ofstream f(model_path); f<<zero_model.dump(); }
        { std::ofstream f(air_path,std::ios::binary); std::vector<float> flat(6,100);f.write(reinterpret_cast<const char*>(flat.data()),flat.size()*sizeof(float)); }
        bhc.enabled=false;
        auto& scatter_config=config.reconstruction.preprocessing.scatter;
        scatter_config.enabled=true;scatter_config.model_file=model_path;scatter_config.air_projection_file=air_path;
        scatter_config.solver.type="fixed";scatter_config.correction.iterations=3;
        for(bool guided : {false,true}) {
            scatter_config.guide.enabled=guided;
            if(!prepareReconstructionProjection(config,input_path,output_path,diagnostic))throw std::runtime_error(diagnostic);
            std::ifstream f(output_path,std::ios::binary);f.read(reinterpret_cast<char*>(output.data()),output.size()*sizeof(float));
            if(!f)throw std::runtime_error("could not read fixed guide test output");
            for(size_t p=0;p<input.size();++p)if(std::abs(input[p]-output[p])>2e-6f)
                throw std::runtime_error("fixed guide replaced raw projection with smoothed data");
        }
        std::filesystem::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}
