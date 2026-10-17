#include "ProjectionUiConfig.hpp"
#include "examples/IcrpExamples.hpp"
#include <iostream>
using namespace yk::spectral;
int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--list") {
            for (const auto& p : examples::icrpPhantoms())
                std::cout << p.id() << "\t" << p.rawName() << '\n';
            return 0;
        }
        if (argc < 3) {
            std::cerr << "Usage: multispectrum_icrp_examples <ICRP-root> <new-output-directory> [example-id] [--results directory] [--materials directory] [--configs directory]\n       multispectrum_icrp_examples --list\n";
            return 2;
        }
        const auto source = std::filesystem::absolute(std::filesystem::u8path(argv[1])), destination = std::filesystem::absolute(std::filesystem::u8path(argv[2]));
        std::string selection;
        std::filesystem::path results, materials_directory = MULTISPECTRUM_ICRP_MATERIAL_DIRECTORY, configs = destination;
        for (int i = 3; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--results" || option == "--materials" || option == "--configs") {
                if (i + 1 >= argc)
                    throw std::runtime_error("Missing directory after " + option);
                const auto value = std::filesystem::absolute(std::filesystem::u8path(argv[++i]));
                if (option == "--results")
                    results = value;
                else if (option == "--materials")
                    materials_directory = value;
                else
                    configs = value;
            } else if (option.rfind("--", 0) == 0)
                throw std::runtime_error("Unknown option: " + option);
            else if (selection.empty())
                selection = argv[i];
            else
                throw std::runtime_error("Unexpected argument");
        }
        std::vector<const examples::IcrpPhantom*> selected;
        for (const auto& p : examples::icrpPhantoms())
            if (selection.empty() || p.id() == selection)
                selected.push_back(&p);
        if (selected.empty())
            throw std::runtime_error("Unknown ICRP example ID");
        // Validate every source before writing a pack; no download or modification of the source tree.
        for (const auto* p : selected) {
            const auto materials = materials_directory / (std::string("materials-") + p->sex + ".toml");
            examples::loadIcrpMaterialMap(materials);
            const auto raw = source / "phantom" / p->rawName();
            const auto count = size_t(p->dimensions[0]) * p->dimensions[1] * p->dimensions[2];
            if (!std::filesystem::exists(raw) || std::filesystem::file_size(raw) != count)
                throw std::runtime_error("Missing or invalid ICRP uint8 RAW: " + raw.string() + "; prepare source MAT/map.xlsx with the original ICRP workflow first");
            if (std::filesystem::exists(destination / "inputs" / (p->id() + ".raw")) || std::filesystem::exists(configs / (p->id() + ".toml")))
                throw std::runtime_error("Destination example exists; choose a new output directory");
        }
        std::filesystem::create_directories(configs);
        for (const auto* p : selected) {
            const auto mapping = examples::loadIcrpMaterialMap(materials_directory / (std::string("materials-") + p->sex + ".toml"));
            auto config = examples::makeIcrpConfig(*p, mapping, destination, results);
            examples::remapIcrpVolume(source / "phantom" / p->rawName(), config.projection.label_volume, size_t(p->dimensions[0]) * p->dimensions[1] * p->dimensions[2], mapping);
            // Relative paths make the examples portable with the repository/data directory.
            auto relative = [&](std::filesystem::path& path) {if(!path.empty())path=path.lexically_relative(configs); };
            relative(config.projection.label_volume);
            relative(config.projection.output_file);
            relative(config.projection.energy_output_file);
            relative(config.projection.air_output_file);
            relative(config.projection.spectrum.export_file);
            relative(config.reconstruction.output_volume_file);
            relative(config.reconstruction.input_projection_file);
            saveProjectionUiConfig(config, configs / (p->id() + ".toml"), WorkflowMode::ProjectAndReconstruct);
            std::cout << p->id() << " materials=" << mapping.materials.size() << " config=" << (configs / (p->id() + ".toml")).string() << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
