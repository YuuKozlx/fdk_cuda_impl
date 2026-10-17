#pragma once
#include "config/SimConfig.hpp"
#include "pipeline/Stage.hpp"
#include <array>
#include <map>
namespace yk::spectral::examples {
struct ScanPreset {
    std::string name;
    GeometryKind kind;
    int views, views_per_turn, detector_u, detector_v;
    double sid_mm, sdd_mm, pixel_u_mm, pixel_v_mm, pitch_mm_per_turn, start_z_mm;
    std::string reconstruction;
};
struct Parameter {
    std::string id, name, group;
    double value, minimum, maximum;
    bool integer = false;
};
struct ExampleRequest {
    std::filesystem::path output_directory; // Calculation output only; generate may read packaged input data.
    int views = 0;
    std::map<std::string, double> parameters;
};
enum class Shape { Sphere,
    CylinderZ };
struct Landmark {
    Shape shape = Shape::Sphere;
    std::array<double, 3> center_mm {}; // Phantom-local coordinates, before configured pose.
    double radius_mm = 0, half_height_mm = 0;
    std::uint8_t label = 100;
};
struct ExampleResult {
    SimulationConfig config;
    std::vector<std::uint8_t> labels;
    std::vector<Landmark> landmarks;
    double support_radius_mm = 0, support_half_height_mm = 0;
};
// A setup stage returns typed in-memory resources, never a TOML/RAW round trip.
class IExampleStage : public pipeline::IPlugin {
public:
    virtual ScanPreset scanPreset() const = 0;
    virtual std::vector<Parameter> parameters() const = 0;
    virtual ExampleResult generate(const ExampleRequest&) const = 0;
};
using ExampleRegistry = pipeline::FactoryRegistry<IExampleStage>;
ExampleRegistry builtinExamples();
}
