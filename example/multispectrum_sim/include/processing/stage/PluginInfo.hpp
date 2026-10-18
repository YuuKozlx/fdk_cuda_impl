#pragma once
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace yk::spectral::pipeline {
enum class PluginKind { Unspecified,
    ProjectionStage,
    PreprocessStage,
    PostprocessStage,
    Exporter,
    ProjectionSource, ExampleStage, CalibrationStage };
enum class ImplementationStatus { Stable,
    Experimental,
    Deprecated };
enum class ParameterType { Boolean,
    Integer,
    Real,
    String,
    RealArray,
    Choice };
struct ParameterInfo {
    std::string key, name, description;
    ParameterType type = ParameterType::Real;
    std::string unit;
    bool operator==(const ParameterInfo& p) const { return std::tie(key, name, description, type, unit) == std::tie(p.key, p.name, p.description, p.type, p.unit); }
};
struct PluginInfo {
    std::string id, name, description, version;
    PluginKind kind = PluginKind::Unspecified;
    std::string category;
    std::vector<std::string> tags;
    std::string vendor, documentation;
    ImplementationStatus implementation_status = ImplementationStatus::Stable;
    std::string deprecated_message;
    std::vector<ParameterInfo> parameters;
    bool operator==(const PluginInfo& p) const { return std::tie(id, name, description, version, kind, category, tags, vendor, documentation, implementation_status, deprecated_message, parameters) == std::tie(p.id, p.name, p.description, p.version, p.kind, p.category, p.tags, p.vendor, p.documentation, p.implementation_status, p.deprecated_message, p.parameters); }
};
inline void validatePluginInfo(const PluginInfo& info)
{
    if (info.id.empty() || info.name.empty() || info.kind == PluginKind::Unspecified)
        throw std::invalid_argument("plugin ID, name and kind are required");
    switch (info.kind) {
    case PluginKind::ProjectionStage:
    case PluginKind::PreprocessStage:
    case PluginKind::PostprocessStage:
    case PluginKind::Exporter:
    case PluginKind::ProjectionSource:
    case PluginKind::ExampleStage:
    case PluginKind::CalibrationStage:
        break;
    default:
        throw std::invalid_argument("invalid plugin kind");
    }
    switch (info.implementation_status) {
    case ImplementationStatus::Stable:
    case ImplementationStatus::Experimental:
    case ImplementationStatus::Deprecated:
        break;
    default:
        throw std::invalid_argument("invalid implementation status");
    }
    std::set<std::string> keys;
    for (const auto& p : info.parameters) {
        if (p.key.empty() || p.name.empty() || !keys.insert(p.key).second)
            throw std::invalid_argument("invalid or duplicate parameter description");
        switch (p.type) {
        case ParameterType::Boolean:
        case ParameterType::Integer:
        case ParameterType::Real:
        case ParameterType::String:
        case ParameterType::RealArray:
        case ParameterType::Choice:
            break;
        default:
            throw std::invalid_argument("invalid parameter type");
        }
    }
}
class IPlugin {
public:
    virtual ~IPlugin() = default;
    // Immutable metadata independent of prepare/process/reset and device resources.
    virtual const PluginInfo& info() const noexcept = 0;
};
}
