#include "pipeline/BuiltinPluginInfo.hpp"
#include "pipeline/Stage.hpp"
#include <iostream>
#include <set>
using namespace yk::spectral::pipeline;
void require(bool b, const char* s)
{
    if (!b)
        throw std::runtime_error(s);
}
template <class F>
void reject(F f)
{
    bool failed = false;
    try {
        f();
    } catch (const std::exception&) {
        failed = true;
    }
    require(failed, "invalid plugin metadata accepted");
}
class Mock final : public IPlugin {
    PluginInfo info_;

public:
    explicit Mock(PluginInfo info)
        : info_(std::move(info))
    {
    }
    const PluginInfo& info() const noexcept override { return info_; }
};
int main()
{
    try {
        std::set<std::string> ids;
        for (const auto& info : builtinPluginDescriptions()) {
            validatePluginInfo(info);
            require(ids.insert(info.id).second, "duplicate built-in ID");
            require(findBuiltinPluginInfo(info.id) == &info, "catalog lookup mismatch");
        }
        require(ids.size() == 53, "missing built-in descriptors");
        require(builtinPluginInfo(BuiltinPlugin::MultiSpecFp).id=="projection.source.multi_spec_fp", "multi-spectrum source ID mismatch");
        require(!findBuiltinPluginInfo("unknown"), "unknown catalog ID found");
        const auto& bhc = builtinPluginInfo(BuiltinPlugin::Bhc);
        require(bhc.id == "preprocess.bhc" && bhc.parameters.size() == 3 && bhc.category == "bhc", "BHC self description incomplete");
        require(builtinPluginInfo(BuiltinPlugin::CpuOptical).id != builtinPluginInfo(BuiltinPlugin::CudaOptical).id, "CPU/CUDA implementations share ID");
        require(builtinPluginInfo(BuiltinPlugin::ExampleBhc).implementation_status == ImplementationStatus::Experimental, "example status incorrect");
        FactoryRegistry<Mock> registry;
        int constructions = 0;
        registry.add(bhc, [&] {++constructions;return std::make_unique<Mock>(bhc); });
        require(registry.find(bhc.id) && *registry.find(bhc.id) == bhc, "registry description changed");
        auto listed = registry.list();
        require(listed.size() == 1 && listed[0] == bhc, "registry list mismatch");
        listed[0].name = "changed snapshot";
        require(registry.find(bhc.id)->name == bhc.name, "query snapshot changed stored metadata");
        require(!registry.find("unknown") && constructions == 0, "description queries constructed a plugin");
        auto instance = registry.create(bhc.id);
        require(constructions == 1 && instance->info() == bhc, "factory instance metadata changed");
        reject([&] { registry.add(bhc, [] { return std::unique_ptr<Mock> {}; }); });
        auto invalid = bhc;
        invalid.id.clear();
        reject([&] { registry.add(invalid, [] { return std::unique_ptr<Mock> {}; }); });
        invalid = bhc;
        invalid.name.clear();
        reject([&] { validatePluginInfo(invalid); });
        invalid = bhc;
        invalid.kind = PluginKind::Unspecified;
        reject([&] { validatePluginInfo(invalid); });
        invalid = bhc;
        invalid.parameters.push_back(invalid.parameters.front());
        reject([&] { validatePluginInfo(invalid); });
        invalid = bhc;
        invalid.kind = static_cast<PluginKind>(99);
        reject([&] { validatePluginInfo(invalid); });
        auto optional = bhc;
        optional.id = "optional";
        optional.description.clear();
        optional.version.clear();
        optional.category.clear();
        optional.documentation.clear();
        optional.parameters.clear();
        validatePluginInfo(optional);
        auto null_info = optional;
        null_info.id = "null";
        registry.add(null_info, [] { return std::unique_ptr<Mock> {}; });
        reject([&] { registry.create("null"); });
        auto mismatch = optional;
        mismatch.id = "mismatch";
        registry.add(mismatch, [&] {++constructions;return std::make_unique<Mock>(bhc); });
        reject([&] { registry.create("mismatch"); });
        reject([&] { registry.create("unknown"); });
        auto empty = optional;
        empty.id = "empty";
        reject([&] { registry.add(empty, {}); });
        std::cout << "Static descriptors, parameter schema, zero-construction queries and factory consistency passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
