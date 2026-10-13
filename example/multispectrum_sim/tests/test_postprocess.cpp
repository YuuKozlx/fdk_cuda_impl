#include "pipeline/BuiltinPluginInfo.hpp"
#include "recon/postprocess/PostprocessPipeline.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace yk::spectral::postprocess;
void require(bool b, const char* message)
{
    if (!b)
        throw std::runtime_error(message);
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
    require(failed, "invalid chain was accepted");
}
struct AddStage : IStage {
    Location where;
    int& resets;
    bool fail = false;
    AddStage(Location l, int& r)
        : where(l)
        , resets(r)
    {
    }
    const yk::spectral::pipeline::PluginInfo& info()const noexcept override{static const yk::spectral::pipeline::PluginInfo p{"test.postprocessstage","test add","Test-only plugin","",yk::spectral::pipeline::PluginKind::PostprocessStage};return p;}
    yk::spectral::pipeline::Contract<Domain> contract()const override{return {Domain::AttenuationMmInv,Domain::AttenuationMmInv,where,where};}
    void process(Volume& v, const Context&) override
    {
        if (fail)
            throw std::runtime_error("test failure");
        auto* p = where == Location::Host ? v.host_data : v.device_data;
        for (size_t i = 0; i < v.elements; ++i)
            p[i] += 1;
    }
    void reset() noexcept override { ++resets; }
};
struct Transfer : IDataTransfer {
    std::vector<float> gpu;
    int uploads = 0, downloads = 0;
    void ensure(Volume& v, Location l, const Context&) override
    {
        if (l == Location::Device) {
            gpu.assign(v.host_data, v.host_data + v.elements);
            v.device_data = gpu.data();
            ++uploads;
        } else {
            std::copy(gpu.begin(), gpu.end(), v.host_data);
            ++downloads;
        }
        v.location = l;
    }
};
int main(int argc, char** argv)
{
    try {
        require(argc == 2, "output directory required");
        Context c;
        c.volume_file = std::filesystem::path(argv[1]) / "volume.raw";
        c.slice_prefix = std::filesystem::path(argv[1]) / "slices/volume";
        std::vector<float> values(24);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = float(i);
        const auto original = values;
        Volume v;
        v.grid.dimensions = { 2, 3, 4 };
        v.grid.spacing_mm = { .5, 1, 2 };
        v.grid.origin_mm = { 12, 7, -10 };
        v.host_data = values.data();
        v.elements = values.size();
        int last = -1;
        c.progress = [&](yk::spectral::TaskProgress p) {require(p.completed>last,"invalid progress");last=p.completed; };
        Pipeline defaults;
        defaults.appendExporter(makeRawFloatExporter());
        defaults.appendExporter(makeOrthogonalBmpExporter());
        defaults.run(v, c);
        require(values == original && v.grid.origin_mm[0] == 12, "default outputs changed volume");
        require(last == 2 && defaults.timings().size() == 2, "timing/progress missing");
        std::vector<float> read(values.size());
        std::ifstream f(c.volume_file, std::ios::binary);
        f.read(reinterpret_cast<char*>(read.data()), read.size() * sizeof(float));
        require(read == original, "RAW float values changed");
        f.close();
        require(std::filesystem::file_size(c.slice_prefix.string() + "-axial.bmp") == 1078 + 4 * 3, "axial BMP layout changed");
        require(std::filesystem::file_size(c.slice_prefix.string() + "-coronal.bmp") == 1078 + 4 * 4, "coronal BMP layout changed");
        require(std::filesystem::file_size(c.slice_prefix.string() + "-sagittal.bmp") == 1078 + 4 * 4, "sagittal BMP layout changed");
        c.progress = {};
        StageRegistry registry;ExportRegistry outputs;
        registry.add(yk::spectral::pipeline::builtinPluginInfo(yk::spectral::pipeline::BuiltinPlugin::HuConversion), [] { return makeHuConversionStage(.02); });
        outputs.add(yk::spectral::pipeline::builtinPluginInfo(yk::spectral::pipeline::BuiltinPlugin::RawOutput), [] { return makeRawFloatExporter(); });
        reject([&] { registry.create("not-installed"); });
        reject([&] { registry.add(yk::spectral::pipeline::builtinPluginInfo(yk::spectral::pipeline::BuiltinPlugin::HuConversion), [] { return makeHuConversionStage(.02); }); });
        // Invalid final domain is caught before in-place HU conversion or RAW overwrite.
        Pipeline invalid;
        invalid.append(registry.create("postprocess.hu"));
        invalid.appendExporter(outputs.create("export.volume.raw"));
        reject([&] { invalid.run(v, c); });
        require(values == original, "preflight mutated input");
        reject([&] { makeHuConversionStage(0); });
        values.assign(24, .02f);
        values[0] = 0;
        v.host_data = values.data();
        Pipeline hu;
        hu.append(registry.create("postprocess.hu"));
        hu.run(v, c);
        require(v.domain == Domain::HounsfieldUnits && std::abs(values[0] + 1000) < 1e-3 && std::abs(values[1]) < 1e-3, "HU reference conversion mismatch");
        v.domain = Domain::AttenuationMmInv;
        values = original;
        v.host_data = values.data();
        int resets = 0;
        Pipeline gpu;
        gpu.append(std::make_unique<AddStage>(Location::Device, resets));
        gpu.append(std::make_unique<AddStage>(Location::Device, resets));
        gpu.appendExporter(makeRawFloatExporter());
        reject([&] { gpu.run(v, c); });
        require(values == original, "missing-transfer preflight mutated input");
        Transfer transfer;
        c.transfer = &transfer;
        gpu.run(v, c);
        require(transfer.uploads == 1 && transfer.downloads == 1 && resets == 2, "device stages did not stay resident/reset");
        for (size_t i = 0; i < values.size(); ++i)
            require(values[i] == original[i] + 2, "device chain output mismatch");
        c.transfer = nullptr;
        Pipeline failed;
        auto stage = std::make_unique<AddStage>(Location::Host, resets);
        auto* access = stage.get();
        access->fail = true;
        failed.append(std::move(stage));
        reject([&] { failed.run(v, c); });
        require(resets == 3, "failed stage not reset");
        access->fail = false;
        failed.run(v, c);
        require(resets == 4, "pipeline could not run again after failure");
        v.elements = 23;
        reject([&] { defaults.run(v, c); });
        v.elements = 24;
        std::cout << "Postprocess export compatibility, HU/domain guards, registry, device residency and failure lifecycle passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
