#include "pipeline/BuiltinPluginInfo.hpp"
#include "image/GridMetadata.hpp"
#include "recon/postprocess/PostprocessPipeline.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace yk::spectral::postprocess {
namespace {
    void check(bool ok, const std::string& message)
    {
        if (!ok)
            throw std::runtime_error(message);
    }
    void finiteValues(const ReadOnlyVolume& v)
    {
        for (size_t i = 0; i < v.elements; ++i)
            if (!std::isfinite(v.host_data[i]))
                throw std::runtime_error("nonfinite reconstruction output");
    }
    void writeLe16(std::ofstream& out, std::uint16_t value)
    {
        const char bytes[2] = { static_cast<char>(value & 0xffu),
            static_cast<char>((value >> 8u) & 0xffu) };
        out.write(bytes, 2);
    }

    void writeLe32(std::ofstream& out, std::uint32_t value)
    {
        const char bytes[4] = { static_cast<char>(value & 0xffu),
            static_cast<char>((value >> 8u) & 0xffu),
            static_cast<char>((value >> 16u) & 0xffu),
            static_cast<char>((value >> 24u) & 0xffu) };
        out.write(bytes, 4);
    }

    // 写 Windows 常见的 8-bit 灰度 BMP。图像只是诊断输出，原始 float32
    // 体数据仍是数值比较和后处理的唯一依据。
    void writeBmp(const std::filesystem::path& path, int width, int height,
        const std::vector<float>& values, float low, float high)
    {
        check(width > 0 && height > 0 && values.size() == static_cast<std::size_t>(width) * height, "切片尺寸无效");
        if (!(high > low))
            high = low + 1.f;
        const std::size_t row_bytes = (static_cast<std::size_t>(width) + 3u) & ~3u;
        const std::uint32_t pixel_bytes = static_cast<std::uint32_t>(row_bytes * height);
        std::ofstream out(path, std::ios::binary);
        check(static_cast<bool>(out), "无法创建切片图像: " + path.string());
        writeLe16(out, 0x4d42u);
        writeLe32(out, 14u + 40u + 256u * 4u + pixel_bytes);
        writeLe16(out, 0);
        writeLe16(out, 0);
        writeLe32(out, 14u + 40u + 256u * 4u);
        writeLe32(out, 40u);
        writeLe32(out, static_cast<std::uint32_t>(width));
        writeLe32(out, static_cast<std::uint32_t>(height));
        writeLe16(out, 1);
        writeLe16(out, 8);
        writeLe32(out, 0);
        writeLe32(out, pixel_bytes);
        writeLe32(out, 2835);
        writeLe32(out, 2835);
        writeLe32(out, 256);
        writeLe32(out, 0);
        for (std::uint32_t i = 0; i < 256; ++i) {
            const char color[4] = { static_cast<char>(i), static_cast<char>(i),
                static_cast<char>(i), 0 };
            out.write(color, 4);
        }
        std::vector<char> row(row_bytes, 0);
        for (int y = height - 1; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                const float t = std::clamp((values[static_cast<std::size_t>(y) * width + x] - low) / (high - low), 0.f, 1.f);
                row[static_cast<std::size_t>(x)] = static_cast<char>(std::lround(t * 255.f));
            }
            out.write(row.data(), static_cast<std::streamsize>(row.size()));
        }
        check(static_cast<bool>(out), "写入切片图像失败: " + path.string());
    }

    void writeSlices(const std::filesystem::path& prefix, const float* volume,
        int nx, int ny, int nz)
    {
        float low = std::numeric_limits<float>::infinity();
        float high = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < size_t(nx) * ny * nz; ++i) {
            const float v = volume[i];
            if (std::isfinite(v)) {
                low = std::min(low, v);
                high = std::max(high, v);
            }
        }
        check(std::isfinite(low) && std::isfinite(high), "重建体没有有限数值");
        const int z = nz / 2, y = ny / 2, x = nx / 2;
        std::vector<float> axial(static_cast<std::size_t>(nx) * ny);
        std::vector<float> coronal(static_cast<std::size_t>(nx) * nz);
        std::vector<float> sagittal(static_cast<std::size_t>(ny) * nz);
        for (int yy = 0; yy < ny; ++yy)
            for (int xx = 0; xx < nx; ++xx)
                axial[static_cast<std::size_t>(yy) * nx + xx] = volume[(static_cast<std::size_t>(z) * ny + yy) * nx + xx];
        for (int zz = 0; zz < nz; ++zz)
            for (int xx = 0; xx < nx; ++xx)
                coronal[static_cast<std::size_t>(zz) * nx + xx] = volume[(static_cast<std::size_t>(zz) * ny + y) * nx + xx];
        for (int zz = 0; zz < nz; ++zz)
            for (int yy = 0; yy < ny; ++yy)
                sagittal[static_cast<std::size_t>(zz) * ny + yy] = volume[(static_cast<std::size_t>(zz) * ny + yy) * nx + x];
        writeBmp(prefix.string() + "-axial.bmp", nx, ny, axial, low, high);
        writeBmp(prefix.string() + "-coronal.bmp", nx, nz, coronal, low, high);
        writeBmp(prefix.string() + "-sagittal.bmp", ny, nz, sagittal, low, high);
    }

    class RawFloatExporter final : public IExporter {
    public:
        const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::RawOutput);}
        bool acceptsDomain(Domain d) const override { return d == Domain::AttenuationMmInv; } // Existing RAW/UI contract is attenuation.
        void write(const ReadOnlyVolume& v, const Context& c) override
        {
            finiteValues(v);
            check(!c.volume_file.empty(), "empty reconstruction output path");
            if (!c.volume_file.parent_path().empty())
                std::filesystem::create_directories(c.volume_file.parent_path());
            check(v.elements <= size_t(std::numeric_limits<std::streamsize>::max()) / sizeof(float), "RAW output too large");
            std::ofstream file(c.volume_file, std::ios::binary);
            check(bool(file), "cannot create reconstruction output");
            file.write(reinterpret_cast<const char*>(v.host_data), std::streamsize(v.elements * sizeof(float)));
            file.close();
            check(bool(file), "cannot write reconstruction output");
            image::writeGridMetadata(c.volume_file,v.grid);
        }
    };
    class OrthogonalBmpExporter final : public IExporter {
    public:
        const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::BmpOutput);}
        bool acceptsDomain(Domain) const override { return true; }
        void write(const ReadOnlyVolume& v, const Context& c) override
        {
            finiteValues(v);
            check(!c.slice_prefix.empty(), "empty slice prefix");
            if (!c.slice_prefix.parent_path().empty())
                std::filesystem::create_directories(c.slice_prefix.parent_path());
            writeSlices(c.slice_prefix, v.host_data, v.grid.dimensions[0], v.grid.dimensions[1], v.grid.dimensions[2]);
        }
    };
}
std::unique_ptr<IExporter> makeRawFloatExporter() { return std::make_unique<RawFloatExporter>(); }
std::unique_ptr<IExporter> makeOrthogonalBmpExporter() { return std::make_unique<OrthogonalBmpExporter>(); }
}
