#include "PathCache.hpp"
#include <fstream>
#include <memory>
#include <stdexcept>
namespace yk::spectral {
bool PathCacheWriter::open(const std::filesystem::path& path,
    std::uint32_t views, std::uint32_t u, std::uint32_t v,
    std::uint32_t materials, std::string& error, bool material_major)
{
    try {
        owned_ = std::make_unique<std::ofstream>(path,
            std::ios::binary | std::ios::trunc);
        if (!*owned_)
            throw std::runtime_error("无法创建路径积分缓存: " + path.string());
        header_ = {};
        header_.version = material_major ? 2u : 1u;
        header_.views = views;
        header_.detector_u = u;
        header_.detector_v = v;
        header_.material_count = materials;
        owned_->write(reinterpret_cast<const char*>(&header_), sizeof(header_));

        // V1 是 CPU 的逐视图布局，需要预留随机写槽位；V2 由 DLL FP
        // 按材料顺序追加，不提前占用完整缓存空间。
        const std::uint64_t bytes = static_cast<std::uint64_t>(views) *
            materials * u * v * sizeof(float);
        if (bytes && !material_major) {
            owned_->seekp(static_cast<std::streamoff>(sizeof(header_) + bytes - 1));
            const char zero = 0;
            owned_->write(&zero, 1);
        }
        if (!*owned_) throw std::runtime_error("初始化路径积分缓存失败");
        owned_->seekp(static_cast<std::streamoff>(sizeof(header_)));
        stream_ = owned_.get();
        written_ = 0;
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool PathCacheWriter::writeMaterialView(std::uint32_t view,
    std::uint32_t material, const float* paths, std::size_t count,
    std::string& error)
{
    if (!stream_) { error = "路径积分缓存未打开"; return false; }
    const std::size_t pixels = static_cast<std::size_t>(header_.detector_u) *
        header_.detector_v;
    if (view >= header_.views || material >= header_.material_count ||
        count != pixels || !paths) {
        error = "路径积分缓存写入范围无效";
        return false;
    }
    const std::uint64_t index = header_.version == 2
        ? (static_cast<std::uint64_t>(material) * header_.views + view) * pixels
        : (static_cast<std::uint64_t>(view) * header_.material_count + material) * pixels;
    stream_->seekp(static_cast<std::streamoff>(sizeof(header_) + index * sizeof(float)));
    stream_->write(reinterpret_cast<const char*>(paths),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (header_.version == 2 && view + 1 == header_.views)
        stream_->flush();
    if (!*stream_) { error = "写入路径积分缓存失败"; return false; }
    return true;
}
bool PathCacheWriter::writeView(const std::vector<float>& paths,std::string& error){
    if(!stream_){error="路径积分缓存未打开";return false;}
    if(written_ >= 0xFFFFFFFFu){error="路径积分缓存视图数量溢出";return false;}
    stream_->write(reinterpret_cast<const char*>(paths.data()),static_cast<std::streamsize>(paths.size()*sizeof(float)));
    if(!*stream_){error="写入路径积分缓存失败";return false;}++written_;return true;
}
void PathCacheWriter::close(){if(owned_){owned_->flush();owned_->close();}stream_=nullptr;owned_.reset();}

bool PathCacheReader::open(const std::filesystem::path& path, PathCacheHeader& header, std::string& error)
{
    try {
        owned_ = std::make_unique<std::ifstream>(path, std::ios::binary);
        if (!*owned_) throw std::runtime_error("无法打开路径积分缓存: " + path.string());
        owned_->read(reinterpret_cast<char*>(&header_), sizeof(header_));
        if (!*owned_ || header_.magic != PathCacheHeader{}.magic ||
            (header_.version != 1 && header_.version != 2) ||
            header_.views == 0 || header_.detector_u == 0 || header_.detector_v == 0 || header_.material_count == 0)
            throw std::runtime_error("路径积分缓存头无效");
        header = header_; read_ = 0; return true;
    } catch (const std::exception& e) { error = e.what(); close(); return false; }
}

bool PathCacheReader::readView(std::vector<float>& paths, std::string& error)
{
    if (!owned_) { error = "路径积分缓存未打开"; return false; }
    if (read_ >= header_.views) { error = "路径积分缓存视图已读完"; return false; }
    const std::size_t pixels = static_cast<std::size_t>(header_.detector_u) * header_.detector_v;
    const std::size_t count = static_cast<std::size_t>(header_.material_count) * pixels;
    paths.resize(count);
    if (header_.version == 1) {
        owned_->read(reinterpret_cast<char*>(paths.data()), static_cast<std::streamsize>(count * sizeof(float)));
        if (!*owned_) { error = "读取路径积分缓存失败"; return false; }
    } else {
        for (std::uint32_t material = 0; material < header_.material_count; ++material) {
            const std::uint64_t index =
                (static_cast<std::uint64_t>(material) * header_.views + read_) * pixels;
            owned_->seekg(static_cast<std::streamoff>(sizeof(header_) + index * sizeof(float)));
            owned_->read(reinterpret_cast<char*>(paths.data() + material * pixels),
                static_cast<std::streamsize>(pixels * sizeof(float)));
            if (!*owned_) { error = "读取材料分块路径积分缓存失败"; return false; }
        }
    }
    ++read_; return true;
}

void PathCacheReader::close() { if (owned_) owned_->close(); owned_.reset(); read_ = 0; }
}
