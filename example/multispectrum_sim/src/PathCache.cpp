#include "PathCache.hpp"
#include <fstream>
#include <memory>
#include <stdexcept>
namespace yk::spectral {
bool PathCacheWriter::open(const std::filesystem::path& path,std::uint32_t views,std::uint32_t u,std::uint32_t v,std::uint32_t materials,std::string& error){try{owned_=std::make_unique<std::ofstream>(path,std::ios::binary);if(!*owned_)throw std::runtime_error("无法创建路径积分缓存: "+path.string());PathCacheHeader h;h.views=views;h.detector_u=u;h.detector_v=v;h.material_count=materials;owned_->write(reinterpret_cast<const char*>(&h),sizeof(h));stream_=owned_.get();written_=0;return true;}catch(const std::exception& e){error=e.what();return false;}}
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
        if (!*owned_ || header_.magic != PathCacheHeader{}.magic || header_.version != 1 ||
            header_.views == 0 || header_.detector_u == 0 || header_.detector_v == 0 || header_.material_count == 0)
            throw std::runtime_error("路径积分缓存头无效");
        header = header_; read_ = 0; return true;
    } catch (const std::exception& e) { error = e.what(); close(); return false; }
}

bool PathCacheReader::readView(std::vector<float>& paths, std::string& error)
{
    if (!owned_) { error = "路径积分缓存未打开"; return false; }
    if (read_ >= header_.views) { error = "路径积分缓存视图已读完"; return false; }
    const std::size_t count = static_cast<std::size_t>(header_.material_count) * header_.detector_u * header_.detector_v;
    paths.resize(count);
    owned_->read(reinterpret_cast<char*>(paths.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!*owned_) { error = "读取路径积分缓存失败"; return false; }
    ++read_; return true;
}

void PathCacheReader::close() { if (owned_) owned_->close(); owned_.reset(); read_ = 0; }
}
