#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace yk::spectral {
struct PathCacheHeader { std::uint32_t magic=0x594B5043; std::uint32_t version=1; std::uint32_t views=0, detector_u=0, detector_v=0, material_count=0; };
class PathCacheWriter {
public:
    bool open(const std::filesystem::path&, std::uint32_t views, std::uint32_t u,
              std::uint32_t v, std::uint32_t materials, std::string& error,
              bool material_major = false);
    // 按 [view][material][pixel] 写入，允许每种材料分别调用 DLL FP。
    bool writeMaterialView(std::uint32_t view, std::uint32_t material,
                           const float* paths, std::size_t count,
                           std::string& error);
    bool writeView(const std::vector<float>& paths, std::string& error);
    void close();
private: std::ofstream* stream_=nullptr; std::unique_ptr<std::ofstream> owned_; std::uint32_t written_=0; PathCacheHeader header_{};
};

class PathCacheReader {
public:
    bool open(const std::filesystem::path&, PathCacheHeader& header, std::string& error);
    bool readView(std::vector<float>& paths, std::string& error);
    void close();
private:
    std::unique_ptr<std::ifstream> owned_;
    PathCacheHeader header_{};
    std::uint32_t read_=0;
};
}
