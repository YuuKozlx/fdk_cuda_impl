#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cstring>


extern "C" {
#include "tinytiffwriter.h"   // TinyTIFFWriterSampleFormat、TinyTIFFWriter_*
#include "tinytiffreader.h"   // TinyTIFFReader_*
}


// ============================================================
//  公共体描述（三个命名空间共用）
// ============================================================
namespace miniio
{
    struct VolumeDesc
    {
        int    width = 0;
        int    height = 0;
        int    depth = 0;
        double spacingX = 1.0;  // mm/pixel，对应 mha ElementSpacing[0] / tiff XResolution
        double spacingY = 1.0;  // mm/pixel，对应 mha ElementSpacing[1] / tiff YResolution
        double spacingZ = 1.0;  // mm/slice，对应 mha ElementSpacing[2] / tiff ImageDescription,非标准字段
    };
} // namespace miniio

  // ============================================================
  //  minimha  ── read_mha
  // ============================================================
namespace minimha
{
    using miniio::VolumeDesc;

    template<typename T> struct MetaImageType;

#define DEFINE_METATYPE(cpp_type, meta_str) \
    template<> struct MetaImageType<cpp_type> { \
        static constexpr const char* name = meta_str; \
    };

    DEFINE_METATYPE(int8_t, "MET_CHAR")
        DEFINE_METATYPE(uint8_t, "MET_UCHAR")
        DEFINE_METATYPE(int16_t, "MET_SHORT")
        DEFINE_METATYPE(uint16_t, "MET_USHORT")
        DEFINE_METATYPE(int32_t, "MET_INT")
        DEFINE_METATYPE(uint32_t, "MET_UINT")
        DEFINE_METATYPE(float, "MET_FLOAT")
        DEFINE_METATYPE(double, "MET_DOUBLE")

#undef DEFINE_METATYPE

        // ── save_mha（原版，保持不变）──────────────────────────────
        template<typename T>
    inline bool save_mha(
        const std::string& filename,
        const T* data,
        const VolumeDesc& desc)
    {
        static_assert(std::is_trivially_copyable<T>::value,
            "Data type must be POD");

        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) return false;

        std::ostringstream header;
        header << "ObjectType = Image\n";
        header << "NDims = 3\n";
        header << "DimSize = "
            << desc.width << " "
            << desc.height << " "
            << desc.depth << "\n";
        header << "ElementType = "
            << MetaImageType<T>::name << "\n";
        header << "ElementSpacing = "
            << desc.spacingX << " "
            << desc.spacingY << " "
            << desc.spacingZ << "\n";
        header << "BinaryData = True\n";
        header << "BinaryDataByteOrderMSB = False\n";
        header << "ElementDataFile = LOCAL\n";

        std::string header_str = header.str();
        ofs.write(header_str.c_str(), header_str.size());

        size_t voxel_count =
            (size_t)desc.width * desc.height * desc.depth;
        ofs.write(reinterpret_cast<const char*>(data),
            voxel_count * sizeof(T));
        return ofs.good();
    }

    // ── read_mha ───────────────────────────────────────────────
    // 从 .mha 文件中读取体数据（仅支持 ElementDataFile = LOCAL）。
    //
    // 模板参数 T 必须与文件中的 ElementType 匹配；若不匹配则返回 false。
    // 读取成功后 desc 和 out 被填充，out 的大小 = width×height×depth。
    template<typename T>
    inline bool read_mha(
        const std::string& filename,
        std::vector<T>& out,
        VolumeDesc& desc)
    {
        static_assert(std::is_trivially_copyable<T>::value,
            "Data type must be POD");

        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) return false;

        // ── 1. 逐行解析 header ──────────────────────────────
        bool data_local = false;
        std::string elem_type;

        std::string line;
        while (std::getline(ifs, line))
        {
            // 去掉尾部的 \r（Windows 换行）
            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            // header 结束标志
            if (line.find("ElementDataFile") != std::string::npos)
            {
                auto eq = line.find('=');
                if (eq != std::string::npos)
                {
                    std::string val = line.substr(eq + 1);
                    // 去首尾空格
                    val.erase(0, val.find_first_not_of(" \t"));
                    val.erase(val.find_last_not_of(" \t") + 1);
                    data_local = (val == "LOCAL");
                }
                break; // 数据紧跟其后
            }

            auto eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);

            // 去首尾空格
            auto trim = [](std::string& s) {
                s.erase(0, s.find_first_not_of(" \t"));
                s.erase(s.find_last_not_of(" \t") + 1);
                };
            trim(key); trim(val);

            if (key == "DimSize") {
                std::istringstream ss(val);
                ss >> desc.width >> desc.height >> desc.depth;
            }
            else if (key == "ElementSpacing") {
                std::istringstream ss(val);
                ss >> desc.spacingX >> desc.spacingY >> desc.spacingZ;
            }
            else if (key == "ElementType") {
                elem_type = val;
            }
        }

        // ── 2. 校验 ElementType ─────────────────────────────
        if (!data_local) return false;
        if (elem_type != MetaImageType<T>::name) return false;

        // ── 3. 读取二进制体数据 ─────────────────────────────
        size_t voxel_count =
            (size_t)desc.width * desc.height * desc.depth;
        out.resize(voxel_count);
        ifs.read(reinterpret_cast<char*>(out.data()),
            voxel_count * sizeof(T));

        return (size_t)ifs.gcount() == voxel_count * sizeof(T);
    }

} // namespace minimha


namespace minimhd
{
    using miniio::VolumeDesc;

    // ── 类型映射 ─────────────────────────────────────────────
    template<typename T> struct MhdTypeStr;
#define DEF_MHD_TYPE(T, s) \
    template<> struct MhdTypeStr<T> { static constexpr const char* value = s; };
    DEF_MHD_TYPE(uint8_t, "MET_UCHAR")
        DEF_MHD_TYPE(int8_t, "MET_CHAR")
        DEF_MHD_TYPE(uint16_t, "MET_USHORT")
        DEF_MHD_TYPE(int16_t, "MET_SHORT")
        DEF_MHD_TYPE(uint32_t, "MET_UINT")
        DEF_MHD_TYPE(int32_t, "MET_INT")
        DEF_MHD_TYPE(float, "MET_FLOAT")
        DEF_MHD_TYPE(double, "MET_DOUBLE")
#undef DEF_MHD_TYPE

        // ── 从完整路径中提取文件名（不含目录）─────────────────────
        inline std::string basename(const std::string& path)
    {
        size_t p = path.find_last_of("/\\");
        return (p == std::string::npos) ? path : path.substr(p + 1);
    }

    // ── 从完整路径中提取目录（含末尾斜杠）────────────────────
    inline std::string dirname(const std::string& path)
    {
        size_t p = path.find_last_of("/\\");
        return (p == std::string::npos) ? "" : path.substr(0, p + 1);
    }

    // ── save_mhd ──────────────────────────────────────────────
    // stem 可以是带路径的前缀，比如 "D:/data/recon_test"
    // 会生成 recon_test.mhd + recon_test.raw（同目录）
    template<typename T>
    inline bool save_mhd(
        const std::string& stem,
        const T* data,
        const VolumeDesc& desc)
    {
        static_assert(std::is_trivially_copyable<T>::value, "T must be POD");

        const std::string rawPath = stem + ".raw";
        const std::string mhdPath = stem + ".mhd";
        const size_t totalBytes =
            (size_t)desc.width * desc.height * desc.depth * sizeof(T);

        // 写 raw
        std::ofstream raw(rawPath, std::ios::binary);
        if (!raw) return false;
        raw.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(totalBytes));
        if (!raw.good()) return false;

        // 写 mhd（只需 raw 文件名，不含目录）
        std::ofstream mhd(mhdPath);
        if (!mhd) return false;
        mhd << "ObjectType = Image\n"
            << "NDims = 3\n"
            << "BinaryData = True\n"
            << "BinaryDataByteOrderMSB = False\n"
            << "CompressedData = False\n"
            << "DimSize = "
            << desc.width << " "
            << desc.height << " "
            << desc.depth << "\n"
            << "ElementSpacing = "
            << desc.spacingX << " "
            << desc.spacingY << " "
            << desc.spacingZ << "\n"
            << "ElementType = " << MhdTypeStr<T>::value << "\n"
            << "ElementDataFile = " << basename(rawPath) << "\n";

        return mhd.good();
    }

    // ── read_mhd ──────────────────────────────────────────────
    // 传入 .mhd 文件路径，自动定位同目录下的 .raw
    template<typename T>
    inline bool read_mhd(
        const std::string& mhdPath,
        std::vector<T>& out,
        VolumeDesc& desc)
    {
        static_assert(std::is_trivially_copyable<T>::value, "T must be POD");

        std::ifstream mhd(mhdPath);
        if (!mhd) return false;

        std::string rawFile;
        std::string expectedType = MhdTypeStr<T>::value;
        bool typeOk = false;

        std::string line;
        while (std::getline(mhd, line))
        {
            std::istringstream ss(line);
            std::string key, eq, val;
            ss >> key >> eq;
            if (eq != "=") continue;

            if (key == "DimSize") {
                ss >> desc.width >> desc.height >> desc.depth;
            }
            else if (key == "ElementSpacing") {
                ss >> desc.spacingX >> desc.spacingY >> desc.spacingZ;
            }
            else if (key == "ElementType") {
                ss >> val;
                typeOk = (val == expectedType);
            }
            else if (key == "ElementDataFile") {
                ss >> rawFile;
            }
        }

        if (!typeOk)   return false;
        if (rawFile.empty()) return false;

        // raw 文件与 mhd 同目录
        const std::string rawPath = dirname(mhdPath) + rawFile;
        std::ifstream raw(rawPath, std::ios::binary);
        if (!raw) return false;

        const size_t totalVoxels =
            (size_t)desc.width * desc.height * desc.depth;
        out.resize(totalVoxels);
        raw.read(reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(totalVoxels * sizeof(T)));

        return raw.good() || raw.eof();
    }

} // namespace minimhd



namespace minitiff {
    using miniio::VolumeDesc;

    // =====================================================
    // TIFF type mapping
    // =====================================================
    template<typename T>
    struct TiffType;

    template<> struct TiffType<uint8_t> {
        static constexpr uint16_t bits = 8;
        static constexpr TinyTIFFWriterSampleFormat format = TinyTIFFWriter_UInt;
    };
    template<> struct TiffType<uint16_t> {
        static constexpr uint16_t bits = 16;
        static constexpr TinyTIFFWriterSampleFormat format = TinyTIFFWriter_UInt;
    };
    template<> struct TiffType<int16_t> {
        static constexpr uint16_t bits = 16;
        static constexpr TinyTIFFWriterSampleFormat format = TinyTIFFWriter_Int;
    };
    template<> struct TiffType<float> {
        static constexpr uint16_t bits = 32;
        static constexpr TinyTIFFWriterSampleFormat format = TinyTIFFWriter_Float;
    };

    // =====================================================
    // type constraint
    // =====================================================
    template<typename T>
    struct is_tiff_supported : std::false_type {};

    template<> struct is_tiff_supported<uint8_t> : std::true_type {};
    template<> struct is_tiff_supported<uint16_t> : std::true_type {};
    template<> struct is_tiff_supported<int16_t> : std::true_type {};
    template<> struct is_tiff_supported<float> : std::true_type {};

    // =====================================================
    // save TIFF stack
    // =====================================================
    template<typename T>
    inline bool save_tiff(
        const std::string& stem,
        const T* data,
        const VolumeDesc& desc)
    {
        static_assert(is_tiff_supported<T>::value, "Unsupported TIFF type");

        if (!data) return false;

        const uint32_t W = static_cast<uint32_t>(desc.width);
        const uint32_t H = static_cast<uint32_t>(desc.height);
        const uint32_t D = static_cast<uint32_t>(desc.depth);
        if (W == 0 || H == 0 || D == 0) return false;

        const std::string file = stem;

        TinyTIFFWriterFile* tif = TinyTIFFWriter_open(
            file.c_str(),
            TiffType<T>::bits,
            TiffType<T>::format,
            1,          // samples per pixel
            W, H,
            TinyTIFFWriter_Greyscale);

        if (!tif) return false;

        for (uint32_t z = 0; z < D; ++z)
        {
            const T* src = data + static_cast<size_t>(z) * W * H;
            if (TinyTIFFWriter_writeImage(tif, src) != TINYTIFF_TRUE)
            {
                TinyTIFFWriter_close(tif);
                return false;
            }
        }

        TinyTIFFWriter_close(tif);
        return true;
    }

    // =====================================================
    // load TIFF stack
    // =====================================================
    template<typename T>
    inline bool load_tiff(
        const std::string& stem,
        std::vector<T>& out,
        VolumeDesc& desc)
    {
        static_assert(is_tiff_supported<T>::value, "Unsupported TIFF type");

        const std::string file = stem + ".tiff";

        TinyTIFFReaderFile* tif = TinyTIFFReader_open(file.c_str());
        if (!tif) return false;

        const uint32_t W = TinyTIFFReader_getWidth(tif);
        const uint32_t H = TinyTIFFReader_getHeight(tif);
        if (W == 0 || H == 0) {
            TinyTIFFReader_close(tif);
            return false;
        }

        // 帧数：TinyTIFF 提供直接查询，无需二次打开
        const uint32_t D = TinyTIFFReader_countFrames(tif);
        if (D == 0) {
            TinyTIFFReader_close(tif);
            return false;
        }

        desc.width = static_cast<int>(W);
        desc.height = static_cast<int>(H);
        desc.depth = static_cast<int>(D);

        const size_t sliceVoxels = static_cast<size_t>(W) * H;
        out.resize(sliceVoxels * D);

        // countFrames 会移动内部指针，需重新打开
        TinyTIFFReader_close(tif);
        tif = TinyTIFFReader_open(file.c_str());
        if (!tif) return false;

        uint32_t z = 0;
        do {
            // getSampleData 写入调用方提供的 buffer，sample index = 0
            if (TinyTIFFReader_getSampleData(
                tif,
                out.data() + z * sliceVoxels,
                0) != TINYTIFF_TRUE)
            {
                TinyTIFFReader_close(tif);
                return false;
            }
            ++z;
        } while (TinyTIFFReader_readNextImage(tif) && z < D);

        TinyTIFFReader_close(tif);
        return (z == D);
    }
}