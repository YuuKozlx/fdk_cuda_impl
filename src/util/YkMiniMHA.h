#pragma once
#include <fstream>
#include <string>
#include <sstream>
#include <type_traits>
#include <stdexcept>
namespace minimha
{
    struct VolumeDesc
    {
        int width;
        int height;
        int depth;
        double spacing[3] = { 1.0, 1.0, 1.0 };
    };
    // =========================
    // 🔥 类型映射（核心）
    // =========================
    template<typename T>
    struct MetaImageType;
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
        // =========================
        // 🔥 主函数（模板版）
        // =========================
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
        // =========================
        // 1️⃣ 写 header
        // =========================
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
            << desc.spacing[0] << " "
            << desc.spacing[1] << " "
            << desc.spacing[2] << "\n";
        // ⭐ 必加（更标准）
        header << "BinaryData = True\n";
        header << "BinaryDataByteOrderMSB = False\n";
        header << "HeaderSize = 0\n";
        header << "ElementDataFile = LOCAL\n\n";
        std::string header_str = header.str();
        ofs.write(header_str.c_str(), header_str.size());
        // =========================
        // 2️⃣ 写数据
        // =========================
        size_t voxel_count =
            (size_t)desc.width *
            desc.height *
            desc.depth;
        size_t data_size = voxel_count * sizeof(T);
        ofs.write(reinterpret_cast<const char*>(data), data_size);
        return ofs.good();
    }
} // namespace minimha

namespace minibigtiff
{
    // ── 类型映射 ──────────────────────────────────────────────────
    template<typename T> struct TiffSampleInfo;
#define DEF_TIFF_TYPE(T, bits, sfmt) \
    template<> struct TiffSampleInfo<T> { \
        static constexpr uint16_t bitsPerSample = bits; \
        static constexpr uint16_t sampleFormat  = sfmt; \
    };
    DEF_TIFF_TYPE(uint8_t, 8, 1)   // MET_UCHAR  → uint
        DEF_TIFF_TYPE(uint16_t, 16, 1)   // MET_USHORT → uint
        DEF_TIFF_TYPE(int16_t, 16, 2)   // MET_SHORT  → int  (HU)
        DEF_TIFF_TYPE(int32_t, 32, 2)   // MET_INT    → int
        DEF_TIFF_TYPE(float, 32, 3)   // MET_FLOAT  → float
        DEF_TIFF_TYPE(double, 64, 3)   // MET_DOUBLE → float
#undef DEF_TIFF_TYPE

        struct VolumeDesc {
        int    width;
        int    height;
        int    depth;
        double spacingXY = 1.0;  // mm/pixel，写入 XResolution/YResolution
        double spacingZ = 1.0;  // mm/slice，写入 ImageDescription (ImageJ)
    };

    // ── 小端写辅助 ────────────────────────────────────────────────
    namespace detail {
        inline void w16(std::ofstream& f, uint16_t v) { f.write((char*)&v, 2); }
        inline void w32(std::ofstream& f, uint32_t v) { f.write((char*)&v, 4); }
        inline void w64(std::ofstream& f, uint64_t v) { f.write((char*)&v, 8); }

        // BigTIFF IFD entry = 20 bytes
        // tag(2) + type(2) + count(8) + value_or_offset(8)
        // type: 3=SHORT(u16), 4=LONG(u32), 16=LONG8(u64), 5=RATIONAL, 2=ASCII
        inline void entry_short(std::ofstream& f, uint16_t tag, uint16_t val) {
            w16(f, tag); w16(f, 3); w64(f, 1);
            w16(f, val); f.write("\0\0\0\0\0\0", 6); // pad to 8 bytes
        }
        inline void entry_long(std::ofstream& f, uint16_t tag, uint32_t val) {
            w16(f, tag); w16(f, 4); w64(f, 1);
            w32(f, val); w32(f, 0);
        }
        inline void entry_long8(std::ofstream& f, uint16_t tag, uint64_t val) {
            w16(f, tag); w16(f, 16); w64(f, 1);
            w64(f, val);
        }
        // RATIONAL: value stored externally, pass offset
        inline void entry_rational(std::ofstream& f, uint16_t tag, uint64_t offset) {
            w16(f, tag); w16(f, 5); w64(f, 1);
            w64(f, offset);
        }
        // ASCII: count includes null terminator
        inline void entry_ascii(std::ofstream& f, uint16_t tag,
            uint64_t count, uint64_t offset) {
            w16(f, tag); w16(f, 2); w64(f, count);
            w64(f, offset);
        }
    }

    template<typename T>
    inline bool save_bigtiff(
        const std::string& filename,
        const T* data,
        const VolumeDesc& desc)
    {
        static_assert(std::is_trivially_copyable<T>::value, "T must be POD");
        using Info = TiffSampleInfo<T>;

        std::ofstream f(filename, std::ios::binary);
        if (!f) return false;

        const int W = desc.width;
        const int H = desc.height;
        const int D = desc.depth;
        const uint64_t sliceBytes = (uint64_t)W * H * sizeof(T);

        // ── BigTIFF Header (16 bytes) ─────────────────────────────
        // 偏移   内容
        //  0-1   'II' (little endian)
        //  2-3   43  (BigTIFF magic)
        //  4-5   8   (offset field size = 8 bytes)
        //  6-7   0   (reserved)
        //  8-15  first IFD offset (uint64, placeholder)
        f.write("II", 2);
        detail::w16(f, 43);   // BigTIFF
        detail::w16(f, 8);    // offset width
        detail::w16(f, 0);    // reserved
        const uint64_t hdr_ifd_pos = 8;
        detail::w64(f, 0);    // placeholder for first IFD offset

        // ── 元数据准备 ────────────────────────────────────────────
        // Resolution 有理数：res_numer / res_denom pixels/mm
        const uint32_t res_numer = 1000000u;
        const uint32_t res_denom =
            static_cast<uint32_t>(desc.spacingXY * 1000000.0 + 0.5);

        // ImageJ spacing tag（Z 方向）
        std::string imgDesc = "spacing=" + std::to_string(desc.spacingZ) + "\n";
        uint64_t imgDescCount = imgDesc.size() + 1; // 含 null

        // ── 每页布局计算 ──────────────────────────────────────────
        // Tags（升序）：
        //   ImageWidth ImageLength BitsPerSample Compression
        //   PhotometricInterp ImageDescription StripOffsets SamplesPerPixel
        //   RowsPerStrip StripByteCounts XResolution YResolution
        //   ResolutionUnit SampleFormat PageNumber（多页时）
        // 共 14 或 15 个
        const bool multiPage = (D > 1);
        const uint64_t nTags = multiPage ? 15 : 14;

        // BigTIFF IFD 结构：
        //   entry_count(8) + nTags×20 + next_ifd_offset(8)
        const uint64_t ifdSize = 8 + nTags * 20 + 8;

        // extra data per page:
        //   XRes rational(8) + YRes rational(8) + imgDesc(imgDescCount) + align
        uint64_t imgDescPadded = (imgDescCount + 1) & ~1ULL;
        const uint64_t extraSize = 8 + 8 + imgDescPadded; // 16 + desc

        // 记录每页各区域偏移
        std::vector<uint64_t> stripOff(D), ifdOff(D), xresOff(D), imgDescOff(D);

        uint64_t cursor = 16; // header = 16 bytes

        for (int z = 0; z < D; ++z) {
            stripOff[z] = cursor;           cursor += sliceBytes;
            ifdOff[z] = cursor;           cursor += ifdSize;
            xresOff[z] = cursor;
            imgDescOff[z] = cursor + 16;      cursor += extraSize;
        }

        // ── 回填 first IFD offset ─────────────────────────────────
        f.seekp((std::streamoff)hdr_ifd_pos);
        detail::w64(f, ifdOff[0]);

        // ── 写每页 strip + IFD + extra ────────────────────────────
        for (int z = 0; z < D; ++z)
        {
            // 1. strip data
            f.seekp((std::streamoff)stripOff[z]);
            f.write(reinterpret_cast<const char*>(data + (uint64_t)z * W * H),
                (std::streamsize)sliceBytes);

            // 2. IFD
            f.seekp((std::streamoff)ifdOff[z]);
            detail::w64(f, nTags); // entry count (uint64 in BigTIFF)

            // Tags 必须按 tag 值升序排列
            detail::entry_long(f, 0x0100, (uint32_t)W);              // ImageWidth
            detail::entry_long(f, 0x0101, (uint32_t)H);              // ImageLength
            detail::entry_short(f, 0x0102, Info::bitsPerSample);      // BitsPerSample
            detail::entry_short(f, 0x0103, 1);                        // Compression=none
            detail::entry_short(f, 0x0106, 1);                        // PhotometricInterp
            detail::entry_ascii(f, 0x010E, imgDescCount, imgDescOff[z]); // ImageDescription
            detail::entry_long8(f, 0x0111, stripOff[z]);              // StripOffsets
            detail::entry_short(f, 0x0115, 1);                        // SamplesPerPixel
            detail::entry_long(f, 0x0116, (uint32_t)H);              // RowsPerStrip
            detail::entry_long8(f, 0x0117, sliceBytes);               // StripByteCounts
            detail::entry_rational(f, 0x011A, xresOff[z]);             // XResolution
            detail::entry_rational(f, 0x011B, xresOff[z] + 8);        // YResolution
            detail::entry_short(f, 0x0128, 1);                        // ResolutionUnit=none
            detail::entry_short(f, 0x0153, Info::sampleFormat);       // SampleFormat
            if (multiPage) {
                // PageNumber: tag=0x0146, two SHORT values (page, total)
                // 在 BigTIFF 中 8-byte value 区直接存两个 SHORT + padding
                detail::w16(f, 0x0146); detail::w16(f, 3); detail::w64(f, 2);
                detail::w16(f, (uint16_t)z);
                detail::w16(f, (uint16_t)D);
                f.write("\0\0\0\0", 4); // pad to 8 bytes
            }

            // next IFD
            detail::w64(f, (z + 1 < D) ? ifdOff[z + 1] : 0ULL);

            // 3. extra: rational + imgDesc
            f.seekp((std::streamoff)xresOff[z]);
            detail::w32(f, res_numer); detail::w32(f, res_denom); // XRes
            detail::w32(f, res_numer); detail::w32(f, res_denom); // YRes
            f.write(imgDesc.c_str(), (std::streamsize)imgDesc.size());
            f.put('\0');
            if (imgDesc.size() % 2 == 0) f.put('\0'); // 偶数对齐
        }

        return f.good();
    }

} // namespace minibigtiff


namespace minitiff
{
    // ── 类型 → TIFF SampleFormat + BitsPerSample ──────────────────
    template<typename T> struct TiffSampleInfo;
    // SampleFormat: 1=uint, 2=int, 3=float
#define DEF_TIFF_TYPE(T, bits, sfmt) \
    template<> struct TiffSampleInfo<T> { \
        static constexpr uint16_t bitsPerSample = bits; \
        static constexpr uint16_t sampleFormat  = sfmt; \
    };
    DEF_TIFF_TYPE(uint8_t, 8, 1)
        DEF_TIFF_TYPE(uint16_t, 16, 1)
        DEF_TIFF_TYPE(int16_t, 16, 2)
        DEF_TIFF_TYPE(int32_t, 32, 2)
        DEF_TIFF_TYPE(float, 32, 3)
        DEF_TIFF_TYPE(double, 64, 3)
#undef DEF_TIFF_TYPE

        // ── 小端写辅助 ─────────────────────────────────────────────────
        namespace detail {
        inline void write16(std::ofstream& f, uint16_t v) {
            f.write(reinterpret_cast<const char*>(&v), 2);
        }
        inline void write32(std::ofstream& f, uint32_t v) {
            f.write(reinterpret_cast<const char*>(&v), 4);
        }
        // IFD entry: tag(2) type(2) count(4) value_or_offset(4)
        // type: 3=SHORT(uint16), 4=LONG(uint32), 5=RATIONAL, 11=FLOAT
        inline void ifd_short(std::ofstream& f,
            uint16_t tag, uint16_t val)
        {
            write16(f, tag);
            write16(f, 3);      // SHORT
            write32(f, 1);
            write16(f, val);
            write16(f, 0);      // padding to 4 bytes
        }
        inline void ifd_long(std::ofstream& f,
            uint16_t tag, uint32_t val)
        {
            write16(f, tag);
            write16(f, 4);      // LONG
            write32(f, 1);
            write32(f, val);
        }
        // RATIONAL = two LONG (numerator/denominator)
        // value stored externally, pass offset
        inline void ifd_rational(std::ofstream& f,
            uint16_t tag, uint32_t offset)
        {
            write16(f, tag);
            write16(f, 5);      // RATIONAL
            write32(f, 1);
            write32(f, offset);
        }
    }

    struct VolumeDesc {
        int width;
        int height;
        int depth;
        double spacingXY = 1.0; // mm，用于 XResolution/YResolution
        double spacingZ = 1.0; // mm，写入 ImageDescription
    };

    // ── 主函数 ────────────────────────────────────────────────────
    // 每个 Z 层写成一个 TIFF page（directory），ITK/ImageJ 均可读
    template<typename T>
    inline bool save_tiff(
        const std::string& filename,
        const T* data,
        const VolumeDesc& desc)
    {
        static_assert(std::is_trivially_copyable<T>::value,
            "T must be POD");
        using Info = TiffSampleInfo<T>;

        std::ofstream f(filename, std::ios::binary);
        if (!f) return false;

        const int W = desc.width;
        const int H = desc.height;
        const int D = desc.depth;
        const size_t sliceBytes = (size_t)W * H * sizeof(T);

        // ── TIFF header (8 bytes) ──────────────────────────────
        // 'II' = little endian
        f.write("II", 2);
        detail::write16(f, 42);     // magic
        // offset of first IFD — will be patched below
        // 先占位，之后 seekp 回来写
        uint32_t firstIFDOffset_pos = 4;
        detail::write32(f, 0);      // placeholder

        // ── 预先计算每页的布局 ─────────────────────────────────
        // 结构（每页）：
        //   [rational data: 8 bytes for XRes, 8 bytes for YRes]
        //   [strip data: sliceBytes]
        //   [IFD: 2 + N*12 + 4 bytes]
        //
        // 我们按 page 顺序写，记录每页 IFD 的 offset

        // Resolution = 1/spacingXY (pixels per mm)
        // 用有理数表示: numerator=1000000, denominator=spacingXY*1000000
        // 避免浮点，用整数比
        const uint32_t res_numer = 1000000u;
        const uint32_t res_denom =
            static_cast<uint32_t>(desc.spacingXY * 1000000.0 + 0.5);

        // ImageDescription for Z spacing (ImageJ convention)
        // ImageJ 读取 "spacing=X\n" 来获得 Z spacing（单位同 XY）
        std::string imgDesc =
            "spacing=" + std::to_string(desc.spacingZ) + "\n";

        // IFD entry count
        // Tags: ImageWidth ImageLength BitsPerSample Compression
        //       PhotometricInterpretation StripOffsets SamplesPerPixel
        //       RowsPerStrip StripByteCounts XResolution YResolution
        //       ResolutionUnit SampleFormat ImageDescription
        //       (PageNumber if multi-page)
        // 共 15 tags（当 D>1 时含 PageNumber）
        const bool multiPage = (D > 1);
        const int  nTags = multiPage ? 15 : 14;
        // IFD size = 2 + nTags*12 + 4
        const uint32_t ifdSize = 2 + nTags * 12 + 4;

        // 额外数据区（rational + ImageDescription）紧接在 IFD 之后
        // rational: 2 × 8 = 16 bytes
        // imgDesc: imgDesc.size() bytes (可能需要对齐到偶数)
        uint32_t imgDescPadded =
            (uint32_t)((imgDesc.size() + 1) & ~1u); // 对齐到偶数

        // 每页总占用 = rational(16) + stripData(sliceBytes) + IFD + imgDesc
        // 但我们把 rational 和 imgDesc 放在 IFD 之后（extra data）
        // Layout per page:
        //   [strip data]  sliceBytes
        //   [IFD]         ifdSize
        //   [extra]       16 + imgDescPadded

        uint32_t extraSize = 16 + imgDescPadded;
        // 注意 extraSize 中16 = 2 rationals × 8 bytes each

        // 当前文件写入位置（从 header 8 bytes 开始）
        uint32_t cursor = 8;

        std::vector<uint32_t> ifdOffsets(D);
        std::vector<uint32_t> stripOffsets(D);
        std::vector<uint32_t> xresOffsets(D); // rational data offset
        std::vector<uint32_t> imgDescOffsets(D);

        // ── 写每个页 ───────────────────────────────────────────
        for (int z = 0; z < D; ++z)
        {
            // 1. strip data
            stripOffsets[z] = cursor;
            f.write(reinterpret_cast<const char*>(data + (size_t)z * W * H),
                sliceBytes);
            cursor += (uint32_t)sliceBytes;

            // 2. IFD
            ifdOffsets[z] = cursor;
            cursor += ifdSize;

            // 3. extra: rational(16) + imgDesc
            xresOffsets[z] = cursor;        // XResolution rational
            // YResolution rational at cursor+8
            imgDescOffsets[z] = cursor + 16;
            cursor += extraSize;
        }

        // ── 回填 first IFD offset ──────────────────────────────
        f.seekp(firstIFDOffset_pos);
        detail::write32(f, ifdOffsets[0]);

        // ── 写 IFD + extra，每页一次 ──────────────────────────
        for (int z = 0; z < D; ++z)
        {
            // seek 到 IFD 位置
            f.seekp(ifdOffsets[z]);

            // entry count
            detail::write16(f, (uint16_t)nTags);

            // Tags（必须升序排列）
            // 0x0100 ImageWidth
            detail::ifd_long(f, 0x0100, (uint32_t)W);
            // 0x0101 ImageLength
            detail::ifd_long(f, 0x0101, (uint32_t)H);
            // 0x0102 BitsPerSample
            detail::ifd_short(f, 0x0102, Info::bitsPerSample);
            // 0x0103 Compression = 1 (none)
            detail::ifd_short(f, 0x0103, 1);
            // 0x0106 PhotometricInterpretation = 1 (BlackIsZero)
            detail::ifd_short(f, 0x0106, 1);
            // 0x010E ImageDescription
            {
                // ASCII type=2
                detail::write16(f, 0x010E);
                detail::write16(f, 2);
                detail::write32(f, (uint32_t)imgDesc.size() + 1); // +1 for null
                detail::write32(f, imgDescOffsets[z]);
            }
            // 0x0111 StripOffsets
            detail::ifd_long(f, 0x0111, stripOffsets[z]);
            // 0x0115 SamplesPerPixel = 1
            detail::ifd_short(f, 0x0115, 1);
            // 0x0116 RowsPerStrip = H（整页一个 strip）
            detail::ifd_long(f, 0x0116, (uint32_t)H);
            // 0x0117 StripByteCounts
            detail::ifd_long(f, 0x0117, (uint32_t)sliceBytes);
            // 0x011A XResolution
            detail::ifd_rational(f, 0x011A, xresOffsets[z]);
            // 0x011B YResolution
            detail::ifd_rational(f, 0x011B, xresOffsets[z] + 8);
            // 0x0128 ResolutionUnit = 3 (centimeter) — mm→cm 转换见下
            // 实际用 mm 时习惯写 ResolutionUnit=1 (No absolute unit)
            // 这样 spacing 语义完全由 ImageDescription 给出
            detail::ifd_short(f, 0x0128, 1);
            // 0x0153 SampleFormat
            detail::ifd_short(f, 0x0153, Info::sampleFormat);
            // 0x0146 PageNumber（仅多页时）
            if (multiPage) {
                detail::write16(f, 0x0146);
                detail::write16(f, 3);       // SHORT
                detail::write32(f, 2);       // count=2 (page, total)
                detail::write16(f, (uint16_t)z);
                detail::write16(f, (uint16_t)D);
            }

            // Next IFD offset
            if (z + 1 < D)
                detail::write32(f, ifdOffsets[z + 1]);
            else
                detail::write32(f, 0); // end

            // ── extra data ────────────────────────────────────
            f.seekp(xresOffsets[z]);
            // XResolution rational
            detail::write32(f, res_numer);
            detail::write32(f, res_denom);
            // YResolution rational（同值）
            detail::write32(f, res_numer);
            detail::write32(f, res_denom);
            // ImageDescription
            f.write(imgDesc.c_str(), imgDesc.size());
            f.put('\0');
            // 对齐填充
            if (imgDesc.size() % 2 == 0) f.put('\0');
        }

        return f.good();
    }

} // namespace minitiff