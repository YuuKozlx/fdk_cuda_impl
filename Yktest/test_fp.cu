#include <cstdio>
#include <fstream>
#include <vector>
#include "FP/YkFPRunner.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkMacro.hpp"


// ----------------------------------------------------------------
// 读取 raw 体积文件
// ----------------------------------------------------------------
namespace fptest {
    static bool read_raw_float(const char* path, std::vector<float>& data) {
        FILE* fp = std::fopen(path, "rb");
        if (!fp) return false;
        size_t n = std::fread(data.data(), sizeof(float), data.size(), fp);
        std::fclose(fp);
        return n == data.size();
    }

    static bool write_raw_float(const char* path, const std::vector<float>& data) {
        FILE* fp = std::fopen(path, "wb");
        if (!fp) return false;
        size_t n = std::fwrite(data.data(), sizeof(float), data.size(), fp);
        std::fclose(fp);
        return n == data.size();
    }


    static bool write_raw_float(const char* path, const float* data, uint64_t element_count) {
        FILE* fp = std::fopen(path, "wb");
        if (!fp) return false;
        size_t n = std::fwrite(data, sizeof(float), element_count, fp);
        std::fclose(fp);
        return n == element_count;
    }
}


