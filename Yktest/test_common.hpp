#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static bool read_raw_float(const char* path, std::vector<float>& data)
{
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    size_t n = std::fread(data.data(), sizeof(float), data.size(), fp);
    std::fclose(fp);
    return n == data.size();
}

static bool read_raw_float(const char* path, float* data, uint64_t element_count)
{
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    size_t n = std::fread(data, sizeof(float), element_count, fp);
    std::fclose(fp);
    return n == element_count;
}

static bool write_raw_float(const char* path, const std::vector<float>& data)
{
    FILE* fp = std::fopen(path, "wb");
    if (!fp) return false;
    size_t n = std::fwrite(data.data(), sizeof(float), data.size(), fp);
    std::fclose(fp);
    return n == data.size();
}

static bool write_raw_float(const char* path, const float* data, uint64_t element_count)
{
    FILE* fp = std::fopen(path, "wb");
    if (!fp) return false;
    size_t n = std::fwrite(data, sizeof(float), element_count, fp);
    std::fclose(fp);
    return n == element_count;
}

static void printStats(const std::vector<float>& v, const char* tag)
{
    if (v.empty()) { printf("[%s] empty\n", tag); return; }
    float minv = v[0], maxv = v[0], sum = 0.f;
    for (auto x : v) {
        if (x < minv) minv = x;
        if (x > maxv) maxv = x;
        sum += x;
    }
    printf("[%s] min=%.4f  max=%.4f  mean=%.6f  n=%zu\n",
        tag, minv, maxv, sum / (float)v.size(), v.size());
}

struct datapath {
    inline static const std::string test_data_dir =
        R"(G:\Code\fanproj\fdk-test\TestData\)";
};

static std::string dataPath(const std::string& filename)
{
    return datapath::test_data_dir + "/" + filename;
}