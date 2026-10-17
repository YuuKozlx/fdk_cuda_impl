#include "io/MatLabelVolume.hpp"
#include <cctype>
#include <matio.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace yk::spectral {
namespace {
using File = std::unique_ptr<mat_t, decltype(&Mat_Close)>;
using Variable = std::unique_ptr<matvar_t, decltype(&Mat_VarFree)>;
std::size_t count(const std::array<int, 3>& dims) {
    std::size_t n = 1;
    for (int d : dims) {
        if (d <= 0 || n > std::numeric_limits<std::size_t>::max() / std::size_t(d))
            throw std::runtime_error("Invalid volume dimensions");
        n *= d;
    }
    return n;
}
void rejectLfsPointer(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open volume: " + path.string());
    char prefix[43]{}; in.read(prefix, 42);
    if (std::string(prefix).find("version https://git-lfs.github.com/spec/v1") == 0)
        throw std::runtime_error("Volume is a Git LFS pointer. Run git lfs pull to fetch the example data: " + path.string());
}
template<class T> void convert(const matvar_t& variable, std::vector<std::uint8_t>& labels, const OrganLabelMap& organs) {
    const auto* values = static_cast<const T*>(variable.data);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const long double v = values[i];
        // Organ IDs and uint8 labels are small exact integers, never intensities.
        if (!std::isfinite(v) || v < 0 || v > 4294967295.0L || std::floor(v) != v)
            throw std::runtime_error("MAT volume contains a non-integer or invalid organ/label value");
        const auto id = static_cast<std::uint64_t>(v);
        if (organs.empty()) {
            if (id > 255) throw std::runtime_error("MAT organ IDs require a label mapping; expected uint8 material labels");
            labels[i] = static_cast<std::uint8_t>(id);
        } else {
            const auto found = organs.find(id);
            if (found == organs.end()) throw std::runtime_error("Unmapped MAT organ ID: " + std::to_string(id));
            labels[i] = found->second;
        }
    }
}
}
std::vector<std::uint8_t> readMatLabels(const std::filesystem::path& path, const std::array<int,3>& dims, const OrganLabelMap& organs) {
    const auto n = count(dims); rejectLfsPointer(path);
    File file(Mat_Open(path.string().c_str(), MAT_ACC_RDONLY), &Mat_Close);
    if (!file) throw std::runtime_error("Cannot read MAT v5 volume (v7.3/HDF5 is not enabled): " + path.string());
    Variable selected(nullptr, &Mat_VarFree);
    while (Variable item{Mat_VarReadNextInfo(file.get()), &Mat_VarFree}) {
        if (item->rank != 3) continue;
        if (selected) throw std::runtime_error("Ambiguous MAT volume: more than one 3-D array");
        selected = std::move(item);
    }
    if (!selected) throw std::runtime_error("MAT volume has no 3-D array");
    for (int axis = 0; axis < 3; ++axis)
        if (selected->dims[axis] != std::size_t(dims[axis])) throw std::runtime_error("MAT dimensions do not match configured X/Y/Z");
    if (selected->isComplex || selected->class_type < MAT_C_DOUBLE || selected->class_type > MAT_C_UINT64)
        throw std::runtime_error("MAT volume must be a real dense numeric array");
    if (Mat_VarReadDataAll(file.get(), selected.get()) != 0 || !selected->data)
        throw std::runtime_error("Cannot decompress MAT volume");
    std::vector<std::uint8_t> labels(n);
    switch(selected->data_type) {
    case MAT_T_UINT8: convert<std::uint8_t>(*selected,labels,organs); break;
    case MAT_T_INT8: convert<std::int8_t>(*selected,labels,organs); break;
    case MAT_T_UINT16: convert<std::uint16_t>(*selected,labels,organs); break;
    case MAT_T_INT16: convert<std::int16_t>(*selected,labels,organs); break;
    case MAT_T_UINT32: convert<std::uint32_t>(*selected,labels,organs); break;
    case MAT_T_INT32: convert<std::int32_t>(*selected,labels,organs); break;
    case MAT_T_UINT64: convert<std::uint64_t>(*selected,labels,organs); break;
    case MAT_T_INT64: convert<std::int64_t>(*selected,labels,organs); break;
    case MAT_T_SINGLE: convert<float>(*selected,labels,organs); break;
    case MAT_T_DOUBLE: convert<double>(*selected,labels,organs); break;
    default: throw std::runtime_error("Unsupported MAT numeric storage type");
    }
    return labels;
}
void writeMatLabels(const std::filesystem::path& path,const std::array<int,3>& dims,const std::vector<std::uint8_t>& labels) {
    if(labels.size()!=count(dims)) throw std::runtime_error("MAT output dimensions do not match labels");
    if(std::filesystem::exists(path)) throw std::runtime_error("MAT output already exists: "+path.string());
    if(!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    const auto partial=std::filesystem::path(path.string()+".partial");
    if(std::filesystem::exists(partial)) throw std::runtime_error("MAT partial output already exists");
    try {
        File file(Mat_CreateVer(partial.string().c_str(), "YKCBCT uint8 label volume; x fastest", MAT_FT_MAT5), &Mat_Close);
        if(!file) throw std::runtime_error("Cannot create MAT output");
        std::size_t shape[]={std::size_t(dims[0]),std::size_t(dims[1]),std::size_t(dims[2])};
        Variable variable(Mat_VarCreate("labels",MAT_C_UINT8,MAT_T_UINT8,3,shape,const_cast<std::uint8_t*>(labels.data()),MAT_F_DONT_COPY_DATA), &Mat_VarFree);
        if(!variable || Mat_VarWrite(file.get(),variable.get(),MAT_COMPRESSION_ZLIB)!=0) throw std::runtime_error("Cannot write compressed MAT output");
        auto* handle=file.release(); if(Mat_Close(handle)!=0) throw std::runtime_error("Cannot finish MAT output");
        std::filesystem::rename(partial,path);
    } catch(...) { std::error_code error;std::filesystem::remove(partial,error);throw; }
}
std::vector<std::uint8_t> readClientLabels(const std::filesystem::path& path,const std::array<int,3>& dims) {
    auto ext=path.extension().string();std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return char(std::tolower(c));});
    if(ext==".mat") return readMatLabels(path,dims);
    rejectLfsPointer(path);const auto n=count(dims);
    if(std::filesystem::file_size(path)!=n) throw std::runtime_error("RAW uint8 volume size does not match X/Y/Z");
    std::ifstream input(path,std::ios::binary);std::vector<std::uint8_t> labels(n);
    if(!input.read(reinterpret_cast<char*>(labels.data()),n)) throw std::runtime_error("Cannot read RAW label volume");
    return labels;
}
}
