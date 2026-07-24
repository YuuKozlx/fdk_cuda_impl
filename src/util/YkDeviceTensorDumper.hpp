#pragma once

#include <cstdint>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

#include "util/YkIoDump.hpp"

namespace YK::IO {

// 对连续 float32 设备内存的算法无关描述。shape 采用调用方的自然布局：
// 投影可为 [view, v, u]，体数据可为 [z, y, x]；导出器不解释维度含义。
struct DeviceTensorView {
    std::string name;
    const float* d_data = nullptr;
    std::vector<int64_t> shape;
    cudaStream_t stream = nullptr;

    size_t elementCount() const
    {
        size_t count = 1;
        for (const int64_t dim : shape) {
            if (dim <= 0) return 0;
            count *= static_cast<size_t>(dim);
        }
        return count;
    }
    bool valid() const { return !name.empty() && d_data && elementCount() != 0; }
};

struct DeviceTensorDumpConfig {
    std::string output_directory;
    bool enabled = false;
    bool write_metadata = true;
};

// 独立的数据导出器：接受任意算法提供的设备张量视图，写出 raw 和可选 JSON
// 元数据。它不属于 FDK/迭代器，也不主动订阅任何计算过程。
class DeviceTensorDumper {
public:
    bool initialize(const DeviceTensorDumpConfig& config)
    {
        config_ = config;
        return dump_.init(config.output_directory, config.enabled);
    }

    bool dump(const DeviceTensorView& tensor) const
    {
        if (!config_.enabled) return true;
        if (!tensor.valid()) return false;
        const std::string stem = sanitize_(tensor.name);
        bool ok = dump_.dumpDeviceF32(stem + ".raw", tensor.d_data,
            tensor.elementCount(), tensor.stream);
        if (ok && config_.write_metadata)
            ok = dump_.dumpText(stem + ".json", metadata_(tensor));
        return ok;
    }

private:
    static std::string sanitize_(std::string name)
    {
        for (char& c : name)
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) c = '_';
        return name;
    }

    static std::string metadata_(const DeviceTensorView& tensor)
    {
        std::ostringstream os;
        os << "{\n  \"dtype\": \"float32\",\n  \"layout\": \"contiguous\",\n  \"shape\": [";
        for (size_t i = 0; i < tensor.shape.size(); ++i) {
            if (i) os << ", ";
            os << tensor.shape[i];
        }
        os << "],\n  \"element_count\": " << tensor.elementCount() << "\n}\n";
        return os.str();
    }

    DeviceTensorDumpConfig config_{};
    mutable DumpManager dump_{};
};

} // namespace YK::IO
