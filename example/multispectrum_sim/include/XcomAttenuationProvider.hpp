#pragma once
#include "AttenuationProvider.hpp"
#include <filesystem>
#include <string_view>
namespace yk::spectral {
struct BuiltinMaterialInfo {
    // 稳定配置标识采用 category.name，例如 tissue.brain。
    const char* id;
    const char* category;
    const char* display_name;
    const char* aliases;
    double density_g_cm3;
    const char* composition; // 例如 "1:0.111902,8:0.888098"
};

// 返回内置组织/材料预设；名称不区分大小写。表中组成来自仓库已有的
// GateMaterials.db，并与 MC-GPU 的 ICRU 材料文件交叉核对。
const BuiltinMaterialInfo* findBuiltinMaterial(std::string_view name);
const BuiltinMaterialInfo* builtinMaterialsBegin();
const BuiltinMaterialInfo* builtinMaterialsEnd();

class XcomAttenuationProvider final : public IMassAttenuationProvider {
public:
    explicit XcomAttenuationProvider(std::filesystem::path data_directory);
    bool query(const MaterialSpec&, const std::vector<double>&,
               std::vector<double>&, std::string& error) const override;
private: std::filesystem::path data_directory_;
};
}
